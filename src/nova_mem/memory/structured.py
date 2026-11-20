"""System E — structured memory (minimal, supersession-aware).

Pipeline (reproduction_plan.md §4.1):
1. Offline extraction pass over every session with a fixed extractor model
   (part of the architecture; the answer reader stays the primary reader):
   atomic facts -> JSON rows with verbatim evidence spans.
2. SQLite store with supersession: process sessions oldest-to-newest by date;
   on insert, an active memory matching (entity, attribute) with a different
   value is marked superseded (kept for audit; excluded from retrieval).
3. Answer time: dense retrieval over active facts; the prompt renders top-k
   facts WITH their evidence spans, using the OFFICIAL facts template
   (``merge_key_expansion_into_value='replace'``) so E stays inside the
   official prompt family.

Measured per question: extraction tokens/latency, storage bytes (SQLite file),
retrieval latency, generation tokens/latency. Extraction is cached by
(session_id, prompt) so shared filler sessions are extracted once.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Optional

from ..data.loader import Question
from ..models.cache import GenerationCache
from ..models.client import InferenceClient
from ..retrieval import DenseRetriever, Document
from ..utils import get_logger, timer
from . import prompts as P
from .base import AnswerResult, MemorySystem, reader_generate

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fact TEXT NOT NULL,
    entity TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    evidence_span TEXT NOT NULL,
    superseded_by INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ea ON memories(entity, attribute);
"""


class StructuredMemorySystem(MemorySystem):
    def __init__(
        self,
        embedder=None,
        top_k: int = 10,
        include_evidence: bool = True,
        latest_first: bool = True,
        db_path: str | None = None,  # None -> temp file per run
        skip_filtered_sessions: bool = False,
        extract_max_tokens: int = 800,
    ):
        from ..retrieval import _HashEmbedder

        self.name = "E_structured"
        self.embedder = embedder or _HashEmbedder()
        self.top_k = top_k
        self.include_evidence = include_evidence
        self.latest_first = latest_first
        self.db_path = db_path
        # Sandbox continuation option (deviation-tagged): when the cloud backend's
        # content filter deterministically rejects a session's extraction prompt,
        # record the session as extracted-with-zero-facts instead of failing the
        # question. Off by default: the frozen protocol has no such concept.
        self.skip_filtered_sessions = skip_filtered_sessions
        # v2 (2026-11-14, diagnosis-driven): extraction completion budget. The dev-1
        # pilot measured that fact-dense sessions overflow the original 800-token cap
        # (15/50 sessions truncated mid-array -> unparseable). Default stays 800 so
        # the frozen protocol is unchanged; the diagnosis-driven continuation runs
        # pass a larger budget explicitly and record it as a deviation.
        self.extract_max_tokens = extract_max_tokens

    # ------------------------------------------------------------------
    def _extract_session(self, session, extractor, cache) -> tuple[list[dict], dict]:
        prompt = P.NOVA_FACT_EXTRACTION.format(
            session.date,
            P._json_dumps([{"role": m.role, "content": m.content} for m in session.messages]),
        )
        text, t_ext, tok_ext = reader_generate(
            extractor, prompt, cache=cache, max_tokens=self.extract_max_tokens, stage="extract"
        )
        facts = self._parse_facts(text, session)
        return facts, {"timings": t_ext, "tokens": tok_ext, "raw": text}

    @staticmethod
    def _parse_facts(text: str, session) -> list[dict]:
        """Parse the extractor output; tolerate prose-wrapped JSON."""
        candidates = []
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
        candidates.append(text)
        for cand in candidates:
            try:
                arr = json.loads(cand)
            except json.JSONDecodeError:
                continue
            facts = []
            if isinstance(arr, list):
                for el in arr:
                    if not isinstance(el, dict):
                        continue
                    if not all(k in el for k in ("fact", "entity", "attribute", "value")):
                        continue
                    facts.append(
                        {
                            "fact": str(el["fact"]),
                            "entity": str(el["entity"]),
                            "attribute": str(el["attribute"]),
                            "value": str(el.get("value", "")),
                            "timestamp": str(el.get("timestamp") or session.date),
                            "session_id": session.session_id,
                            "evidence_span": str(el.get("evidence_span", "")),
                        }
                    )
                return facts
        return []

    # ------------------------------------------------------------------
    def _build_store(
        self, question: Question, extractor, cache
    ) -> tuple[sqlite3.Connection, Path, dict]:
        temporary_db = self.db_path is None
        if temporary_db:
            handle = tempfile.NamedTemporaryFile(suffix=".sqlite", prefix="nova_mem_", delete=False)
            db_path = Path(handle.name)
            handle.close()
        else:
            db_path = Path(self.db_path)
            if db_path.exists():
                db_path.unlink()

        conn = None
        try:
            conn = sqlite3.connect(str(db_path))
            conn.executescript(_SCHEMA)
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            extract_p_total = 0
            extract_c_total = 0
            n_sessions = 0
            n_facts = 0
            n_superseded = 0
            parse_failures = 0
            n_valid_empty = 0
            n_filter_blocked = 0
            filter_blocked_sessions: list[str] = []
            active: dict[tuple[str, str], tuple[int, str]] = {}
            sessions = sorted(question.sessions, key=lambda item: item.date)

            with timer() as t_extract:
                for session_index, session in enumerate(sessions, 1):
                    logger.info(
                        "E extraction session %d/%d started",
                        session_index,
                        len(sessions),
                    )
                    session_t0 = time.perf_counter()
                    try:
                        facts, meta = self._extract_session(session, extractor, cache)
                    except Exception as e:  # noqa: BLE001
                        if (
                            self.skip_filtered_sessions
                            and type(e).__name__ == "ZAIContentFilteredError"
                        ):
                            facts = []
                            meta = {"timings": {}, "tokens": {}, "raw": ""}
                            n_filter_blocked += 1
                            filter_blocked_sessions.append(session.session_id)
                            logger.warning(
                                "E extraction session %s skipped: content-filter blocked (%d blocked so far)",
                                session.session_id,
                                n_filter_blocked,
                            )
                        else:
                            raise
                    tok = meta["tokens"]
                    extract_p_total += tok.get("extract_prompt", 0)
                    extract_c_total += tok.get("extract_completion", 0)
                    n_sessions += 1
                    raw_text = meta["raw"]
                    if not facts and raw_text.strip():
                        # Distinguish genuine parse failures from valid empty
                        # extractions: the extraction prompt asks for [] when a
                        # session contains no user facts, so an empty JSON array
                        # (possibly prose-wrapped) is a SUCCESS, not a failure.
                        stripped = raw_text.strip()
                        start, end = stripped.find("["), stripped.rfind("]")
                        is_valid_empty = False
                        if start != -1 and end != -1 and end > start:
                            try:
                                arr = json.loads(stripped[start : end + 1])
                                is_valid_empty = isinstance(arr, list) and len(arr) == 0
                            except json.JSONDecodeError:
                                is_valid_empty = False
                        if is_valid_empty:
                            n_valid_empty += 1
                        else:
                            parse_failures += 1
                    for f in facts:
                        cur = conn.execute(
                            "INSERT INTO memories (fact, entity, attribute, value, timestamp, session_id,"
                            " evidence_span, created_at) VALUES (?,?,?,?,?,?,?,?)",
                            (
                                f["fact"],
                                f["entity"],
                                f["attribute"],
                                f["value"],
                                f["timestamp"],
                                f["session_id"],
                                f["evidence_span"],
                                now,
                            ),
                        )
                        rowid = cur.lastrowid
                        key = (f["entity"].strip().lower(), f["attribute"].strip().lower())
                        if key in active and active[key][1] != f["value"]:
                            conn.execute(
                                "UPDATE memories SET superseded_by=? WHERE memory_id=?",
                                (rowid, active[key][0]),
                            )
                            n_superseded += 1
                        active[key] = (rowid, f["value"])
                        n_facts += 1
                    logger.info(
                        "E extraction session %d/%d completed in %.1fs (%d facts; %d parse failures)",
                        session_index,
                        len(sessions),
                        time.perf_counter() - session_t0,
                        len(facts),
                        parse_failures,
                    )
            conn.commit()
        except Exception:
            if conn is not None:
                conn.close()
            if temporary_db:
                db_path.unlink(missing_ok=True)
            raise
        if conn is None:
            raise RuntimeError("Structured memory database was not initialized")
        stats = {
            "extract_s": t_extract["s"],
            "extract_prompt_tokens": extract_p_total,
            "extract_completion_tokens": extract_c_total,
            "n_sessions": n_sessions,
            "n_facts": n_facts,
            "n_superseded": n_superseded,
            "extract_parse_failures": parse_failures,
            "extract_valid_empty": n_valid_empty,
            "n_filter_blocked": n_filter_blocked,
            "filter_blocked_sessions": filter_blocked_sessions,
        }
        return conn, db_path, stats

    # ------------------------------------------------------------------
    def answer(
        self,
        question: Question,
        reader: InferenceClient,
        cache: Optional[GenerationCache] = None,
        extractor: Optional[InferenceClient] = None,
    ) -> AnswerResult:
        extractor = extractor or reader
        conn, db_path, stats = self._build_store(question, extractor, cache)
        try:
            storage_bytes = os.path.getsize(db_path)
            rows = conn.execute(
                "SELECT memory_id, fact, entity, attribute, value, timestamp, session_id, evidence_span,"
                " superseded_by FROM memories WHERE superseded_by IS NULL"
            ).fetchall()
            timings = {"extract_s": stats["extract_s"], "index_s": 0.0}
            tokens = {
                "extract_prompt": stats["extract_prompt_tokens"],
                "extract_completion": stats["extract_completion_tokens"],
            }

            evidence_ids = set(question.evidence_session_ids)
            docs = [
                Document(
                    corpus_id=f"fact_{r[0]}",
                    text=f"{r[1]} (entity: {r[2]}; attribute: {r[3]}: {r[4]}; date: {r[5]})",
                    date=r[5],
                    session_id=r[6],
                    messages=[],
                )
                for r in rows
            ]
            with timer() as t_index:
                retriever = DenseRetriever(self.embedder)
                retriever.build(docs)
            timings["index_s"] = t_index["s"]

            with timer() as t_ret:
                hits = retriever.search(question.question, self.top_k)
            timings["retrieve_s"] = t_ret["s"]

            row_by_id = {r[0]: r for r in rows}
            fact_blocks = []
            for h in hits:
                r = row_by_id[int(h.corpus_id.split("_")[1])]
                block = {
                    "fact": r[1],
                    "entity": r[2],
                    "attribute": r[3],
                    "value": r[4],
                    "date": r[5],
                }
                if self.include_evidence and r[7]:
                    block["evidence_span"] = r[7]
                fact_blocks.append((r[5], block))
            fact_blocks.sort(key=lambda x: x[0], reverse=self.latest_first)

            chunks = [(d, block) for d, block in fact_blocks]
            with timer() as t_render:
                history = P.render_history_json(chunks)
            timings["render_s"] = t_render["s"]

            trace = [
                {
                    "rank": h.rank,
                    "corpus_id": h.corpus_id,
                    "session_id": h.session_id,
                    "score": round(h.score, 6),
                    "is_evidence": h.session_id in evidence_ids,
                }
                for h in hits
            ]
            extras = {
                "n_active_facts": len(rows),
                "n_facts_total": stats["n_facts"],
                "n_superseded": stats["n_superseded"],
                "extract_parse_failures": stats["extract_parse_failures"],
                "extract_valid_empty": stats["extract_valid_empty"],
                "n_sessions": stats["n_sessions"],
                "n_filter_blocked": stats["n_filter_blocked"],
                "filter_blocked_sessions": stats["filter_blocked_sessions"],
            }
            return self._final_answer(
                question,
                history,
                reader,
                cache,
                timings=timings,
                tokens=tokens,
                retrieval_trace=trace,
                storage_bytes=storage_bytes,
                extras=extras,
                template=P.ANSWER_FACTS_REPLACE,
            )
        finally:
            conn.close()
            if self.db_path is None:
                db_path.unlink(missing_ok=True)
