"""Systems A (no-memory), B (full-context), Oracle, C (RAG), D (RAG + rerank).

All rendering and prompts follow the official ``run_generation.py`` conventions:
``json`` history format, date-sorted sessions, ``### Session n`` blocks, and the
verbatim answer templates in ``nova_mem.memory.prompts``.
"""

from __future__ import annotations

from typing import Optional

from ..data.loader import Question
from ..models.cache import GenerationCache
from ..models.client import InferenceClient
from ..retrieval import Document, build_retriever
from ..reranking import Reranker
from ..utils import timer
from . import prompts as P
from .base import (
    AnswerResult,
    MemorySystem,
    chunks_from_sessions,
    reader_generate,
    sort_chunks_by_date,
    trace_from_hits,
)


class NoMemorySystem(MemorySystem):
    """System A — official ``no-retrieval``: the prompt is the question alone.

    Note (scientific decision, see docs/): the official no-retrieval template
    carries NO abstention instruction; we keep official parity rather than
    adding instructions that would deviate from the published protocol.
    """

    def __init__(self, cot: bool = False):
        self.name = "A_no_memory"
        self.cot = cot

    def answer(
        self, question: Question, reader: InferenceClient, cache: Optional[GenerationCache] = None
    ) -> AnswerResult:
        if self.cot:
            # official run_generation.py: template '{}' += 'Answer step by step.'
            prompt = question.question + "Answer step by step."
        else:
            prompt = P.ANSWER_NO_RETRIEVAL.format(question=question.question)
        text, t_gen, tok_gen = reader_generate(reader, prompt, cache=cache)
        return AnswerResult(
            question_id=question.question_id,
            hypothesis=text,
            prompt=prompt,
            retrieval_trace=[],
            timings=t_gen,
            tokens=tok_gen,
        )


class FullContextSystem(MemorySystem):
    """System B — entire history, official ``orig-session`` + ``json`` format.

    Profile 2 (Colab): ~115k-token prompts. ``max_history_chars`` provides a
    documented left-truncation guard (official code truncates to the model's
    context window keeping the FIRST tokens).
    """

    def __init__(self, top_k: Optional[int] = None, max_history_chars: Optional[int] = None):
        self.name = "B_full_context"
        self.top_k = top_k  # official orig-* keeps the LATEST top_k sessions
        self.max_history_chars = max_history_chars

    def answer(
        self, question: Question, reader: InferenceClient, cache: Optional[GenerationCache] = None
    ) -> AnswerResult:
        with timer() as t_index:
            chunks = chunks_from_sessions(question.sessions)
        if self.top_k is not None:
            chunks = chunks[-self.top_k :]  # keep latest (official orig-* behavior)
        chunks = sort_chunks_by_date(chunks)
        history = P.render_history_json(chunks)
        if self.max_history_chars is not None and len(history) > self.max_history_chars:
            history = history[: self.max_history_chars]  # official: keep first tokens
        timings = {"index_s": t_index["s"]}
        evidence = set(question.evidence_session_ids)
        trace = [
            {
                "rank": i + 1,
                "corpus_id": s.session_id,
                "session_id": s.session_id,
                "score": None,
                "is_evidence": s.session_id in evidence,
            }
            for i, s in enumerate(question.sessions)
        ]
        return self._final_answer(
            question,
            history,
            reader,
            cache,
            timings=timings,
            tokens={},
            retrieval_trace=trace,
            extras={"n_sessions": len(chunks)},
        )


class OracleContextSystem(MemorySystem):
    """Reference — evidence sessions only (equivalent to running the official
    pipeline on ``longmemeval_oracle.json``). For abstention questions the
    ``noans_`` pseudo-evidence sessions are used, mirroring the oracle file."""

    def __init__(self):
        self.name = "oracle_context"

    def answer(
        self, question: Question, reader: InferenceClient, cache: Optional[GenerationCache] = None
    ) -> AnswerResult:
        evidence_ids = set(question.evidence_session_ids)
        sessions = [s for s in question.sessions if s.session_id in evidence_ids]
        with timer() as t_index:
            chunks = sort_chunks_by_date(chunks_from_sessions(sessions))
        history = P.render_history_json(chunks)
        trace = [
            {
                "rank": i + 1,
                "corpus_id": s.session_id,
                "session_id": s.session_id,
                "score": None,
                "is_evidence": True,
            }
            for i, s in enumerate(sessions)
        ]
        return self._final_answer(
            question,
            history,
            reader,
            cache,
            timings={"index_s": t_index["s"]},
            tokens={},
            retrieval_trace=trace,
            extras={"n_sessions": len(sessions), "oracle": True},
        )


class RAGSystem(MemorySystem):
    """System C — retrieval-augmented memory (flat retrieval, session granularity).

    Config: retriever kind, top-k, ``user_only`` indexing (official key design
    keeps user-side utterances), optional Chain-of-Note reading with the reader
    model (official ``con``), optional turn granularity.
    """

    def __init__(
        self,
        retriever_kind: str = "bm25",
        embedder=None,
        top_k: int = 5,
        granularity: str = "session",  # 'session' | 'turn'
        user_only_index: bool = True,
        user_only_value: bool = False,
        con: bool = True,
        retriever=None,
        candidate_pool_size: int = 25,
        rrf_k: int = 60,
    ):
        self.name = f"C_rag_{granularity}_{retriever_kind}" + ("_con" if con else "")
        self.retriever = (
            retriever
            if retriever is not None
            else build_retriever(
                retriever_kind,
                embedder=embedder,
                candidate_pool_size=candidate_pool_size,
                rrf_k=rrf_k,
            )
        )
        self.top_k = top_k
        self.granularity = granularity
        self.user_only_index = user_only_index
        self.user_only_value = user_only_value
        self.con = con

    # -- corpus ---------------------------------------------------------
    def _docs(self, question: Question) -> list[Document]:
        from ..retrieval import build_session_corpus, build_turn_corpus

        if self.granularity == "session":
            return build_session_corpus(question, user_only=self.user_only_index)
        return build_turn_corpus(question, user_only=self.user_only_index)

    def _sessions_by_id(self, question: Question) -> dict:
        return {s.session_id: s for s in question.sessions}

    # -- main -----------------------------------------------------------
    def answer(
        self, question: Question, reader: InferenceClient, cache: Optional[GenerationCache] = None
    ) -> AnswerResult:
        timings: dict = {}
        tokens: dict = {}
        evidence_ids = set(question.evidence_session_ids)

        with timer() as t_index:
            docs = self._docs(question)
            self.retriever.build(docs)
        timings["index_s"] = t_index["s"]

        with timer() as t_ret:
            hits = self.retriever.search(question.question, self.top_k)
        timings["retrieve_s"] = t_ret["s"]
        trace = trace_from_hits(hits, evidence_ids)

        sessions = self._sessions_by_id(question)
        chunks = []
        for h in hits:
            if self.granularity == "session":
                s = sessions[h.session_id]
                msgs = P.strip_has_answer(
                    [{"role": m.role, "content": m.content} for m in s.messages]
                )
                if self.user_only_value:
                    msgs = [m for m in msgs if m.get("role") == "user"]
                chunks.append((h.date, msgs))
            else:
                doc = next(d for d in docs if d.corpus_id == h.corpus_id)
                chunks.append((h.date, doc.messages))
        chunks = sort_chunks_by_date(chunks)

        if self.con:
            con_p = con_c = 0
            with timer() as t_con:
                noted = []
                for chunk_date, chunk_entry in chunks:
                    con_prompt = P.CON_PROMPT.format(
                        chunk_date,
                        P._json_dumps(chunk_entry),
                        question.question_date,
                        question.question,
                    )
                    note, _, tok = reader_generate(
                        reader, con_prompt, cache=cache, max_tokens=500, stage="con"
                    )
                    con_p += tok.get("con_prompt", 0)
                    con_c += tok.get("con_completion", 0)
                    noted.append((chunk_date, {"session_summary": note}))
            timings["con_s"] = t_con["s"]
            tokens["con_prompt"] = con_p
            tokens["con_completion"] = con_c
            chunks = noted

        with timer() as t_render:
            history = P.render_history_json(chunks)
        timings["render_s"] = t_render["s"]

        return self._final_answer(
            question,
            history,
            reader,
            cache,
            timings=timings,
            tokens=tokens,
            retrieval_trace=trace,
            extras={"n_retrieved": len(hits), "granularity": self.granularity},
        )


class RAGRerankSystem(RAGSystem):
    """System D — C + cross-encoder rerank: retrieve top-N, rerank, keep top-k'.

    ``token_budget`` (optional): match C's rendered-history character budget by
    accumulating reranked sessions until the budget is reached (the plan's
    token-matched comparison; character proxy documented as a deviation).
    """

    def __init__(
        self,
        reranker: Reranker,
        top_n: int = 25,
        top_k_final: int = 5,
        token_budget_chars: Optional[int] = None,
        **rag_kwargs,
    ):
        super().__init__(**rag_kwargs)
        self.name = "D_rag_rerank_" + self.retriever.name + ("_con" if self.con else "")
        self.reranker = reranker
        self.top_n = top_n
        self.top_k_final = top_k_final
        self.token_budget_chars = token_budget_chars

    def answer(
        self, question: Question, reader: InferenceClient, cache: Optional[GenerationCache] = None
    ) -> AnswerResult:
        timings: dict = {}
        tokens: dict = {}
        evidence_ids = set(question.evidence_session_ids)

        with timer() as t_index:
            docs = self._docs(question)
            self.retriever.build(docs)
        timings["index_s"] = t_index["s"]

        with timer() as t_ret:
            hits = self.retriever.search(question.question, self.top_n)
        timings["retrieve_s"] = t_ret["s"]

        doc_by_id = {d.corpus_id: d for d in docs}
        cand_docs = [doc_by_id[h.corpus_id] for h in hits if h.corpus_id in doc_by_id]
        reranked, rerank_elapsed = self.reranker.rerank(
            question.question, cand_docs, top_k=len(cand_docs)
        )
        timings["rerank_s"] = rerank_elapsed
        trace = trace_from_hits(reranked, evidence_ids)

        sessions = self._sessions_by_id(question)
        chunks = []
        budget = self.token_budget_chars
        used = 0
        import json as _json

        for h in reranked:
            if self.granularity == "session":
                s = sessions[h.session_id]
                msgs = P.strip_has_answer(
                    [{"role": m.role, "content": m.content} for m in s.messages]
                )
                if self.user_only_value:
                    msgs = [m for m in msgs if m.get("role") == "user"]
            else:
                msgs = doc_by_id[h.corpus_id].messages
            cost = len(_json.dumps(msgs, ensure_ascii=False)) + 64
            if budget is not None:
                if chunks and used + cost > budget:
                    continue
                used += cost
            chunks.append((h.date, msgs))
            if budget is None and len(chunks) >= self.top_k_final:
                break
        chunks = sort_chunks_by_date(chunks)

        if self.con:
            con_p = con_c = 0
            with timer() as t_con:
                noted = []
                for chunk_date, chunk_entry in chunks:
                    con_prompt = P.CON_PROMPT.format(
                        chunk_date,
                        P._json_dumps(chunk_entry),
                        question.question_date,
                        question.question,
                    )
                    note, _, tok = reader_generate(
                        reader, con_prompt, cache=cache, max_tokens=500, stage="con"
                    )
                    con_p += tok.get("con_prompt", 0)
                    con_c += tok.get("con_completion", 0)
                    noted.append((chunk_date, {"session_summary": note}))
            timings["con_s"] = t_con["s"]
            tokens["con_prompt"] = con_p
            tokens["con_completion"] = con_c
            chunks = noted

        with timer() as t_render:
            history = P.render_history_json(chunks)
        timings["render_s"] = t_render["s"]

        return self._final_answer(
            question,
            history,
            reader,
            cache,
            timings=timings,
            tokens=tokens,
            retrieval_trace=trace,
            extras={
                "n_candidates": len(reranked),
                "n_kept": len(chunks),
                "token_budget_chars": budget,
            },
        )
