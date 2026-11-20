"""LongMemEval data loading.

Schema (verified against ``longmemeval-cleaned`` downloaded 2026-11-07):

Each instance:
    question_id        str   (abstention questions carry the ``_abs`` suffix;
                             their question_type stays the BASE type — the
                             literal value 'abstention' never occurs)
    question_type      str   single-session-user | single-session-assistant |
                             single-session-preference | multi-session |
                             temporal-reasoning | knowledge-update
    question, answer   str
    question_date      str   source date string (YYYY/MM/DD (weekday) HH:MM)
    haystack_dates     list[str]             parallel to haystack_session_ids
    haystack_session_ids list[str]           e.g. sharegpt_xxx_0, <hex>_1, answer_<hex>
    haystack_sessions  list[list[message]]   message = {role, content[, has_answer]}
    answer_session_ids list[str]             evidence session ids (``answer_``
                                            prefix also for _abs questions in the
                                            cleaned release; ``noans_`` appears only
                                            in the official retrieval corpus ids)

The official pipeline (``run_generation.py``) zips dates/ids/sessions and, for
retrieval corpora, uses corpus ids ``<session_id>`` (sessions) and
``<session_id>_<turn#>`` (turns, 1-offset). We preserve both conventions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

QUESTION_TYPES = [
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
    "abstention",
]

ABSTENTION_MARKER = "_abs"


@dataclass
class Message:
    role: str
    content: str


@dataclass
class Session:
    session_id: str
    date: str
    messages: list[Message]

    def user_only(self) -> list[Message]:
        return [m for m in self.messages if m.role == "user"]

    def text(self, user_only: bool = False) -> str:
        msgs = self.user_only() if user_only else self.messages
        return "\n".join(f"{m.role}: {m.content}" for m in msgs)

    def turn_docs(self) -> list[tuple[str, Message]]:
        """(corpus_id, message) with official 1-offset turn ids."""
        return [(f"{self.session_id}_{i + 1}", m) for i, m in enumerate(self.messages)]


@dataclass
class Question:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: list[Session] = field(default_factory=list)
    answer_session_ids: list[str] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        return ABSTENTION_MARKER in self.question_id

    @property
    def evidence_session_ids(self) -> list[str]:
        """Evidence ids with the abstention pseudo-evidence prefix stripped."""
        return [s.replace("noans_", "answer_") for s in self.answer_session_ids]

    @property
    def retrieval_evidence_ids(self) -> list[str]:
        """Evidence ids for retrieval metrics — EMPTY for abstention questions
        (official convention: their evidence is pseudo-evidence and is excluded
        from retrieval evaluation)."""
        return [] if self.is_abstention else self.evidence_session_ids


def _parse_instance(entry: dict) -> Question:
    sessions = [
        Session(
            session_id=sid,
            date=sdate,
            # real files carry an extra 'has_answer' annotation key; the
            # official pipeline strips it before prompting, so we drop it here
            messages=[Message(role=m["role"], content=m["content"]) for m in smessages],
        )
        for sdate, sid, smessages in zip(
            entry["haystack_dates"], entry["haystack_session_ids"], entry["haystack_sessions"]
        )
    ]
    return Question(
        question_id=entry["question_id"],
        question_type=entry["question_type"],
        question=entry["question"],
        answer=entry["answer"],
        question_date=entry["question_date"],
        sessions=sessions,
        answer_session_ids=list(entry.get("answer_session_ids", [])),
    )


def load_benchmark(path: str | Path) -> list[Question]:
    """Load a LongMemEval JSON file (list of instances)."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    questions = [_parse_instance(e) for e in raw]
    # integrity check: types must be known
    for q in questions:
        if q.question_type not in QUESTION_TYPES:
            raise ValueError(f"Unknown question_type {q.question_type!r} on {q.question_id}")
    return questions


def load_split_ids(path: str | Path) -> dict[str, list[str]]:
    """Load committed split file {split_name: [question_id, ...]}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_questions(
    questions: list[Question], split_ids: dict[str, list[str]]
) -> dict[str, list[Question]]:
    """Partition questions by committed id lists (error on unknown/missing ids)."""
    by_id = {q.question_id: q for q in questions}
    out: dict[str, list[Question]] = {}
    seen: set[str] = set()
    for name, ids in split_ids.items():
        out[name] = []
        for qid in ids:
            if qid not in by_id:
                raise KeyError(f"Split references unknown question_id {qid}")
            if qid in seen:
                raise ValueError(f"question_id {qid} appears in multiple splits")
            seen.add(qid)
            out[name].append(by_id[qid])
    missing = set(by_id) - seen
    if missing:
        raise ValueError(
            f"{len(missing)} questions not covered by splits, e.g. {sorted(missing)[:3]}"
        )
    return out
