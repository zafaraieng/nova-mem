"""Dataset statistics for data/README.md and Phase-2 reporting."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass


@dataclass
class DatasetStats:
    n_questions: int
    by_type: dict
    n_abstention: int
    sessions_per_question: dict  # min/mean/max
    messages_per_session: dict
    evidence_sessions_per_question: dict
    chars_per_question_history: dict  # proxy pre-tokenization
    unique_session_ids: int
    total_session_refs: int
    session_reuse_ratio: float


def _mstats(values: list[int]) -> dict:
    return {
        "min": int(min(values)),
        "mean": round(sum(values) / len(values), 2),
        "max": int(max(values)),
    }


def compute_stats(questions) -> DatasetStats:
    from .loader import Question  # avoid circular import at module import time

    assert all(isinstance(q, Question) for q in questions)
    by_type = Counter(q.question_type for q in questions)
    n_sessions = [len(q.sessions) for q in questions]
    n_msgs = [len(s.messages) for q in questions for s in q.sessions]
    n_ev = [len(q.answer_session_ids) for q in questions if not q.is_abstention]
    n_chars = [sum(len(m.content) for s in q.sessions for m in s.messages) for q in questions]
    all_ids = [s.session_id for q in questions for s in q.sessions]
    total_refs = len(all_ids)
    unique = len(set(all_ids))
    return DatasetStats(
        n_questions=len(questions),
        by_type=dict(sorted(by_type.items())),
        # Official convention: abstention is encoded by the ``_abs`` question-id
        # suffix, NEVER as a literal question_type value (which stays the base
        # type). Regression: counting by_type.get("abstention") always yields 0.
        n_abstention=sum(1 for q in questions if q.is_abstention),
        sessions_per_question=_mstats(n_sessions),
        messages_per_session=_mstats(n_msgs),
        evidence_sessions_per_question=_mstats(n_ev) if n_ev else {"min": 0, "mean": 0, "max": 0},
        chars_per_question_history=_mstats(n_chars),
        unique_session_ids=unique,
        total_session_refs=total_refs,
        session_reuse_ratio=round(total_refs / unique, 3) if unique else 0.0,
    )


def stats_markdown(stats: DatasetStats) -> str:
    lines = [
        "| Statistic | Value |",
        "|---|---|",
        f"| Questions | {stats.n_questions} |",
        f"| Abstention questions (`_abs` id suffix; base type kept) | {stats.n_abstention} |",
        f"| Sessions per question (min/mean/max) | {stats.sessions_per_question['min']} / {stats.sessions_per_question['mean']} / {stats.sessions_per_question['max']} |",
        f"| Messages per session (min/mean/max) | {stats.messages_per_session['min']} / {stats.messages_per_session['mean']} / {stats.messages_per_session['max']} |",
        f"| Evidence sessions per answerable question | {stats.evidence_sessions_per_question['min']} / {stats.evidence_sessions_per_question['mean']} / {stats.evidence_sessions_per_question['max']} |",
        f"| History chars per question (min/mean/max) | {stats.chars_per_question_history['min']} / {stats.chars_per_question_history['mean']} / {stats.chars_per_question_history['max']} |",
        f"| Unique session ids across dataset | {stats.unique_session_ids} |",
        f"| Total session references | {stats.total_session_refs} |",
        f"| Session reuse ratio (refs / unique) | {stats.session_reuse_ratio} |",
        "",
        "| Question type | Count |",
        "|---|---|",
    ]
    for t, c in stats.by_type.items():
        lines.append(f"| {t} | {c} |")
    return "\n".join(lines)


def stats_dict(stats: DatasetStats) -> dict:
    return asdict(stats)
