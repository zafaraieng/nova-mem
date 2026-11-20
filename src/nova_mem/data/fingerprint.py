"""Fingerprint the shared question/evidence metadata used for paired runs."""

from __future__ import annotations

import hashlib
import json

from .loader import Question


def dataset_comparison_fingerprint(questions: list[Question]) -> str:
    """Hash shared question fields and evidence sessions, excluding filler.

    Oracle and cleaned-S intentionally have different haystacks. This
    fingerprint permits pairing only when question metadata and every
    evidence-session date/content match.
    """
    rows = []
    seen_question_ids: set[str] = set()
    for question in sorted(questions, key=lambda item: item.question_id):
        if question.question_id in seen_question_ids:
            raise ValueError(f"Duplicate question id: {question.question_id}")
        seen_question_ids.add(question.question_id)
        evidence_id_list = question.evidence_session_ids
        if len(evidence_id_list) != len(set(evidence_id_list)):
            raise ValueError(f"Duplicate evidence label for {question.question_id}")
        evidence_ids = set(evidence_id_list)
        evidence_sessions = {}
        for session in question.sessions:
            if session.session_id not in evidence_ids:
                continue
            if session.session_id in evidence_sessions:
                raise ValueError(
                    f"Duplicate evidence session id {session.session_id!r} "
                    f"for {question.question_id}"
                )
            evidence_sessions[session.session_id] = {
                "date": session.date,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in session.messages
                ],
            }
        if set(evidence_sessions) != evidence_ids:
            raise ValueError(f"Missing evidence session for {question.question_id}")
        rows.append(
            {
                "question_id": question.question_id,
                "question": question.question,
                "answer": question.answer,
                "question_type": question.question_type,
                "question_date": question.question_date,
                "evidence_sessions": [
                    {"session_id": session_id, **evidence_sessions[session_id]}
                    for session_id in sorted(evidence_sessions)
                ],
            }
        )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
