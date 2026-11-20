"""Official-protocol judging.

Replicates ``src/evaluation/evaluate_qa.py`` from the official repository:
verbatim per-type judge prompts, temperature 0, ``max_tokens=10``, label =
``'yes' in response.lower()``. The judge backend is pluggable:

* ``OpenAICompatClient('gpt-4o-2024-08-06')`` — the official judge (API key
  via env var only; optional, never mandatory);
* any ``InferenceClient`` — the validated local judge (plan decision D6).

Output files use the official ``.eval-results-<model>`` naming and record
schema so the official ``print_qa_metrics.py`` can consume them unchanged.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..data.loader import Question
from ..models.client import InferenceClient
from ..memory.prompts import judge_prompt
from ..utils import get_logger

logger = get_logger(__name__)


def judge_label(
    question: Question, hypothesis: str, judge: InferenceClient
) -> tuple[bool, str, str]:
    """Returns (label, prompt, raw_response) using the official prompt."""
    prompt = judge_prompt(
        question.question_type,
        question.question,
        question.answer,
        hypothesis,
        abstention=question.is_abstention,
    )
    resp = judge.generate(prompt, max_tokens=10, temperature=0.0)
    label = "yes" in resp.text.lower()
    return label, prompt, resp.text


def judge_hypotheses(
    questions: list[Question],
    hypotheses: list[dict],
    judge: InferenceClient,
    out_path: str | Path,
    limit: int | None = None,
) -> list[dict]:
    """Judge ``[{question_id, hypothesis}, ...]`` records; write official-format output.

    Returns the same records with ``autoeval_label`` attached:
    ``{"model": judge.model_id, "label": bool}``.
    """
    qmap = {q.question_id: q for q in questions}
    hypothesis_map = {rec["question_id"]: rec for rec in hypotheses}
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    judged = []
    already_judged = set()
    n_skipped = 0
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid not in hypothesis_map:
                    raise ValueError(
                        f"{out_path}:{line_number}: unexpected judged question id {qid!r}"
                    )
                if qid in already_judged:
                    raise ValueError(
                        f"{out_path}:{line_number}: duplicate judged question id {qid!r}"
                    )
                if rec.get("hypothesis") != hypothesis_map[qid].get("hypothesis"):
                    raise ValueError(
                        f"{out_path}:{line_number}: hypothesis changed for question {qid!r}"
                    )
                if rec.get("autoeval_label", {}).get("model") != judge.model_id:
                    raise ValueError(
                        f"{out_path}:{line_number}: judge model changed for question {qid!r}"
                    )
                already_judged.add(qid)
                judged.append(rec)

    with open(out_path, "a", encoding="utf-8") as f:
        for rec in hypotheses:
            qid = rec.get("question_id")
            if qid not in qmap:
                n_skipped += 1
                logger.warning("skipping %s: not in reference data", qid)
                continue
            if qid in already_judged:
                continue
            label, _, _ = judge_label(qmap[qid], rec["hypothesis"], judge)
            rec = dict(rec)
            rec["autoeval_label"] = {"model": judge.model_id, "label": label}
            judged.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            already_judged.add(qid)
            if len(judged) % 10 == 0 or len(judged) == len(hypotheses):
                logger.info("judge progress: %d/%d questions", len(judged), len(hypotheses))
            if limit is not None and len(judged) >= limit:
                break
    if n_skipped:
        logger.warning("%d hypotheses skipped (unknown question ids)", n_skipped)
    return judged


def judge_agreement(records_a: list[dict], records_b: list[dict]) -> dict:
    """Agreement + Cohen's kappa between two judges on the same questions."""
    a = {r["question_id"]: r["autoeval_label"]["label"] for r in records_a}
    b = {r["question_id"]: r["autoeval_label"]["label"] for r in records_b}
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("No common question ids between the two judgment sets")
    n = len(common)
    agree = sum(a[q] == b[q] for q in common)
    pa = agree / n
    p_yes_a = sum(a[q] for q in common) / n
    p_yes_b = sum(b[q] for q in common) / n
    pe = p_yes_a * p_yes_b + (1 - p_yes_a) * (1 - p_yes_b)
    kappa = (pa - pe) / (1 - pe) if pe < 1.0 else 1.0
    return {
        "n": n,
        "agreement": round(pa, 4),
        "cohens_kappa": round(kappa, 4),
        "disagreements": [q for q in common if a[q] != b[q]],
    }
