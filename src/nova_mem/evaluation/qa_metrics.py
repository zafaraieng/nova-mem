"""Answer metrics: official aggregation + clearly-labeled non-official extras.

Official part replicates ``print_qa_metrics.py``: per-type accuracy over the
six answerable types, task-averaged accuracy, overall accuracy, abstention
accuracy. Non-official additions (exact match, token F1) are reported in
appendices only, never mixed with official numbers.
"""

from __future__ import annotations

import re
from collections import defaultdict

from ..data.loader import QUESTION_TYPES

OFFICIAL_TYPES = [t for t in QUESTION_TYPES if t != "abstention"]


def official_qa_metrics(records: list[dict], questions: list[dict]) -> dict:
    """records: [{question_id, autoeval_label:{label}}]; questions: reference
    entries with question_type (dicts or Question objects).

    Official semantics (print_qa_metrics.py): ``_abs`` questions are counted
    under their BASE question_type for per-type / task-averaged / overall
    accuracy, and additionally in the separate abstention accuracy.
    """
    qtype = {q["question_id"]: q["question_type"] for q in questions}
    by_type: dict[str, list[int]] = defaultdict(list)
    abstention: list[int] = []
    for r in records:
        qid = r["question_id"]
        label = int(bool(r["autoeval_label"]["label"]))
        by_type[qtype[qid]].append(label)
        if "_abs" in qid:
            abstention.append(label)
    all_labels = [x for v in by_type.values() for x in v]
    out = {
        "n_judged": len(records),
        "overall_accuracy": round(sum(all_labels) / len(all_labels), 4) if all_labels else None,
        "per_type": {
            t: {"accuracy": round(sum(v) / len(v), 4), "n": len(v)}
            for t, v in sorted(by_type.items())
        },
        "abstention": (
            {"accuracy": round(sum(abstention) / len(abstention), 4), "n": len(abstention)}
            if abstention
            else None
        ),
    }
    type_accs = [sum(v) / len(v) for v in by_type.values() if v]
    out["task_averaged_accuracy"] = round(sum(type_accs) / len(type_accs), 4) if type_accs else None
    return out


# ---------------------------------------------------------------------------
# Non-official robustness metrics (appendix use only)
# ---------------------------------------------------------------------------

_ANS_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> list[str]:
    return _ANS_RE.findall(text.lower())


def exact_match(prediction: str, answer: str) -> int:
    return int(" ".join(_normalize(prediction)) == " ".join(_normalize(answer)))


def token_f1(prediction: str, answer: str) -> float:
    """SQuAD-style token F1 (multiset overlap via Counter)."""
    from collections import Counter

    pred_toks = _normalize(prediction)
    ans_toks = _normalize(answer)
    if not pred_toks or not ans_toks:
        return 1.0 if not pred_toks and not ans_toks else 0.0
    common = Counter(pred_toks) & Counter(ans_toks)
    n_common = sum(common.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(pred_toks)
    recall = n_common / len(ans_toks)
    return 2 * precision * recall / (precision + recall)


def nonofficial_metrics(records: list[dict], questions: list[dict]) -> dict:
    qmap = {q["question_id"]: q for q in questions}
    em, f1s = [], []
    for r in records:
        q = qmap[r["question_id"]]
        if "_abs" in r["question_id"]:
            continue
        em.append(exact_match(r["hypothesis"], q["answer"]))
        f1s.append(token_f1(r["hypothesis"], q["answer"]))
    return {
        "note": "NON-OFFICIAL metrics (EM / token-F1 after normalization); appendix use only",
        "n": len(em),
        "exact_match": round(sum(em) / len(em), 4) if em else None,
        "token_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
    }
