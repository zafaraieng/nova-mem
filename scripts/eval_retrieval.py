#!/usr/bin/env python3
"""Retrieval-only evaluation on a split — no LLM, no judge, pure measurement.

Computes official session-recall@k / NDCG@k plus NOVA-Mem's evidence-density@k
for one or more retrieval configurations, with retrieval latency. Abstention
questions are included in the trace but excluded from evidence metrics
(undefined evidence set) EXCEPT via the noans-pseudo-evidence variant.

This is the cheapest real experiment in the project (CPU seconds) and anchors
RQ3 (rerank cost/benefit) and E7 (sensitivity) before any reader is involved.

Examples:
    python scripts/eval_retrieval.py --retriever bm25 --top-k 5,10 --split dev
    python scripts/eval_retrieval.py --retriever dense --top-k 5,10 --split dev
    python scripts/eval_retrieval.py --retriever dense --rerank --top-k 5,10 --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nova_mem.analysis import RunLedger, hardware_info  # noqa: E402
from nova_mem.config import REPO_ROOT, load_config, load_yaml  # noqa: E402
from nova_mem.data import load_benchmark, load_split  # noqa: E402
from nova_mem.metrics import (
    aggregate_retrieval_metrics,
    retrieval_metrics_for_question,
)  # noqa: E402
from nova_mem.utils import get_logger, sha256_file  # noqa: E402

logger = get_logger("eval_retrieval")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "dense"])
    ap.add_argument("--rerank", action="store_true", help="add cross-encoder rerank (top-N -> k)")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--top-k", default="5,10", help="comma-separated k values")
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument(
        "--offset",
        type=int,
        default=0,
        help="skip the first N questions of the split (chunked execution)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--chunk-tag",
        default=None,
        help="write chunk-suffixed artifacts and skip the ledger; merge later with scripts/merge_retrieval_chunks.py",
    )
    ap.add_argument("--no-ledger", action="store_true")
    ap.add_argument("--config", default="local.yaml")
    args = ap.parse_args()

    ks = [int(x) for x in args.top_k.split(",")]
    cfg = load_yaml(REPO_ROOT / "configs" / args.config)
    models = load_config("models")

    benchmark = load_benchmark(REPO_ROOT / "data" / cfg["data"]["benchmark_file"])
    split = load_split(REPO_ROOT / "data" / cfg["data"]["split_file"])
    if args.split == "all":
        questions = benchmark
    else:
        ids = set(split[args.split])
        questions = [q for q in benchmark if q.question_id in ids]
    if args.offset:
        questions = questions[args.offset :]
    if args.limit:
        questions = questions[: args.limit]

    # ---- build retriever --------------------------------------------------
    if args.retriever == "bm25":
        from nova_mem.retrieval import BM25Retriever

        retriever = BM25Retriever()
        retriever_label = "bm25"
    else:
        from nova_mem.models import _lazy_embedder
        from nova_mem.retrieval import DenseRetriever

        embedder = _lazy_embedder(models["embedder"]["name"])
        retriever = DenseRetriever(embedder)
        retriever_label = f"dense:{models['embedder']['name']}"

    reranker = None
    reranker_label = None
    if args.rerank:
        from nova_mem.reranking import CrossEncoderReranker

        reranker = CrossEncoderReranker(models["reranker"]["primary"])
        reranker_label = models["reranker"]["primary"]

    # ---- run ---------------------------------------------------------------
    from nova_mem.retrieval import Document, build_session_corpus  # noqa: F401

    per_q = []
    traces = {}
    t_retrieve_total = 0.0
    t_rerank_total = 0.0
    t_index_total = 0.0
    for q in questions:
        docs = build_session_corpus(q, user_only=True)
        t0 = time.perf_counter()
        retriever.build(docs)
        t_index_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        hits = retriever.search(q.question, max(ks + [args.top_n] if reranker else ks))
        t_retrieve_total += time.perf_counter() - t0

        if reranker is not None:
            doc_by_id = {d.corpus_id: d for d in docs}
            cand = [doc_by_id[h.corpus_id] for h in hits if h.corpus_id in doc_by_id]
            hits, dt = reranker.rerank(q.question, cand, top_k=max(ks))
            t_rerank_total += dt

        trace = [
            {
                "rank": h.rank,
                "corpus_id": h.corpus_id,
                "session_id": h.session_id,
                "score": round(h.score, 6),
                "is_evidence": h.session_id in set(q.evidence_session_ids),
            }
            for h in hits
        ]
        traces[q.question_id] = trace
        per_q.append(retrieval_metrics_for_question(trace, q.retrieval_evidence_ids, ks=ks))

    summary = aggregate_retrieval_metrics(per_q)
    n_answerable = sum(1 for q in questions if not q.is_abstention)
    result = {
        "retriever": retriever_label,
        "reranker": reranker_label,
        "split": args.split,
        "offset": args.offset,
        "chunk_tag": args.chunk_tag,
        "n_questions": len(questions),
        "n_answerable": n_answerable,
        "top_k": ks,
        "top_n": args.top_n if reranker else None,
        "metrics": summary,
        "latency_s": {
            "index_total": round(t_index_total, 3),
            "retrieve_total": round(t_retrieve_total, 3),
            "rerank_total": round(t_rerank_total, 3),
            "retrieve_per_question_ms": round(1000 * t_retrieve_total / max(1, len(questions)), 1),
            "rerank_per_question_ms": round(1000 * t_rerank_total / max(1, len(questions)), 1),
        },
    }

    # ---- persist -----------------------------------------------------------
    run_id = f"retrieval_{args.retriever}" + ("_rerank" if args.rerank else "") + f"_{args.split}"
    out_dir = REPO_ROOT / "results" / "runs" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f".{args.chunk_tag}" if args.chunk_tag else ""
    (out_dir / f"retrieval_summary{suffix}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with open(out_dir / f"traces{suffix}.jsonl", "w", encoding="utf-8") as f:
        for qid, trace in traces.items():
            f.write(json.dumps({"question_id": qid, "trace": trace}, ensure_ascii=False) + "\n")

    if not args.chunk_tag and not args.no_ledger:
        ledger = RunLedger(REPO_ROOT / "results" / "ledger.jsonl")
        ledger.append(
            {
                "experiment_id": run_id,
                "dataset_revision": "longmemeval-cleaned",
                "dataset_sha256": sha256_file(REPO_ROOT / "data" / cfg["data"]["benchmark_file"]),
                "model_id": retriever_label,
                "quantization": "fp32",
                "prompt_version": "v1",
                "memory_arch": f"retrieval_only{'+rerank' if args.rerank else ''}",
                "retrieval_config": {
                    "retriever": retriever_label,
                    "reranker": reranker_label,
                    "top_k": ks,
                    "top_n": args.top_n if reranker else None,
                    "granularity": "session",
                    "user_only_index": True,
                },
                "eval_config": {
                    "split": args.split,
                    "n_questions": len(questions),
                    "judge": "none",
                },
                "hardware": hardware_info(),
                "runtime_s": round(t_index_total + t_retrieve_total + t_rerank_total, 3),
                "tokens_in": 0,
                "tokens_out": 0,
                "cost_usd": None,
                "seed": 0,
                "system_key": f"retrieval_{args.retriever}{'_rerank' if args.rerank else ''}",
                "retrieval_metrics": summary,
                "run_dir": str(out_dir.relative_to(REPO_ROOT)),
                "deviations": [],
            }
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
