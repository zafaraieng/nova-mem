#!/usr/bin/env python3
"""Merge chunked retrieval runs into the canonical run artifacts + one ledger record.

Chunked execution (``eval_retrieval.py --offset/--limit/--chunk-tag``) exists
because long-running background processes are not viable in some execution
environments. Per-question metrics are unaffected: every question is measured
independently inside its chunk; chunking only interleaves process boundaries
(the per-chunk embedder/reranker model loads happen outside the measured
per-question windows).

Usage:
    python scripts/merge_retrieval_chunks.py --run-id retrieval_dense_dev
    python scripts/merge_retrieval_chunks.py --run-id retrieval_dense_rerank_dev
"""

from __future__ import annotations

import argparse
import json
import sys
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
from nova_mem.utils import sha256_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", required=True, help="e.g. retrieval_dense_dev")
    ap.add_argument("--config", default="local.yaml")
    args = ap.parse_args()

    run_dir = REPO_ROOT / "results" / "runs" / args.run_id
    # chunk artifacts are traces.<tag>.jsonl for any --chunk-tag; exclude the
    # canonical merged output traces.jsonl itself
    chunk_traces = sorted(p for p in run_dir.glob("traces.*.jsonl") if p.name != "traces.jsonl")
    if not chunk_traces:
        sys.exit(f"No chunk files (traces.chunk_*.jsonl) in {run_dir}")

    summaries = []
    for cf in chunk_traces:
        tag = cf.stem.split(".", 1)[1]  # traces.<tag>.jsonl -> <tag>
        sf = run_dir / f"retrieval_summary.{tag}.json"
        if not sf.exists():
            sys.exit(f"Missing chunk summary {sf}")
        summaries.append(json.load(open(sf, encoding="utf-8")))

    base = summaries[0]
    for s in summaries[1:]:
        for key in ("retriever", "reranker", "split", "top_k", "top_n"):
            if s.get(key) != base.get(key):
                sys.exit(f"Chunk config mismatch on {key}: {s.get(key)!r} vs {base.get(key)!r}")
    ks = base["top_k"]

    # evidence ids per question (official convention; abstention -> [])
    cfg = load_yaml(REPO_ROOT / "configs" / args.config)
    benchmark = load_benchmark(REPO_ROOT / "data" / cfg["data"]["benchmark_file"])
    split = load_split(REPO_ROOT / "data" / cfg["data"]["split_file"])
    split_ids = set(split[base["split"]])
    ev = {q.question_id: q.retrieval_evidence_ids for q in benchmark if q.question_id in split_ids}

    per_q, traces, seen = [], {}, set()
    t_index = t_retrieve = t_rerank = 0.0
    n_questions = 0
    for s in summaries:
        t_index += s["latency_s"]["index_total"]
        t_retrieve += s["latency_s"]["retrieve_total"]
        t_rerank += s["latency_s"]["rerank_total"]
        n_questions += s["n_questions"]
    for cf in chunk_traces:
        with open(cf, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                qid = rec["question_id"]
                if qid in seen:
                    sys.exit(f"Duplicate question {qid} across chunks")
                seen.add(qid)
                traces[qid] = rec["trace"]
                per_q.append(retrieval_metrics_for_question(rec["trace"], ev.get(qid, []), ks=ks))

    missing = split_ids - seen
    if missing:
        sys.exit(f"{len(missing)} split questions missing from chunks, e.g. {sorted(missing)[:3]}")

    n_answerable = sum(1 for qid in split_ids if ev.get(qid))
    summary = aggregate_retrieval_metrics(per_q)
    result = {
        "retriever": base["retriever"],
        "reranker": base["reranker"],
        "split": base["split"],
        "n_questions": n_questions,
        "n_answerable": n_answerable,
        "top_k": ks,
        "top_n": base["top_n"],
        "metrics": summary,
        "latency_s": {
            "index_total": round(t_index, 3),
            "retrieve_total": round(t_retrieve, 3),
            "rerank_total": round(t_rerank, 3),
            "retrieve_per_question_ms": round(1000 * t_retrieve / max(1, n_questions), 1),
            "rerank_per_question_ms": round(1000 * t_rerank / max(1, n_questions), 1),
        },
        "merged_from": [cf.name for cf in chunk_traces],
    }
    (run_dir / "retrieval_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with open(run_dir / "traces.jsonl", "w", encoding="utf-8") as f:
        for qid, trace in traces.items():
            f.write(json.dumps({"question_id": qid, "trace": trace}, ensure_ascii=False) + "\n")

    ledger = RunLedger(REPO_ROOT / "results" / "ledger.jsonl")
    ledger.append(
        {
            "experiment_id": args.run_id,
            "dataset_revision": "longmemeval-cleaned",
            "dataset_sha256": sha256_file(REPO_ROOT / "data" / cfg["data"]["benchmark_file"]),
            "model_id": base["retriever"],
            "quantization": "fp32",
            "prompt_version": "v1",
            "memory_arch": f"retrieval_only{'+rerank' if base['reranker'] else ''}",
            "retrieval_config": {
                "retriever": base["retriever"],
                "reranker": base["reranker"],
                "top_k": ks,
                "top_n": base["top_n"],
                "granularity": "session",
                "user_only_index": True,
            },
            "eval_config": {"split": base["split"], "n_questions": n_questions, "judge": "none"},
            "hardware": hardware_info(),
            "runtime_s": round(t_index + t_retrieve + t_rerank, 3),
            "tokens_in": 0,
            "tokens_out": 0,
            "cost_usd": None,
            "seed": 0,
            "system_key": args.run_id.rsplit("_", 1)[0],
            "retrieval_metrics": summary,
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "deviations": [
                f"executed in {len(chunk_traces)} chunks; per-question metrics unaffected "
                "(chunking only interleaves process boundaries; per-chunk model loads are "
                "outside the measured per-question windows)"
            ],
        }
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
