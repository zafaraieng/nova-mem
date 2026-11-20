#!/usr/bin/env python3
"""Run experiments (systems A/B/oracle/C/D/E) on a split with a configured backend.

Every run appends one record to results/ledger.jsonl and writes per-question
artifacts under results/runs/<experiment_id>/:
    hypotheses.jsonl   official-format {question_id, hypothesis}
    answers.jsonl      full AnswerResult instrumentation (prompt, trace, timings, tokens)
    eval-results-<judge>.jsonl   official-format judged records (if judging)

Examples:
    # smoke: 3 dev questions, mock backend, mock judge
    python scripts/run_experiment.py --system A --split dev --limit 3 --backend mock

    # real: system C with BM25 on dev-100 via Ollama
    python scripts/run_experiment.py --system C --split dev --reader primary
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
from nova_mem.config import (  # noqa: E402
    REPO_ROOT,
    apply_overrides,
    config_hash,
    load_config,
    load_yaml,
)
from nova_mem.data import dataset_comparison_fingerprint, load_benchmark, load_split  # noqa: E402
from nova_mem.evaluation import judge_hypotheses, official_qa_metrics  # noqa: E402
from nova_mem.memory import (  # noqa: E402
    FullContextSystem,
    NoMemorySystem,
    OracleContextSystem,
    RAGRerankSystem,
    RAGSystem,
    REGISTRY_VERSION,
    StructuredMemorySystem,
)
from nova_mem.metrics import (
    aggregate_retrieval_metrics,
    retrieval_metrics_for_question,
)  # noqa: E402
from nova_mem.models import GenerationCache, client_from_config  # noqa: E402
from nova_mem.retrieval import _HashEmbedder  # noqa: E402
from nova_mem.utils import get_logger, sha256_file  # noqa: E402

logger = get_logger("run_experiment")

CLEANED_S_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
ALIGNED_ORACLE_SHA256 = "a7e0011505b11219a3401284d3a86dd3238218e4a78a946ef86f9827a1979c7a"
SPLIT_FILE_SHA256 = "e1e0081783cf908901de0db83894ff44b343ebd1efdf646b6bb480b1317b34cb"


def build_systems(args, cfg, embedder):
    systems = {}
    local = cfg if "retrieval" in cfg else load_config("local")
    rag_c = local["retrieval"]["system_C"]
    rag_d = local["retrieval"]["system_D"]
    systems["A"] = lambda: NoMemorySystem(cot=False)
    systems["B"] = lambda: FullContextSystem()
    systems["oracle"] = lambda: OracleContextSystem()
    systems["C"] = lambda: RAGSystem(
        retriever_kind=rag_c["retriever"],
        embedder=embedder,
        top_k=rag_c["top_k"],
        granularity=rag_c["granularity"],
        user_only_index=rag_c["user_only_index"],
        con=rag_c["con"],
    )
    for system_key, config_key in (
        ("C_dense", "system_C_dense"),
        ("C_hybrid", "system_C_hybrid"),
        ("C_rrf", "system_C_rrf"),
    ):
        fusion_cfg = local["retrieval"][config_key]
        systems[system_key] = lambda fusion_cfg=fusion_cfg: RAGSystem(
            retriever_kind=fusion_cfg["retriever"],
            embedder=embedder,
            top_k=fusion_cfg["top_k"],
            granularity=fusion_cfg["granularity"],
            user_only_index=fusion_cfg["user_only_index"],
            user_only_value=fusion_cfg.get("user_only_value", False),
            con=fusion_cfg["con"],
            candidate_pool_size=fusion_cfg.get("candidate_pool_size", 25),
            rrf_k=fusion_cfg.get("rrf_k", 60),
        )
    systems["D"] = lambda: _build_d(rag_d, embedder)
    import os

    skip_filtered = os.environ.get("NOVA_E_SKIP_FILTERED") == "1"
    # v2 extraction budget (diagnosis-driven, deviation-tagged): default keeps the
    # frozen 800; NOVA_E_EXTRACT_MAX_TOKENS raises it for continuation runs.
    extract_max = int(os.environ.get("NOVA_E_EXTRACT_MAX_TOKENS", "800"))
    systems["E"] = lambda: StructuredMemorySystem(
        embedder=embedder,
        top_k=10,
        skip_filtered_sessions=skip_filtered,
        extract_max_tokens=extract_max,
    )
    return systems


def _build_d(rag_d, embedder):
    from nova_mem.reranking import CrossEncoderReranker, MockReranker

    reranker = (
        MockReranker() if embedder.__class__.__name__ == "_HashEmbedder" else CrossEncoderReranker()
    )
    return RAGRerankSystem(
        reranker=reranker,
        retriever_kind=rag_d["retriever"],
        embedder=embedder,
        top_n=rag_d["top_n"],
        top_k_final=rag_d["top_k_final"],
        granularity=rag_d["granularity"],
        con=rag_d["con"],
    )


def _parse_override_pairs(overrides: list[str]) -> list[tuple[str, object]]:
    """[(dotted.key, typed value), ...] preserving CLI order (dict() dedupes last-wins)."""
    from nova_mem.config import parse_override

    return [parse_override(s) for s in overrides]


def _validate_test_run(args, cfg) -> None:
    if args.split == "all":
        raise ValueError(
            "The all split includes held-out test data; run dev or the frozen test split only"
        )
    if args.split != "test":
        return
    if args.config != "frozen_v1.yaml" or cfg.get("freeze") != "v1":
        raise ValueError("Test evaluation requires configs/frozen_v1.yaml with freeze: v1")
    if args.overrides:
        raise ValueError("Config overrides are not allowed for held-out test evaluation")
    if args.limit is not None:
        raise ValueError("Partial test runs are not allowed; evaluate the complete test-400 split")
    if args.no_ledger:
        raise ValueError("Held-out test evaluation must write its result to the experiment ledger")
    if args.judge == "none":
        raise ValueError("Held-out test evaluation requires a judge")
    if args.backend == "mock":
        raise ValueError("The mock backend cannot be used for held-out test evaluation")
    if not args.experiment_id:
        raise ValueError("Held-out test evaluation requires an explicit unique --experiment-id")

    ledger_path = REPO_ROOT / "results" / "ledger.jsonl"
    if ledger_path.is_file():
        with ledger_path.open("r", encoding="utf-8") as ledger:
            for line in ledger:
                if line.strip() and json.loads(line).get("experiment_id") == args.experiment_id:
                    raise ValueError(
                        f"Experiment id already exists in the ledger: {args.experiment_id}"
                    )
    run_dir = REPO_ROOT / "results" / "runs" / args.experiment_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")


def _validate_frozen_test_data(
    args, cfg, benchmark_name, dataset_sha, split_path, split, benchmark
) -> None:
    if args.split != "test":
        return
    expected_name = (
        "derived/longmemeval_oracle_s_aligned_v1.json"
        if args.system == "oracle"
        else "longmemeval_s_cleaned.json"
    )
    expected_hash = ALIGNED_ORACLE_SHA256 if args.system == "oracle" else CLEANED_S_SHA256
    dataset_cfg = cfg.get("dataset", {})
    frozen_files = dataset_cfg.get("files", {})
    if benchmark_name.replace("\\", "/") != expected_name:
        raise ValueError(f"Test evaluation requires the frozen dataset {expected_name}")
    if dataset_sha != expected_hash or frozen_files.get(expected_name) != expected_hash:
        raise ValueError(f"Test dataset checksum does not match the frozen {expected_name} hash")
    if (
        cfg.get("data", {}).get("split_file", "").replace("\\", "/") != "splits/dev_test_v0.json"
        or dataset_cfg.get("split") != "splits/dev_test_v0.json"
        or dataset_cfg.get("split_seed") != 42
        or dataset_cfg.get("dev_size") != 100
    ):
        raise ValueError("Test evaluation must use the committed seed-42 dev-100/test-400 split")
    if sha256_file(split_path) != SPLIT_FILE_SHA256:
        raise ValueError("Committed split file checksum differs from the audited split")
    dev_ids = split.get("dev", [])
    test_ids = split.get("test", [])
    question_ids = [question.question_id for question in benchmark]
    if (
        len(dev_ids) != 100
        or len(test_ids) != 400
        or len(set(dev_ids + test_ids)) != 500
        or set(dev_ids) & set(test_ids)
        or len(question_ids) != 500
        or set(question_ids) != set(dev_ids + test_ids)
    ):
        raise ValueError(
            "Frozen benchmark and split do not contain the exact committed 500 question IDs"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--system",
        required=True,
        choices=["A", "B", "oracle", "C", "C_dense", "C_hybrid", "C_rrf", "D", "E"],
    )
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--limit", type=int, default=None, help="first N questions of the split")
    ap.add_argument(
        "--reader",
        default="primary",
        help="models.yaml readers key (primary/secondary/full_context)",
    )
    ap.add_argument(
        "--backend",
        default=None,
        choices=[None, "mock"],
        help="force a backend (mock for smoke tests)",
    )
    ap.add_argument("--judge", default="local", choices=["local", "official", "none"])
    ap.add_argument("--config", default="local.yaml")
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override, e.g. --set retrieval.system_C.top_k=3 "
        "(dev-only overrides; recorded in the ledger and hashed)",
    )
    ap.add_argument(
        "--no-ledger",
        action="store_true",
        help="skip the ledger append (mechanism tests only; scientific runs must always write it)",
    )
    ap.add_argument("--no-judge", dest="judge", action="store_const", const="none")
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="record blocked/failed questions as generation errors and complete the run with "
        "the answered subset (deviation-tagged; the ledger stores the failed ids)",
    )
    args = ap.parse_args()

    if args.split == "test":
        frozen_v1 = REPO_ROOT / "configs" / "frozen_v1.yaml"
        if args.config != "frozen_v1.yaml" or not frozen_v1.is_file():
            raise ValueError(
                "Test evaluation is blocked until configs/frozen_v1.yaml exists and is selected"
            )
    cfg = load_yaml(REPO_ROOT / "configs" / args.config)
    _validate_test_run(args, cfg)
    if args.overrides:
        cfg = apply_overrides(cfg, args.overrides)
    import os

    models = load_config(os.environ.get("NOVA_MODELS", "models"))
    data_cfg = cfg["data"]

    # ---- data --------------------------------------------------------
    dataset_variant = "custom"
    benchmark_name = data_cfg["benchmark_file"]
    if args.system == "oracle":
        benchmark_name = data_cfg.get("oracle_aligned_file")
        if not benchmark_name:
            raise ValueError(
                "Oracle runs require data.oracle_aligned_file; raw Oracle dates are not aligned"
            )
        if benchmark_name.replace("\\", "/") != "derived/longmemeval_oracle_s_aligned_v1.json":
            raise ValueError("Oracle runs must use the audited oracle_s_aligned_v1 dataset variant")
        dataset_variant = "oracle_s_aligned_v1"
    elif benchmark_name.replace("\\", "/") == "longmemeval_s_cleaned.json":
        dataset_variant = "cleaned_s"
    benchmark_path = REPO_ROOT / "data" / benchmark_name
    dataset_sha = sha256_file(benchmark_path)
    if dataset_variant == "cleaned_s" and dataset_sha != CLEANED_S_SHA256:
        raise ValueError("Cleaned-S input checksum differs from the audited source")
    if dataset_variant == "oracle_s_aligned_v1":
        if dataset_sha != ALIGNED_ORACLE_SHA256:
            raise ValueError("Aligned Oracle input checksum differs from the audited variant")
        manifest_path = REPO_ROOT / "data" / "derived" / "oracle_s_aligned_v1_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        if (
            manifest.get("variant") != dataset_variant
            or manifest.get("derived_sha256") != dataset_sha
        ):
            raise ValueError("Aligned Oracle data does not match its recorded manifest checksum")
    benchmark = load_benchmark(benchmark_path)
    comparison_fingerprint = dataset_comparison_fingerprint(benchmark)
    split_path = REPO_ROOT / "data" / data_cfg["split_file"]
    split = load_split(split_path)
    _validate_frozen_test_data(args, cfg, benchmark_name, dataset_sha, split_path, split, benchmark)
    if args.split == "test" and len(split.get("test", [])) != 400:
        raise ValueError("The frozen test evaluation requires exactly 400 committed test IDs")
    if args.split == "all":
        questions = benchmark
    else:
        ids = set(split[args.split])
        questions = [q for q in benchmark if q.question_id in ids]
    if args.split == "test" and len(questions) != 400:
        raise ValueError("The benchmark does not contain all 400 committed test questions")
    if args.limit:
        questions = questions[: args.limit]
    qmap = {q.question_id: q for q in benchmark}

    # ---- clients ------------------------------------------------------
    reader_model_config = {}
    if args.backend == "mock":
        from nova_mem.models import MockLLMClient

        reader = MockLLMClient(model_id="mock-reader")
        extractor = MockLLMClient(model_id="mock-extractor", respond=lambda p: "[]")
        judge_client = MockLLMClient(
            model_id="mock-judge", respond=lambda p: "yes" if "unanswerable" in p else "yes"
        )
        embedder = _HashEmbedder()
    else:
        reader_model_config = models["readers"][args.reader]
        reader = client_from_config(reader_model_config)
        extractor = client_from_config(models["extractor"])
        embedder = None  # real embedder constructed lazily per system kind
        if args.system in ("C", "C_dense", "C_hybrid", "C_rrf", "D", "E"):
            from nova_mem.models import _lazy_embedder

            embedder = _lazy_embedder(models["embedder"]["name"])

    cache = GenerationCache(REPO_ROOT / cfg["cache"]["root"], enabled=cfg["cache"]["enabled"])

    # ---- system -------------------------------------------------------
    systems = build_systems(
        args, cfg if args.config != "models.yaml" else load_config("local"), embedder
    )
    system = systems[args.system]()

    experiment_id = args.experiment_id or (
        f"{args.system}_{args.split}_{reader.model_id}_{time.strftime('%Y%m%d-%H%M%S')}"
    ).replace("/", "_").replace(":", "-")
    run_dir = REPO_ROOT / "results" / "runs" / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    generation_errors_path = run_dir / "generation-errors.jsonl"
    generation_errors_path.write_text("", encoding="utf-8")

    # ---- run ----------------------------------------------------------

    logger.info(
        "running %s on %s (%d questions) with reader %s",
        system.name,
        args.split,
        len(questions),
        reader.model_id,
    )
    t0 = time.time()
    hyp_records, answer_records, generation_errors = [], [], []
    # v2 (2026-11-15, S-31): incremental flush — hypotheses and answers are
    # appended per question instead of written once at the end. A recycled
    # sandbox or killed process then leaves a recoverable partial run rather
    # than losing hours of extraction work (bit the E dev-10 pilot on S-30:
    # ~50 completed extractions vanished with zero persisted output).
    hyp_path = run_dir / "hypotheses.jsonl"
    ans_path = run_dir / "answers.jsonl"
    # Both files must exist even when nothing is answered, so that partial
    # runs are always discoverable on disk (expected by the failure tests).
    if not ans_path.is_file():
        ans_path.write_text("", encoding="utf-8")
    if not hyp_path.is_file():
        hyp_path.write_text("", encoding="utf-8")
    # v3 (2026-11-17, S-35.2): true resume — if answers.jsonl already holds
    # rows for this experiment id (restored partial run after a sandbox
    # recycle), keep them, preload the records, and skip those questions.
    # v2 truncated the files unconditionally, which erased a restored
    # 56-answer C-bm25 run and forced a full (cache-replayed) regeneration.
    answered: dict[str, dict] = {}
    if ans_path.is_file():
        for _line in ans_path.read_text(encoding="utf-8").splitlines():
            if _line.strip():
                _rec = json.loads(_line)
                answered[_rec["question_id"]] = _rec
    if answered:
        logger.info(
            "resuming: %d answered question(s) already present; skipping them",
            len(answered),
        )
    hyp_records = [
        {"question_id": qid, "hypothesis": rec["hypothesis"]}
        for qid, rec in answered.items()
        if rec.get("hypothesis") is not None
    ]
    answer_records = list(answered.values())
    generation_errors = []
    for index, q in enumerate(questions, 1):
        if q.question_id in answered:
            continue
        question_t0 = time.time()
        kwargs = {"extractor": extractor} if args.system == "E" else {}
        try:
            res = system.answer(q, reader, cache=cache, **kwargs)
        except Exception as e:  # noqa: BLE001 — retain successes, then fail incomplete runs
            logger.error("question %s failed: %r", q.question_id, e)
            generation_errors.append(
                {
                    "question_id": q.question_id,
                    "error_type": type(e).__name__,
                    "error": str(e),
                }
            )
            with open(generation_errors_path, "a", encoding="utf-8") as error_file:
                error_file.write(json.dumps(generation_errors[-1], ensure_ascii=False) + "\n")
                error_file.flush()
        else:
            rec = {"question_id": q.question_id, "hypothesis": res.hypothesis}
            with open(hyp_path, "a", encoding="utf-8") as hf:
                hf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                hf.flush()
            with open(ans_path, "a", encoding="utf-8") as af:
                af.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
                af.flush()
            hyp_records.append(rec)
            answer_records.append(res.to_dict())
        logger.info(
            "generation progress: %d/%d questions (%d failed; question %s took %.1fs)",
            index,
            len(questions),
            len(generation_errors),
            q.question_id,
            time.time() - question_t0,
        )
    runtime_s = time.time() - t0

    if generation_errors and not args.allow_partial:
        raise RuntimeError(
            f"generation incomplete: {len(generation_errors)} question(s) failed "
            f"({[error['question_id'] for error in generation_errors]}); "
            f"{len(answer_records)}/{len(questions)} answers saved under {run_dir}"
        )

    tokens_in = sum(
        r["tokens"].get("generate_prompt", 0)
        + r["tokens"].get("con_prompt", 0)
        + r["tokens"].get("extract_prompt", 0)
        for r in answer_records
    )
    tokens_out = sum(
        r["tokens"].get("generate_completion", 0)
        + r["tokens"].get("con_completion", 0)
        + r["tokens"].get("extract_completion", 0)
        for r in answer_records
    )

    # ---- retrieval metrics --------------------------------------------
    per_q_metrics = []
    for r in answer_records:
        q = qmap[r["question_id"]]
        per_q_metrics.append(
            retrieval_metrics_for_question(r["retrieval_trace"], q.retrieval_evidence_ids)
        )
    retrieval_summary = aggregate_retrieval_metrics(per_q_metrics)

    # ---- judge ----------------------------------------------------------
    eval_summary = None
    judged_records = []
    judge_runtime_s = 0.0
    judge_model_config = {}
    if args.judge != "none" and hyp_records:
        if args.backend == "mock":
            judge = judge_client
        elif args.judge == "official":
            judge_model_config = models["judge"]["official"]
            judge = client_from_config(judge_model_config)
        else:
            judge_model_config = models["judge"]["local"]
            judge = client_from_config(judge_model_config)
        eval_name = f"eval-results-{judge.model_id}".replace("/", "_").replace(":", "-")
        eval_path = run_dir / f"{eval_name}.jsonl"
        judge_t0 = time.time()
        judged_records = judge_hypotheses(benchmark, hyp_records, judge, eval_path)
        judge_runtime_s = time.time() - judge_t0
        eval_summary = official_qa_metrics(
            judged_records,
            [{"question_id": q.question_id, "question_type": q.question_type} for q in benchmark],
        )

    # ---- ledger ---------------------------------------------------------
    total_wall_s = time.time() - t0
    cache_statuses = [
        cached
        for answer in answer_records
        for timing, cached in answer["timings"].items()
        if timing.endswith("_cached")
    ]
    server_timing_sums = {}
    for name in ("prompt_eval", "generation_eval", "total"):
        durations = [
            answer["timings"][key]
            for answer in answer_records
            for key in answer["timings"]
            if key.endswith(f"_{name}_s") and "_cached_" not in key
        ]
        server_timing_sums[name] = round(sum(durations), 6) if durations else None
    cached_source_generation_durations = [
        answer["timings"][key]
        for answer in answer_records
        for key in answer["timings"]
        if key.endswith("_cached_source_s")
    ]
    cached_source_server_timing_sums = {}
    for name in ("prompt_eval", "generation_eval", "total"):
        durations = [
            answer["timings"][key]
            for answer in answer_records
            for key in answer["timings"]
            if key.endswith(f"_cached_{name}_s")
        ]
        cached_source_server_timing_sums[name] = round(sum(durations), 6) if durations else None
    ledger = RunLedger(REPO_ROOT / "results" / "ledger.jsonl")
    record = {
        "experiment_id": experiment_id,
        "dataset_revision": "longmemeval-cleaned",
        "dataset_sha256": dataset_sha,
        "dataset_variant": dataset_variant,
        "dataset_comparison_group": (
            "longmemeval_s_evidence_aligned_v1"
            if dataset_variant in {"cleaned_s", "oracle_s_aligned_v1"}
            else None
        ),
        "dataset_comparison_fingerprint": comparison_fingerprint,
        "model_id": reader.model_id,
        "model_digest": reader_model_config.get("digest"),
        "reader_context_length": getattr(reader, "num_ctx", None),
        "quantization": reader_model_config.get("quantization", "mock"),
        "extractor_model_id": (
            models["extractor"]["model_id"]
            if args.system == "E" and args.backend != "mock"
            else None
        ),
        "extractor_digest": (
            models["extractor"].get("digest")
            if args.system == "E" and args.backend != "mock"
            else None
        ),
        "prompt_version": REGISTRY_VERSION,
        "memory_arch": system.name,
        "retrieval_config": {
            k: v for k, v in cfg.get("retrieval", {}).get(f"system_{args.system}", {}).items()
        }
        or {"retriever": getattr(getattr(system, "retriever", None), "name", None)},
        "eval_config": {
            "judge": args.judge,
            "judge_model": (
                None
                if args.judge == "none"
                else (judged_records[0]["autoeval_label"]["model"] if judged_records else None)
            ),
            "judge_model_digest": judge_model_config.get("digest"),
            "split": args.split,
            "n_questions": len(questions),
            "n_failed_questions": len(generation_errors),
            "failed_question_ids": [error["question_id"] for error in generation_errors],
        },
        "hardware": hardware_info(),
        "runtime_s": round(runtime_s, 3),
        "generation_runtime_s": round(runtime_s, 3),
        "judge_runtime_s": round(judge_runtime_s, 3),
        "total_wall_s": round(total_wall_s, 3),
        "cache_enabled": cache.enabled,
        "cache_hits": cache.hits,
        "cache_misses": cache.misses,
        "generation_calls": len(cache_statuses),
        "live_generation_calls": sum(not cached for cached in cache_statuses),
        "ollama_server_timing_sums": server_timing_sums,
        "cached_source_generation_s": (
            round(sum(cached_source_generation_durations), 6)
            if cached_source_generation_durations
            else None
        ),
        "cached_source_server_timing_sums": cached_source_server_timing_sums,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "cost_usd": None,
        "seed": cfg["generation"]["seed"],
        "system_key": args.system,
        "backend": reader.backend,
        "qa_metrics": eval_summary,
        "retrieval_metrics": retrieval_summary,
        "run_dir": str(run_dir.relative_to(REPO_ROOT)),
        "config_hash": config_hash(cfg),
        "config_overrides": dict(_parse_override_pairs(args.overrides)),
        "deviations": (
            ["mock-backend: numbers are NOT scientific results"] if args.backend == "mock" else []
        )
        + (
            [
                "modified-config: zai-glm4-plus-backend (sandbox continuation; not comparable to frozen Ollama/Qwen runs)"
            ]
            if reader.backend == "zai"
            else []
        )
        + (
            [
                "allow-partial: %d of %d questions failed generation and are excluded"
                % (len(generation_errors), len(questions))
            ]
            if (generation_errors and args.allow_partial)
            else []
        )
        + (
            [
                "E-skip-filtered: cloud content-filter blocked extraction sessions recorded "
                "as zero-fact (sandbox deviation; count per question in answers.jsonl extras)"
            ]
            if (args.system == "E" and os.environ.get("NOVA_E_SKIP_FILTERED") == "1")
            else []
        )
        + (["config-overrides: non-frozen development run"] if args.overrides else []),
    }
    if args.no_ledger:
        logger.warning("--no-ledger: run NOT recorded in the ledger (mechanism test only)")
    else:
        ledger.append(record)
    logger.info(
        "done | generation=%.1fs judge=%.1fs total=%.1fs | qa=%s | retrieval=%s",
        runtime_s,
        judge_runtime_s,
        total_wall_s,
        eval_summary and eval_summary.get("overall_accuracy"),
        retrieval_summary,
    )
    print(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "runtime_s": round(runtime_s, 1),
                "generation_runtime_s": round(runtime_s, 1),
                "judge_runtime_s": round(judge_runtime_s, 1),
                "total_wall_s": round(total_wall_s, 1),
                "cache_hits": cache.hits,
                "cache_misses": cache.misses,
                "config_overrides": record["config_overrides"],
                "qa": eval_summary,
                "retrieval": retrieval_summary,
            },
            indent=2,
        )
    )


def _parse_override_pairs(overrides: list[str]) -> list[tuple[str, object]]:
    from nova_mem.config import parse_override

    return [parse_override(s) for s in overrides]


if __name__ == "__main__":
    main()
