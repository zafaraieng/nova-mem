#!/usr/bin/env python3
"""Pilot runs (Phase 5): dev-100 for a system grid; E7 sensitivity sweeps.

Thin orchestration over run_experiment.py; every underlying run still lands in
the ledger with its own experiment_id prefixed 'pilot_'.

E7 sensitivity (plan §4.5, dev-only): one-factor-at-a-time (OFAT) around the
base config by default — vary k, vary retriever, vary granularity, plus one
retriever×granularity interaction check — or the full cross-product with
``--full-grid``. Sweep arms pass ``--set`` overrides (recorded per-run in the
ledger and reflected in the config hash); the arm identical to the base config
is skipped (the main pilot run covers it).

Examples:
    python scripts/run_pilot.py --systems A oracle C D          # main pilot grid
    python scripts/run_pilot.py --sensitivity --dry-run         # show sweep plan
    python scripts/run_pilot.py --sensitivity                   # OFAT sweeps (5 arms)
    python scripts/run_pilot.py --sensitivity --full-grid       # k×retriever×granularity
    python scripts/run_pilot.py --sensitivity --sweep-systems C D --sweep-k 3 10
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from nova_mem.config import load_yaml  # noqa: E402


def build_sweep_commands(
    base_cfg: dict,
    reader: str,
    judge: str,
    config: str = "local.yaml",
    split: str = "dev",
    ks: tuple[int, ...] = (3, 5, 10),
    retrievers: tuple[str, ...] = ("bm25", "dense"),
    granularities: tuple[str, ...] = ("session", "turn"),
    systems: tuple[str, ...] = ("C",),
    full_grid: bool = False,
) -> list[tuple[str, list[str]]]:
    """E7 sweep arms as (experiment_id, argv) pairs; skips the base-config arm.

    OFAT design (default): for each factor, every value except the base value,
    all other factors at base; plus retriever=dense × granularity=turn as one
    interaction check. ``--full-grid``: complete k × retriever × granularity
    cross-product minus the base arm.
    """
    py = sys.executable
    base_c = base_cfg["retrieval"]["system_C"]
    base_d = base_cfg["retrieval"]["system_D"]
    base_k = base_c["top_k"]
    base_ret = base_c["retriever"]
    base_gran = base_c["granularity"]

    def arm(system: str, params: dict[str, object], tag: str) -> tuple[str, list[str]]:
        section = f"system_{system}"
        argv = [
            py,
            "scripts/run_experiment.py",
            "--system",
            system,
            "--split",
            split,
            "--reader",
            reader,
            "--config",
            config,
            "--judge",
            judge,
            "--experiment-id",
            f"pilot_sens_{system}_{tag}",
        ]
        for key, value in params.items():
            argv += ["--set", f"retrieval.{section}.{key}={value}"]
        return f"pilot_sens_{system}_{tag}", argv

    arms: list[tuple[str, list[str]]] = []
    if full_grid:
        for system in systems:
            k_key = "top_k" if system == "C" else "top_k_final"
            for ret in retrievers:
                for k in ks:
                    for gran in granularities:
                        if system == "C" and (ret, k, gran) == (base_ret, base_k, base_gran):
                            continue  # base arm covered by the main pilot run
                        params = {"retriever": ret, k_key: k, "granularity": gran}
                        arms.append(arm(system, params, f"{ret}_k{k}_{gran}"))
        return arms

    # OFAT: one factor off-base at a time (system C sweep space)
    if "C" in systems:
        for k in ks:
            if k != base_k:
                arms.append(
                    arm(
                        "C",
                        {"retriever": base_ret, "top_k": k, "granularity": base_gran},
                        f"{base_ret}_k{k}_{base_gran}",
                    )
                )
        for ret in retrievers:
            if ret != base_ret:
                arms.append(
                    arm(
                        "C",
                        {"retriever": ret, "top_k": base_k, "granularity": base_gran},
                        f"{ret}_k{base_k}_{base_gran}",
                    )
                )
        for gran in granularities:
            if gran != base_gran:
                arms.append(
                    arm(
                        "C",
                        {"retriever": base_ret, "top_k": base_k, "granularity": gran},
                        f"{base_ret}_k{base_k}_{gran}",
                    )
                )
        # one interaction check: non-base retriever × non-base granularity
        other_ret = next((r for r in retrievers if r != base_ret), None)
        other_gran = next((g for g in granularities if g != base_gran), None)
        if other_ret and other_gran:
            arms.append(
                arm(
                    "C",
                    {"retriever": other_ret, "top_k": base_k, "granularity": other_gran},
                    f"{other_ret}_k{base_k}_{other_gran}",
                )
            )
    # system D: k analog only (top_k_final at base retriever/granularity)
    if "D" in systems:
        base_dk = base_d["top_k_final"]
        for k in ks:
            if k != base_dk:
                arms.append(
                    arm(
                        "D",
                        {
                            "retriever": base_d["retriever"],
                            "top_k_final": k,
                            "granularity": base_d["granularity"],
                        },
                        f"{base_d['retriever']}_k{k}_{base_d['granularity']}",
                    )
                )
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--systems",
        nargs="+",
        default=["A", "oracle", "C", "D"],
        choices=["A", "B", "oracle", "C", "D", "E"],
    )
    ap.add_argument("--reader", default="primary")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--config", default="local.yaml")
    ap.add_argument("--judge", default="local")
    ap.add_argument(
        "--sensitivity",
        action="store_true",
        help="E7 sweep on dev only (k, retriever, granularity)",
    )
    ap.add_argument(
        "--sweep-systems",
        nargs="+",
        default=["C"],
        choices=["C", "D"],
        help="systems included in the sweep (D sweeps top_k_final)",
    )
    ap.add_argument("--sweep-k", nargs="+", type=int, default=[3, 5, 10])
    ap.add_argument(
        "--sweep-retriever", nargs="+", default=["bm25", "dense"], choices=["bm25", "dense"]
    )
    ap.add_argument(
        "--sweep-granularity", nargs="+", default=["session", "turn"], choices=["session", "turn"]
    )
    ap.add_argument(
        "--full-grid",
        action="store_true",
        help="full k×retriever×granularity cross-product instead of OFAT",
    )
    ap.add_argument("--dry-run", action="store_true", help="print planned commands, run nothing")
    args = ap.parse_args()

    py = sys.executable
    commands: list[tuple[str, list[str]]] = []
    for system in args.systems:
        commands.append(
            (
                f"pilot_{system}_{args.split}_{args.reader}",
                [
                    py,
                    "scripts/run_experiment.py",
                    "--system",
                    system,
                    "--split",
                    args.split,
                    "--reader",
                    args.reader,
                    "--config",
                    args.config,
                    "--judge",
                    args.judge,
                    "--experiment-id",
                    f"pilot_{system}_{args.split}_{args.reader}",
                ],
            )
        )

    if args.sensitivity:
        base_cfg = load_yaml(REPO / "configs" / args.config)
        commands += build_sweep_commands(
            base_cfg,
            reader=args.reader,
            judge=args.judge,
            config=args.config,
            split=args.split,
            ks=tuple(args.sweep_k),
            retrievers=tuple(args.sweep_retriever),
            granularities=tuple(args.sweep_granularity),
            systems=tuple(args.sweep_systems),
            full_grid=args.full_grid,
        )

    for experiment_id, argv in commands:
        print(f"[pilot] {experiment_id}\n  {' '.join(argv)}")
        if not args.dry_run:
            subprocess.run(argv, cwd=REPO, check=False)
    n = len(commands)
    suffix = " (dry run — nothing executed)" if args.dry_run else ""
    print(f"[done] pilot orchestration complete: {n} run(s){suffix} — inspect results/ledger.jsonl")


if __name__ == "__main__":
    main()
