"""Append-only run ledger.

Every experiment appends one JSON record per run. Records are never edited or
deleted; aggregation reads the ledger. Required schema (reproduction_plan.md
§4.6):

    experiment_id, date_utc, dataset_revision, dataset_sha256, model_id,
    quantization, prompt_version, memory_arch, retrieval_config, eval_config,
    hardware, runtime_s, tokens_in, tokens_out, cost_usd|null, seed
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

REQUIRED_KEYS = [
    "experiment_id",
    "date_utc",
    "dataset_revision",
    "dataset_sha256",
    "model_id",
    "quantization",
    "prompt_version",
    "memory_arch",
    "retrieval_config",
    "eval_config",
    "hardware",
    "runtime_s",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "seed",
]


def hardware_info() -> dict:
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "system": platform.system(),
        "release": platform.release(),
        "node_label": platform.node(),  # rename before publishing if identifying
    }


class RunLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def append(self, record: dict) -> dict:
        record = dict(record)
        record.setdefault("date_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        missing = [k for k in REQUIRED_KEYS if k not in record]
        if missing:
            raise KeyError(f"ledger record missing required keys: {missing}")
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def read_all(self) -> list[dict]:
        records = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def filter(self, **eq) -> list[dict]:
        """Records where every given key equals the given value (dotted for nested)."""
        out = []
        for r in self.read_all():
            ok = True
            for key, val in eq.items():
                cur = r
                for part in key.split("."):
                    if isinstance(cur, dict) and part in cur:
                        cur = cur[part]
                    else:
                        cur = None
                        break
                if cur != val:
                    ok = False
                    break
            if ok:
                out.append(r)
        return out
