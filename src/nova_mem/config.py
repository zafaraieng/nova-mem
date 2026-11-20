"""Configuration loading and freezing.

Configs are YAML files under ``configs/``. A *frozen* config
(``configs/frozen_v0.yaml``) pins every experimental parameter; runs record its
stable hash in the ledger. Dev-only sensitivity sweeps (plan §4.5 E7) apply
``--set key=value`` overrides on top of a config file; overridden runs record
both the override list and the hash of the *overridden* config in the ledger.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .utils import stable_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(name: str) -> dict:
    """Load configs/<name>.yaml."""
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return load_yaml(path)


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge; override wins."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_hash(cfg: dict) -> str:
    return stable_hash(cfg)


def parse_override(spec: str) -> tuple[str, Any]:
    """Parse a ``dotted.key=value`` string into (key, yaml-typed value).

    ``3`` -> int, ``0.5`` -> float, ``true`` -> bool, ``session`` -> str.
    """
    if "=" not in spec:
        raise ValueError(f"override must be KEY=VALUE, got: {spec!r}")
    key, _, raw = spec.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"empty key in override: {spec!r}")
    return key, yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: Any) -> dict:
    """Return a deep copy of ``cfg`` with dotted-path overrides applied.

    ``overrides`` is either a ``{dotted.key: value}`` mapping (typed values)
    or an iterable of ``"dotted.key=value"`` strings. Anything else — e.g. a
    bare ``(key, value)`` tuple — is rejected (ambiguous).

    Every path segment must already exist in ``cfg``: an override that silently
    created a new key would corrupt the sweep audit trail, so typos fail loudly
    (KeyError). Overriding a whole mapping (non-leaf) is rejected for the same
    reason — sweep one scalar at a time.
    """
    if isinstance(overrides, dict):
        pairs = [(str(k), v) for k, v in overrides.items()]
    else:
        pairs = []
        for spec in overrides:
            if not isinstance(spec, str):
                raise TypeError(f"override must be a 'key=value' string, got: {spec!r}")
            pairs.append(parse_override(spec))

    out = copy.deepcopy(cfg)
    for key, value in pairs:
        node: Any = out
        parts = key.split(".")
        for p in parts[:-1]:
            if not isinstance(node, dict) or p not in node:
                raise KeyError(f"override {key!r}: segment {p!r} not found in config (typo?)")
            node = node[p]
        leaf = parts[-1]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"override {key!r}: key {leaf!r} not found in config (typo?)")
        if isinstance(node[leaf], dict):
            raise ValueError(f"override {key!r} targets a mapping; override a leaf scalar instead")
        node[leaf] = value
    return out


def flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten nested dict to dotted keys (for ledger records)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(flatten(v, key))
        else:
            out[key] = v
    return out
