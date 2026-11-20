"""Utility helpers: hashing, timing, logging."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

_LOG_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format=_LOG_FMT, stream=sys.stderr)
    return logger


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a UTF-8 string (cache keys, content digests)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash(obj) -> str:
    """Deterministic hash of a JSON-serializable object (configs, prompts)."""
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


@contextmanager
def timer():
    """Context manager yielding a dict filled with elapsed wall-clock seconds.

    Usage::

        with timer() as t:
            do_work()
        elapsed = t["s"]
    """
    out = {"s": 0.0}
    start = time.perf_counter()
    try:
        yield out
    finally:
        out["s"] = time.perf_counter() - start
