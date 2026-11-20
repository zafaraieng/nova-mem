"""Memory system abstractions and shared machinery.

A *memory system* turns (question, reader) into an answer plus a full
measurement trace. All systems emit the official hypothesis JSONL fields
(``question_id``, ``hypothesis``) plus NOVA-Mem's instrumentation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional

from ..data.loader import Question
from ..models.cache import GenerationCache
from ..models.client import InferenceClient
from . import prompts as P


@dataclass
class AnswerResult:
    question_id: str
    hypothesis: str
    prompt: str
    retrieval_trace: list[dict] = field(default_factory=list)
    timings: dict = field(default_factory=dict)  # seconds per stage
    tokens: dict = field(default_factory=dict)  # tokens per stage
    storage_bytes: int = 0
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def reader_generate(
    reader: InferenceClient,
    prompt: str,
    *,
    cache: Optional[GenerationCache] = None,
    max_tokens: int = 500,
    stage: str = "generate",
) -> tuple[str, dict, dict]:
    """Call the reader (or its cache). Returns (text, timings, tokens)."""
    seed = 0  # fixed sampling seed; temperature 0
    context_length = getattr(reader, "num_ctx", None)
    cached = None
    if cache is not None:
        cached = cache.get(
            reader.model_id,
            reader.backend,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
            context_length=context_length,
        )
    if cached is not None:
        timings = {
            f"{stage}_s": 0.0,
            f"{stage}_cached": True,
            f"{stage}_cached_source_s": float(cached.get("latency_s", 0.0)),
        }
        cached_server_timings = cached.get("extra", {}).get("server_timings_s", {})
        for server_stage, duration_s in cached_server_timings.items():
            if duration_s is not None:
                timings[f"{stage}_cached_{server_stage}_s"] = duration_s
        return (
            cached["text"],
            timings,
            {
                f"{stage}_prompt": cached["prompt_tokens"],
                f"{stage}_completion": cached["completion_tokens"],
            },
        )
    result = reader.generate(prompt, max_tokens=max_tokens, temperature=0.0, seed=seed)
    if cache is not None:
        cache.put(
            reader.model_id,
            reader.backend,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            seed=seed,
            result=result.to_dict(),
            context_length=context_length,
        )
    timings = {f"{stage}_s": result.latency_s, f"{stage}_cached": False}
    for server_stage, duration_s in result.extra.get("server_timings_s", {}).items():
        if duration_s is not None:
            timings[f"{stage}_{server_stage}_s"] = duration_s
    return (
        result.text,
        timings,
        {f"{stage}_prompt": result.prompt_tokens, f"{stage}_completion": result.completion_tokens},
    )


def merge_dicts(*dicts: dict) -> dict:
    out: dict = {}
    for d in dicts:
        out.update(d)
    return out


class MemorySystem(ABC):
    """Abstract memory system."""

    name: str

    @abstractmethod
    def answer(
        self,
        question: Question,
        reader: InferenceClient,
        cache: Optional[GenerationCache] = None,
    ) -> AnswerResult: ...

    # ------------------------------------------------------------------
    def _final_answer(
        self,
        question: Question,
        history_string: str,
        reader: InferenceClient,
        cache: Optional[GenerationCache],
        timings: dict,
        tokens: dict,
        template: str = P.ANSWER_WITH_HISTORY,
        retrieval_trace: Optional[list[dict]] = None,
        storage_bytes: int = 0,
        extras: Optional[dict] = None,
        max_tokens: int = 500,
    ) -> AnswerResult:
        prompt = template.format(history_string, question.question_date, question.question)
        text, t_gen, tok_gen = reader_generate(reader, prompt, cache=cache, max_tokens=max_tokens)
        return AnswerResult(
            question_id=question.question_id,
            hypothesis=text,
            prompt=prompt,
            retrieval_trace=retrieval_trace or [],
            timings=merge_dicts(timings, t_gen),
            tokens=merge_dicts(tokens, tok_gen),
            storage_bytes=storage_bytes,
            extras=extras or {},
        )


def chunks_from_sessions(sessions) -> list[tuple[str, list[dict]]]:
    """(date, messages) chunks in dataset order, official message dicts."""
    chunks = []
    for s in sessions:
        chunks.append(
            (
                s.date,
                P.strip_has_answer([{"role": m.role, "content": m.content} for m in s.messages]),
            )
        )
    return chunks


def sort_chunks_by_date(chunks: list[tuple]) -> list[tuple]:
    return sorted(chunks, key=lambda x: x[0])


def trace_from_hits(hits, evidence_ids: set[str]) -> list[dict]:
    """Retrieval trace aligned with metrics: official evidence ids."""
    return [
        {
            "rank": h.rank,
            "corpus_id": h.corpus_id,
            "session_id": h.session_id,
            "score": round(h.score, 6),
            "is_evidence": h.session_id in evidence_ids,
        }
        for h in hits
    ]
