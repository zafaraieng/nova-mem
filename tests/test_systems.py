"""Ledger + systems integration tests (mock backends only — no scientific claims)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nova_mem.analysis import REQUIRED_KEYS, RunLedger
from nova_mem.data.loader import Message, Question, Session
from nova_mem.memory import (
    FullContextSystem,
    NoMemorySystem,
    OracleContextSystem,
    RAGRerankSystem,
    RAGSystem,
    StructuredMemorySystem,
)
from nova_mem.models import GenerationCache, MockLLMClient
from nova_mem.reranking import MockReranker
from nova_mem.retrieval import Document, Hit, _HashEmbedder, build_retriever


def _question(n_sessions=4, qid="q1", abstention=False, noans=False):
    prefix = "noans_" if noans else "answer_"
    sessions = []
    for i in range(n_sessions):
        if i == 2:  # the evidence session
            sessions.append(
                Session(
                    session_id="answer_evid1",
                    date="2023-01-03",
                    messages=[
                        Message(role="user", content="I work as a teacher at Lincoln Elementary"),
                        Message(role="assistant", content="That's wonderful!"),
                    ],
                )
            )
        else:
            sessions.append(
                Session(
                    session_id=f"filler_{i}",
                    date=f"2023-01-0{i + 1}",
                    messages=[Message(role="user", content=f"filler message {i} about weather")],
                )
            )
    return Question(
        question_id=qid + ("_abs" if abstention else ""),
        question_type="abstention" if abstention else "single-session-user",
        question="Where do I work?",
        answer="Lincoln Elementary",
        question_date="2023-06-01",
        sessions=sessions,
        answer_session_ids=[prefix + "evid1"],
    )


@pytest.fixture
def reader():
    return MockLLMClient(model_id="mock-reader")


@pytest.fixture
def cache(tmp_path):
    return GenerationCache(tmp_path / "cache")


def test_ledger_schema(tmp_path):
    led = RunLedger(tmp_path / "ledger.jsonl")
    rec = {k: None for k in REQUIRED_KEYS}
    rec.update({"experiment_id": "e1", "memory_arch": "A_no_memory"})
    led.append(rec)
    with pytest.raises(KeyError):
        led.append({"experiment_id": "bad"})
    assert len(led.read_all()) == 1
    assert led.filter(memory_arch="A_no_memory")[0]["experiment_id"] == "e1"


def test_system_a_prompt_is_question(reader, cache):
    q = _question()
    res = NoMemorySystem().answer(q, reader, cache=cache)
    assert res.prompt == "Where do I work?"
    assert res.hypothesis.startswith("MOCK-RESPONSE")
    assert res.timings["generate_s"] >= 0
    assert res.tokens["generate_prompt"] > 0


def test_oracle_uses_evidence_only(reader, cache):
    q = _question()
    res = OracleContextSystem().answer(q, reader, cache=cache)
    assert res.extras["n_sessions"] == 1
    assert "Lincoln Elementary" in res.prompt
    assert "### Session 1:" in res.prompt


def test_full_context_renders_all_sorted(reader, cache):
    q = _question()
    res = FullContextSystem().answer(q, reader, cache=cache)
    assert res.extras["n_sessions"] == 4
    assert res.prompt.count("### Session") == 4
    # date-sorted: 2023-01-01 block precedes 2023-01-04
    assert res.prompt.index("2023-01-01") < res.prompt.index("2023-01-04")


def test_rag_c_with_con_and_trace(reader, cache):
    q = _question()
    system = RAGSystem(retriever_kind="bm25", top_k=2, con=True)
    res = system.answer(q, reader, cache=cache)
    assert len(res.retrieval_trace) == 2
    assert res.extras["n_retrieved"] == 2
    assert "session_summary" in res.prompt  # CoN replaced session content
    assert res.timings.get("con_s", 0) >= 0
    assert res.tokens.get("con_prompt", 0) > 0


@pytest.mark.parametrize("kind", ["dense", "hybrid", "rrf"])
def test_rag_c_runs_stage2_retrievers_end_to_end(kind, reader, cache):
    q = _question()
    system = RAGSystem(
        retriever_kind=kind,
        embedder=_HashEmbedder(),
        top_k=2,
        candidate_pool_size=4,
        con=False,
    )

    res = system.answer(q, reader, cache=cache)

    assert res.extras["n_retrieved"] == 2
    assert len(res.retrieval_trace) == 2
    assert [trace["rank"] for trace in res.retrieval_trace] == [1, 2]


@pytest.mark.parametrize("kind", ["hybrid", "rrf"])
def test_fusion_retrievers_dedupe_session_ids_and_keep_canonical_date(kind):
    docs = [
        Document("dup", "coffee plans with the team", "2024/01/01", "dup"),
        Document("other", "coffee and project updates", "2024/01/03", "other"),
        Document("dup", "coffee plans with the team", "2024/01/02", "dup"),
    ]
    retriever = build_retriever(kind, _HashEmbedder(), candidate_pool_size=3)
    retriever.build(docs)

    hits = retriever.search("coffee plans", top_k=5)

    ids = [hit.corpus_id for hit in hits]
    assert len(ids) == len(set(ids))
    duplicate_hit = next(hit for hit in hits if hit.corpus_id == "dup")
    assert duplicate_hit.date == "2024/01/02"
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


@pytest.mark.parametrize(
    ("kind", "expected_ids", "expected_scores"),
    [
        ("hybrid", ["a", "b"], [0.5, 0.5]),
        (
            "rrf",
            ["b", "a"],
            [1 / 61 + 1 / 62, 1 / 61 + 1 / 63],
        ),
    ],
)
def test_fusion_retrievers_apply_frozen_score_formulas(kind, expected_ids, expected_scores):
    class FixedRetriever:
        def __init__(self, hits):
            self.hits = hits

        def build(self, _docs):
            pass

        def search(self, _query, top_k):
            return self.hits[:top_k]

    docs = [
        Document("a", "alpha", "2024/01/01", "a"),
        Document("b", "bravo", "2024/01/02", "b"),
    ]
    retriever = build_retriever(kind, _HashEmbedder(), candidate_pool_size=2)
    retriever.build(docs)
    retriever._bm25 = FixedRetriever(
        [
            Hit("a", "a", 10.0, 1, "2024/01/01"),
            Hit("b", "b", 0.0, 2, "2024/01/02"),
        ]
    )
    retriever._dense = FixedRetriever(
        [
            Hit("b", "b", 0.9, 1, "2024/01/02"),
            Hit("a", "a", 0.1, 3, "2024/01/01"),
        ]
    )

    hits = retriever.search("unused", top_k=2)

    assert [hit.corpus_id for hit in hits] == expected_ids
    assert [hit.score for hit in hits] == pytest.approx(expected_scores)


def test_stage2_fusion_systems_use_separate_local_configs():
    from nova_mem.config import load_config
    from scripts.run_experiment import build_systems

    systems = build_systems(None, load_config("local"), _HashEmbedder())
    dense = systems["C_dense"]()
    hybrid = systems["C_hybrid"]()
    rrf = systems["C_rrf"]()

    assert dense.retriever.name == "dense:hash-embedder"
    assert hybrid.retriever.name.startswith("hybrid:")
    assert rrf.retriever.name.startswith("rrf:")
    assert rrf.retriever.rrf_k == 60


def test_rag_d_rerank_changes_order(reader, cache):
    q = _question()
    system = RAGRerankSystem(
        reranker=MockReranker(), retriever_kind="bm25", top_n=4, top_k_final=2, con=False
    )
    res = system.answer(q, reader, cache=cache)
    assert res.extras["n_candidates"] == 4
    assert res.extras["n_kept"] == 2
    # mock reranker reverses candidate order; trace ranks must be 1..n
    assert [t["rank"] for t in res.retrieval_trace] == list(range(1, len(res.retrieval_trace) + 1))


def test_structured_e_supersession(tmp_path):
    q = _question()
    # extractor emits two facts for the same (entity, attribute) with different values
    facts = json.dumps(
        [
            {
                "fact": "User works at West High",
                "entity": "user",
                "attribute": "workplace",
                "value": "West High",
                "evidence_span": "I work at West High",
            },
            {
                "fact": "User works at Lincoln Elementary",
                "entity": "user",
                "attribute": "workplace",
                "value": "Lincoln Elementary",
                "evidence_span": "I work as a teacher at Lincoln Elementary",
            },
        ]
    )
    extractor = MockLLMClient(model_id="mock-extractor", respond=lambda p: facts)
    reader = MockLLMClient(model_id="mock-reader")
    system = StructuredMemorySystem(embedder=_HashEmbedder(), top_k=5)
    res = system.answer(q, reader, cache=None, extractor=extractor)
    # every session yields the same two facts; per insert, the newer value supersedes
    assert res.extras["n_superseded"] >= 1
    assert res.extras["n_active_facts"] < res.extras["n_facts_total"]
    assert res.storage_bytes > 0  # SQLite file
    assert "fact" in res.prompt and "evidence_span" in res.prompt


def test_structured_e_supersession_uses_session_dates_not_input_order(monkeypatch, caplog):
    q = Question(
        question_id="q_temporal",
        question_type="knowledge-update",
        question="What city do I live in?",
        answer="Paris",
        question_date="2024/03/01 (Fri) 00:00",
        sessions=[
            Session("newer", "2024/02/02 (Fri) 00:00", [Message("user", "I moved to Paris.")]),
            Session("older", "2024/01/01 (Mon) 00:00", [Message("user", "I lived in Rome.")]),
        ],
        answer_session_ids=["newer"],
    )
    system = StructuredMemorySystem(db_path=None)

    def extract(session, _extractor, _cache):
        value = "Paris" if session.session_id == "newer" else "Rome"
        return (
            [
                {
                    "fact": f"The user lives in {value}.",
                    "entity": "user",
                    "attribute": "city",
                    "value": value,
                    "timestamp": session.date,
                    "session_id": session.session_id,
                    "evidence_span": value,
                }
            ],
            {"tokens": {}, "raw": "[]"},
        )

    monkeypatch.setattr(system, "_extract_session", extract)
    caplog.set_level("INFO", logger="nova_mem.memory.structured")
    conn, db_path, stats = system._build_store(q, extractor=None, cache=None)
    try:
        rows = conn.execute(
            "SELECT value, superseded_by FROM memories ORDER BY memory_id"
        ).fetchall()
    finally:
        conn.close()
        db_path.unlink(missing_ok=True)

    assert rows == [("Rome", 2), ("Paris", None)]
    assert stats["n_superseded"] == 1
    progress = [
        record.getMessage()
        for record in caplog.records
        if record.name == "nova_mem.memory.structured"
        and "E extraction session" in record.getMessage()
    ]
    assert progress[0] == "E extraction session 1/2 started"
    assert progress[2] == "E extraction session 2/2 started"


def test_structured_e_removes_temporary_database_after_answer(monkeypatch, reader):
    from nova_mem.memory import structured

    created_paths = []
    original = structured.tempfile.NamedTemporaryFile

    def capture_tempfile(*args, **kwargs):
        handle = original(*args, **kwargs)
        created_paths.append(handle.name)
        return handle

    monkeypatch.setattr(structured.tempfile, "NamedTemporaryFile", capture_tempfile)
    system = StructuredMemorySystem(embedder=_HashEmbedder(), top_k=5)
    extractor = MockLLMClient(model_id="mock-extractor", respond=lambda _prompt: "[]")
    system.answer(_question(n_sessions=1), reader, cache=None, extractor=extractor)

    assert len(created_paths) == 1
    assert not Path(created_paths[0]).exists()


def test_structured_e_removes_temporary_database_when_extraction_fails(monkeypatch):
    from nova_mem.memory import structured

    created_paths = []
    original = structured.tempfile.NamedTemporaryFile

    def capture_tempfile(*args, **kwargs):
        handle = original(*args, **kwargs)
        created_paths.append(handle.name)
        return handle

    monkeypatch.setattr(structured.tempfile, "NamedTemporaryFile", capture_tempfile)
    system = StructuredMemorySystem()

    def fail_extraction(*_args):
        raise TimeoutError("extractor timeout")

    monkeypatch.setattr(system, "_extract_session", fail_extraction)
    with pytest.raises(TimeoutError, match="extractor timeout"):
        system._build_store(_question(n_sessions=1), extractor=None, cache=None)

    assert len(created_paths) == 1
    assert not Path(created_paths[0]).exists()


def test_structured_e_empty_extractor(tmp_path):
    q = _question()
    extractor = MockLLMClient(model_id="mock-extractor", respond=lambda p: "[]")
    reader = MockLLMClient(model_id="mock-reader")
    system = StructuredMemorySystem(embedder=_HashEmbedder(), top_k=5)
    res = system.answer(q, reader, cache=None, extractor=extractor)
    assert res.extras["n_facts_total"] == 0
    assert res.extras["n_active_facts"] == 0
    assert res.hypothesis  # still answers (with empty fact history)


def test_cache_prevents_repeat_calls(tmp_path):
    calls = []

    def respond(prompt):
        calls.append(prompt)
        return "same-answer"

    reader = MockLLMClient(model_id="mock", respond=respond)
    cache = GenerationCache(tmp_path / "c")
    from nova_mem.memory.base import reader_generate

    a, _, _ = reader_generate(reader, "prompt-x", cache=cache, max_tokens=100)
    b, t_b, _ = reader_generate(reader, "prompt-x", cache=cache, max_tokens=100)
    assert a == b == "same-answer"
    assert len(calls) == 1
    assert t_b["generate_cached"] is True
    assert cache.hits == 1 and cache.misses == 1


def test_generation_cache_roundtrip(tmp_path):
    cache = GenerationCache(tmp_path / "c")
    from nova_mem.models import GenerationResult

    cache.put(
        "m",
        "ollama",
        "p",
        max_tokens=5,
        temperature=0.0,
        seed=0,
        result=GenerationResult("hi", 1, 1, 0.01, "m").to_dict(),
    )
    got = cache.get("m", "ollama", "p", max_tokens=5, temperature=0.0, seed=0)
    assert got["text"] == "hi"
    assert cache.get("m", "ollama", "DIFFERENT", max_tokens=5, temperature=0.0, seed=0) is None


def test_generation_cache_separates_context_lengths(tmp_path):
    cache = GenerationCache(tmp_path / "c")
    result = {"text": "short-context"}
    cache.put(
        "m",
        "ollama",
        "p",
        max_tokens=5,
        temperature=0.0,
        seed=0,
        result=result,
        context_length=16384,
    )

    assert (
        cache.get(
            "m",
            "ollama",
            "p",
            max_tokens=5,
            temperature=0.0,
            seed=0,
            context_length=16384,
        )
        == result
    )
    assert (
        cache.get(
            "m",
            "ollama",
            "p",
            max_tokens=5,
            temperature=0.0,
            seed=0,
            context_length=32768,
        )
        is None
    )


def test_reader_generate_preserves_server_timings_but_not_cached_time(tmp_path):
    from nova_mem.memory.base import reader_generate
    from nova_mem.models import GenerationResult

    class TimedReader:
        model_id = "timed-reader"
        backend = "ollama"

        def generate(self, prompt, **kwargs):
            return GenerationResult(
                text="answer",
                prompt_tokens=4,
                completion_tokens=1,
                latency_s=5.0,
                model_id=self.model_id,
                backend=self.backend,
                extra={
                    "server_timings_s": {"prompt_eval": 2.0, "generation_eval": 1.0, "total": 4.0}
                },
            )

    cache = GenerationCache(tmp_path / "timed-cache")
    reader = TimedReader()
    _, live_timings, _ = reader_generate(reader, "prompt", cache=cache)
    _, cached_timings, _ = reader_generate(reader, "prompt", cache=cache)

    assert live_timings["generate_prompt_eval_s"] == 2.0
    assert live_timings["generate_generation_eval_s"] == 1.0
    assert live_timings["generate_total_s"] == 4.0
    assert live_timings["generate_cached"] is False
    assert cached_timings["generate_cached"] is True
    assert cached_timings["generate_cached_source_s"] == 5.0
    assert cached_timings["generate_cached_prompt_eval_s"] == 2.0
    assert "generate_prompt_eval_s" not in cached_timings
