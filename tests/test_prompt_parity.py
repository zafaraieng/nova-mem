"""CRITICAL parity test: our prompt registry must equal the vendored official code.

Parses the vendored files with ``ast`` (no imports needed) and asserts each
official template in ``nova_mem.memory.prompts`` appears VERBATIM among the
official source's string literals.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nova_mem.memory import prompts as P

REPO = Path(__file__).resolve().parents[1]
VENDORED_GEN = REPO / "third_party" / "longmemeval_official" / "run_generation.py"
VENDORED_EVAL = REPO / "third_party" / "longmemeval_official" / "evaluate_qa.py"


def string_literals(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) > 30:
            out.add(node.value)
    return out


@pytest.mark.skipif(not VENDORED_EVAL.exists(), reason="vendored evaluate_qa.py missing")
def test_judge_prompts_verbatim():
    literals = string_literals(VENDORED_EVAL)
    for template in [
        P.JUDGE_DEFAULT,
        P.JUDGE_TEMPORAL,
        P.JUDGE_KNOWLEDGE_UPDATE,
        P.JUDGE_PREFERENCE,
        P.JUDGE_ABSTENTION,
    ]:
        assert template in literals, f"judge template diverged from official:\n{template[:80]}..."


@pytest.mark.skipif(not VENDORED_GEN.exists(), reason="vendored run_generation.py missing")
def test_answer_prompts_verbatim():
    literals = string_literals(VENDORED_GEN)
    for template in [
        P.ANSWER_WITH_HISTORY,
        P.ANSWER_WITH_HISTORY_COT,
        P.ANSWER_FACTS_REPLACE,
        P.CON_PROMPT,
    ]:
        assert template in literals, f"answer template diverged from official:\n{template[:80]}..."


@pytest.mark.skipif(not VENDORED_GEN.exists(), reason="vendored run_generation.py missing")
def test_history_block_format_verbatim():
    literals = string_literals(VENDORED_GEN)
    official_block = "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n"
    assert official_block in literals

    # and our renderer produces exactly that per chunk
    rendered = P.render_history_json([("2023-01-01", [{"role": "user", "content": "hi"}])])
    expected = official_block.format(
        1, "2023-01-01", "\n" + P._json_dumps([{"role": "user", "content": "hi"}])
    )
    assert rendered == expected


def test_no_retrieval_semantics():
    # official: template '{}' formatted with the question (plus optional literal suffix)
    assert P.ANSWER_NO_RETRIEVAL.format(question="What is my job?") == "What is my job?"


def test_judge_prompt_selection():
    p = P.judge_prompt("temporal-reasoning", "Q", "A", "R", abstention=False)
    assert "off-by-one" in p and p.startswith("I will give you a question")
    p2 = P.judge_prompt("single-session-user", "Q", "A", "R", abstention=False)
    assert "off-by-one" not in p2
    p3 = P.judge_prompt("abstention", "Q", "A", "R", abstention=True)
    assert "unanswerable" in p3
    with pytest.raises(NotImplementedError):
        P.judge_prompt("nonexistent-type", "Q", "A", "R", abstention=False)


def test_registry_version_pinned():
    assert P.REGISTRY_VERSION == "v1"
