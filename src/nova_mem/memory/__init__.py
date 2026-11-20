"""Memory architectures A–E (see docs/reproduction_plan.md §4.1)."""

from .base import AnswerResult, MemorySystem
from .prompts import (
    ANSWER_FACTS_REPLACE,
    ANSWER_NO_RETRIEVAL,
    ANSWER_WITH_HISTORY,
    CON_PROMPT,
    NOVA_FACT_EXTRACTION,
    REGISTRY_VERSION,
    judge_prompt,
    render_history_json,
)
from .structured import StructuredMemorySystem
from .systems import (
    FullContextSystem,
    NoMemorySystem,
    OracleContextSystem,
    RAGRerankSystem,
    RAGSystem,
)

__all__ = [
    "AnswerResult",
    "MemorySystem",
    "NoMemorySystem",
    "FullContextSystem",
    "OracleContextSystem",
    "RAGSystem",
    "RAGRerankSystem",
    "StructuredMemorySystem",
    "prompts",
    "ANSWER_NO_RETRIEVAL",
    "ANSWER_WITH_HISTORY",
    "ANSWER_FACTS_REPLACE",
    "CON_PROMPT",
    "NOVA_FACT_EXTRACTION",
    "judge_prompt",
    "render_history_json",
    "REGISTRY_VERSION",
]
