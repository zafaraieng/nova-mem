"""Prompt registry — versioned, official-parity templates.

Every string below is copied VERBATIM from the official LongMemEval repository
(https://github.com/xiaowu0162/LongMemEval, MIT license), files:

* ``src/generation/run_generation.py``  — answer prompts, CoN prompt, history rendering
* ``src/evaluation/evaluate_qa.py``     — judge (answer-check) prompts

Fidelity is enforced by ``tests/test_prompt_parity.py``, which re-extracts the
templates from the vendored copies under ``third_party/longmemeval_official/``
and diffs them against this module. DO NOT edit the strings; if the official
repo changes, re-vendor the files and bump the registry version.
"""

from __future__ import annotations

import json

REGISTRY_VERSION = "v1"
SOURCE_COMMIT_NOTE = "vendored 2026-11-07 from xiaowu0162/LongMemEval main branch (MIT)"


def _json_dumps(obj) -> str:
    """Official JSON rendering semantics: ``json.dumps`` with default options
    (ensure_ascii=True, separators (', ', ': ')) — matches run_generation.py."""
    return json.dumps(obj)


# ---------------------------------------------------------------------------
# Answer (generation) prompts — run_generation.py
# ---------------------------------------------------------------------------

#: No memory (system A): the official template is literally the question.
ANSWER_NO_RETRIEVAL = "{question}"

ANSWER_WITH_HISTORY = (
    "I will give you several history chats between you and a user. "
    "Please answer the question based on the relevant chat history."
    "\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
)

ANSWER_WITH_HISTORY_COT = (
    "I will give you several history chats between you and a user. "
    "Please answer the question based on the relevant chat history. "
    "Answer the question step by step: first extract all the relevant "
    "information, and then reason over the information to get the answer."
    "\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer (step by step):"
)

#: Official ``merge_key_expansion_into_value == 'replace'`` template (facts as
#: history). Used by NOVA-Mem system E (structured memory) so that E stays
#: within the official prompt family.
ANSWER_FACTS_REPLACE = (
    "I will give you several facts extracted from history chats between you "
    "and a user. Please answer the question based on the relevant facts."
    "\n\n\nHistory Chats:\n\n{}\n\nCurrent Date: {}\nQuestion: {}\nAnswer:"
)

#: NOVA-Mem design (NOT official): fixed extraction prompt for system E's
#: offline fact pass. Versioned here; changes require a registry bump.
NOVA_FACT_EXTRACTION = (
    "You are building a structured memory of a user from their chat history. "
    "Read the following chat session and extract atomic facts about the user, "
    "their relationships, events, and preferences. Each fact must be supported "
    "by this session alone. Include dates when the session or utterances "
    "mention them. Return ONLY a JSON array (no prose). Each element must have "
    'the keys: "fact" (one sentence), "entity" (e.g. the user or a named '
    'person/thing), "attribute" (e.g. occupation, preference, event), "value" '
    '(the attribute value), "evidence_span" (a short verbatim quote from the '
    "session supporting the fact). If the session contains no user facts, "
    "return [].\n\nSession Date: {}\nSession Content:\n{}\nJSON array of facts:"
)

#: Chain-of-Note reading prompt (official `con`).
CON_PROMPT = (
    "I will give you a chat history between you and a user, as well as a "
    "question from the user. Write reading notes to extract all the relevant "
    "user information relevant to answering the answer. If no relevant "
    'information is found, just output "empty". '
    "\n\n\nChat History:\nSession Date: {}\nSession Content:\n{}\n\n"
    "Question Date: {}\nQuestion: {}\n"
    "Extracted note (information relevant to answering the question):"
)

# ---------------------------------------------------------------------------
# History rendering — run_generation.py (history_format='json', session type)
# ---------------------------------------------------------------------------


def render_history_json(chunks: list[tuple[str, object]]) -> str:
    """Render retrieved chunks exactly like the official pipeline.

    ``chunks``: list of ``(date, session_messages)`` where ``session_messages``
    is a list of {role, content} dicts (or the CoN variant
    ``{"session_summary": note}``). Chunks must already be sorted by date.
    """
    history = ""
    for i, (chunk_date, chunk_entry) in enumerate(chunks):
        sess_string = "\n" + _json_dumps(chunk_entry)
        history += "\n### Session {}:\nSession Date: {}\nSession Content:\n{}\n".format(
            i + 1, chunk_date, sess_string
        )
    return history


def strip_has_answer(messages: list[dict]) -> list[dict]:
    """Official cleanup: drop the ``has_answer`` annotation key if present."""
    return [
        {k: v for k, v in m.items() if k != "has_answer"} if isinstance(m, dict) else m
        for m in messages
    ]


# ---------------------------------------------------------------------------
# Judge prompts — evaluate_qa.py get_anscheck_prompt()
# ---------------------------------------------------------------------------

JUDGE_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. "
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

JUDGE_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. In addition, do not penalize off-by-one "
    "errors for the number of days. If the question asks for the number of "
    "days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    "predicting 19 days when the answer is 18), the model's response is still "
    "correct. "
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

JUDGE_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with "
    "an updated answer, the response should be considered as correct as long as "
    "the updated answer is the required answer."
    "\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

JUDGE_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly."
    "\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

JUDGE_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not."
    "\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)

_JUDGE_BY_TYPE = {
    "single-session-user": JUDGE_DEFAULT,
    "single-session-assistant": JUDGE_DEFAULT,
    "multi-session": JUDGE_DEFAULT,
    "temporal-reasoning": JUDGE_TEMPORAL,
    "knowledge-update": JUDGE_KNOWLEDGE_UPDATE,
    "single-session-preference": JUDGE_PREFERENCE,
}


def judge_prompt(
    question_type: str, question: str, answer: str, response: str, abstention: bool
) -> str:
    """Official answer-check prompt selection (evaluate_qa.py semantics)."""
    if abstention:
        return JUDGE_ABSTENTION.format(question, answer, response)
    try:
        template = _JUDGE_BY_TYPE[question_type]
    except KeyError as e:
        raise NotImplementedError(f"No official judge template for {question_type!r}") from e
    return template.format(question, answer, response)
