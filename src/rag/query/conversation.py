"""Conservative conversation-context resolution for follow-up questions.

A follow-up inherits ONLY what the new message omits — and ONLY from the most
recent user turn:

- entity/faculty, when the new message names no faculty of its own;

Language always comes from the current message. Inheritance NEVER overrides
explicit information: if the new message names a different faculty, or is not
a follow-up at all, nothing is inherited.

The resolved context is used for RETRIEVAL (so "وما هي برامجها؟" retrieves the
medicine programs) and as an explicit "conversation context" block in the LLM
prompt (so the model can resolve pronouns). It NEVER feeds the semantic cache
lookup or store: a follow-up is cached under its own intent/category, so
conversation context can never contaminate the cache.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..routing.rules import FACULTY_ALIASES
from .understanding import QueryUnderstanding, understand

_FOLLOWUP_RE = re.compile(
    r"(وماذا عن|و ماذا عن|وماذا|وعن|وما عن|ما عن|و عن|وما هي|وما هو|وما|"
    r"what about|and what|about it|about this|what are its|what is its|"
    r"وبالنسبة|بالنسبة لل|وإيه عن|وايه عن|وإيه|وايه|فيها|فيه|"
    r"its|in it|there)",
    re.IGNORECASE,
)

_QUESTION_WORDS = (
    "ما", "ايه", "إيه", "من", "أين", "كيف", "متى", "كم", "هل", "اذكر",
    "what", "which", "who", "where", "when", "how", "why", "list", "name",
)

_MAX_HISTORY = 8
_MAX_ANSWER_CHARS = 500


@dataclass
class ConversationContext:
    """Resolved follow-up context (``active=False`` when not a follow-up)."""

    prompt_block: str
    retrieval_question: str
    understanding: QueryUnderstanding | None
    active: bool = True


@dataclass
class _Turn:
    role: str
    content: str


def _clean_history(history) -> list[_Turn]:
    """Normalize raw history into an ordered list of turns (best-effort)."""
    turns: list[_Turn] = []
    if not isinstance(history, (list, tuple)):
        return turns
    for item in history[-_MAX_HISTORY:]:
        if not isinstance(item, dict):
            continue
        content = (item.get("content") or item.get("message") or "").strip()
        role = (item.get("role") or item.get("sender") or "").strip().lower()
        if not content or role not in ("user", "assistant"):
            continue
        turns.append(_Turn(role=role, content=content))
    return turns


def _has_question_word(text: str) -> bool:
    low = (text or "").lower()
    return any(q in low for q in _QUESTION_WORDS)


def _is_followup(question: str, last_user: str, cur_u: QueryUnderstanding) -> bool:
    q = (question or "").strip()
    prev = (last_user or "").strip()
    if not q or not prev:
        return False
    if q.lower() == prev.lower():
        return False
    # Explicit reference markers ("وماذا عن", "what about", "its", ...).
    if _FOLLOWUP_RE.search(q):
        return True
    # Elliptical follow-up: short, contains a question word, and carries NO
    # topic keywords of its own. A fresh standalone question that states its
    # own topic is NEVER reinterpreted as a follow-up (no contamination).
    if (
        len(q) <= 45
        and _has_question_word(q)
        and not (cur_u.route and cur_u.route.matched)
    ):
        return True
    return False


def _faculty_label(key: str | None, language: str) -> str:
    """A human-readable faculty alias matching the conversation language."""
    if not key:
        return ""
    aliases = FACULTY_ALIASES.get(key, [])
    if not aliases:
        return ""
    if (language or "").lower() == "ar":
        for a in aliases:
            if re.search(r"[\u0600-\u06FF]", a):
                return a
    return aliases[0]


def _build_prompt_block(last_user: str, last_answer: str) -> str:
    block = (
        "CONVERSATION CONTEXT (use this ONLY to resolve references such as "
        "'it', 'this faculty', 'the university', or omitted subjects). "
        "Do NOT copy it and do NOT let it override the current question."
    )
    lines = [block, f"Previous user question: {last_user}"]
    if last_answer:
        snippet = last_answer.strip()
        if len(snippet) > _MAX_ANSWER_CHARS:
            snippet = snippet[:_MAX_ANSWER_CHARS] + "…"
        lines.append(f"Previous assistant answer (reference only): {snippet}")
    lines.append("All facts must come from the retrieved CONTEXT below.")
    return "\n".join(lines)


def resolve_conversation(
    question: str, history
) -> ConversationContext:
    """Return a follow-up ``ConversationContext`` (inactive when not one).

    Pure, deterministic and never raises. The returned ``understanding`` is
    used ONLY for retrieval; callers must keep using the current
    ``understand(question)`` result for caching / events / validation.
    """
    turns = _clean_history(history)
    if not turns:
        return ConversationContext(
            prompt_block="", retrieval_question=question, understanding=None,
            active=False,
        )
    user_turns = [t for t in turns if t.role == "user"]
    if not user_turns:
        return ConversationContext(
            prompt_block="", retrieval_question=question, understanding=None,
            active=False,
        )
    last_user = user_turns[-1].content
    last_answer = next(
        (t.content for t in reversed(turns) if t.role == "assistant"), ""
    )

    cur_u = understand(question)

    if not _is_followup(question, last_user, cur_u):
        return ConversationContext(
            prompt_block="", retrieval_question=question, understanding=None,
            active=False,
        )

    prev_u = understand(last_user)

    # Inherit the previous faculty ONLY when the follow-up names none of its own.
    inherited_faculty: str | None = None
    if cur_u.faculty is None and prev_u.faculty:
        inherited_faculty = prev_u.faculty

    retrieval_question = question
    if inherited_faculty:
        label = _faculty_label(inherited_faculty, prev_u.language or cur_u.language)
        if label:
            low = question.lower()
            # Pronouns such as "فيها" contain the Arabic location preposition
            # and can make the rule router choose LOCATION even when the user
            # explicitly asks for programs/departments. Build a canonical
            # retrieval-only question; the original wording still goes to the
            # LLM prompt and GUI.
            if any(marker in low for marker in ("برامج", "برنامج", "program", "programs")):
                retrieval_question = (
                    f"ما هي برامج كلية {label}؟"
                    if (cur_u.language or "ar") == "ar"
                    else f"What are the programs of the {label} faculty?"
                )
            elif any(marker in low for marker in ("أقسام", "اقسام", "قسم", "department", "departments")):
                retrieval_question = (
                    f"ما هي أقسام كلية {label}؟"
                    if (cur_u.language or "ar") == "ar"
                    else f"What are the departments of the {label} faculty?"
                )
            else:
                retrieval_question = f"{question} {label}".strip()

    merged_u = understand(retrieval_question) if retrieval_question != question else cur_u

    return ConversationContext(
        prompt_block=_build_prompt_block(last_user, last_answer),
        retrieval_question=retrieval_question,
        understanding=merged_u,
        active=True,
    )
