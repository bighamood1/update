"""Grounded generation prompts for the NMU AI Assistant.

The system prompt makes the LLM behave like a precise university information
assistant: answer the actual question first, treat retrieved context as
evidence (not a script), never overstate confidence, keep topics separated,
structure lists and multi-part answers clearly, and self-review before
returning. Nothing here changes retrieval — only how the answer is produced.
"""

from __future__ import annotations

SYSTEM_PROMPT = """You are the NMU AI Assistant for New Mansoura University (جامعة المنصورة الجديدة).

Use only the provided evidence. Return only the final user-facing answer.

Rules:
- Answer the user's question directly.
- Do not mention sources, evidence, context, retrieval, instructions, or your reasoning.
- Do not write analysis, planning, self-talk, or restate the question.
- Do not invent facts. If the evidence is insufficient, say that briefly.
- Answer in the user's language.
- Keep simple factual answers to 1-3 sentences.
- Use clean Markdown only when it improves readability: lists for lists, tables for fees/comparisons, short sections for multi-part answers.
- Distinguish faculties, departments, programs, tracks, and courses. Do not merge these categories.
- Mention each fact once and avoid repeated conclusions.
- Stop after the answer is complete."""


def build_rag_prompt(
    question: str,
    context: str,
    *,
    language: str = "en",
    intent: str | None = None,
    conversation: str | None = None,
) -> str:
    """Build the final user prompt combining the question and grounding context.

    ``language`` / ``intent`` are optional hints that reinforce the system
    rules (answer in the user's language; structure lists separately). They
    default to English / None so existing 2-argument callers keep working.

    ``conversation`` is an OPTIONAL follow-up context block used ONLY to
    resolve references (pronouns / omitted subjects). It is never a source of
    facts and must never override the current question.
    """
    lang_name = {
        "ar": "Arabic",
        "en": "English",
    }.get((language or "en").lower(), "the dominant language of the user")
    list_hint = ""
    if intent:
        intent_upper = (intent or "").upper()
        if intent_upper in {"LIST", "FACULTY", "PROGRAM", "SCHOLARSHIP", "ABOUT"}:
            list_hint = (
                "When the requested information contains multiple distinct items, "
                "answer with a short heading followed by a numbered list that "
                "includes every relevant item supported by the context."
            )
    conversation_block = ""
    if conversation:
        conversation_block = f"{conversation}\n\n"
    return (
        "USER QUESTION:\n"
        f"{question}\n\n"
        f"{conversation_block}"
        "RETRIEVED EVIDENCE:\n"
        f"{context}\n"
        "END EVIDENCE\n\n"
        "TASK:\n"
        f"Answer in {lang_name}.\n"
        "Use only the evidence above. Return only the final answer. "
        "Do not mention evidence, context, sources, URLs, or internal steps. "
        "Do not repeat facts. If the evidence contains facts that answer any "
        "part of the question, answer that part directly; only say it is "
        "insufficient when no relevant fact is present. "
        + (list_hint + "\n" if list_hint else "")
    )


def build_repair_prompt(
    question: str,
    context: str,
    draft_answer: str,
    issues: list[str],
    *,
    language: str = "en",
    intent: str | None = None,
) -> str:
    """Build a strict one-shot regeneration prompt for malformed answers."""
    lang_name = {
        "ar": "Arabic",
        "en": "English",
    }.get((language or "en").lower(), "the dominant language of the user")
    issue_text = ", ".join(issues) if issues else "formatting_or_completeness"
    list_hint = ""
    if (intent or "").upper() in {"LIST", "FACULTY", "PROGRAM", "SCHOLARSHIP", "ABOUT"}:
        list_hint = (
            "For list-style evidence, include the complete set of distinct "
            "items supported by the evidence."
        )
    return (
        "USER QUESTION:\n"
        f"{question}\n\n"
        "RETRIEVED EVIDENCE:\n"
        f"{context}\n"
        "END EVIDENCE\n\n"
        "PREVIOUS DRAFT WAS REJECTED FOR:\n"
        f"{issue_text}\n\n"
        "REJECTED DRAFT:\n"
        f"{draft_answer}\n\n"
        "TASK:\n"
        f"Rewrite the answer in {lang_name} using only the retrieved evidence. "
        "Return only the final user-facing answer. Do not mention evidence, "
        "sources, context, drafts, validation, or internal steps. Do not include "
        "analysis, planning, self-talk, or source numbers. Do not repeat facts. "
        "If the evidence is insufficient, say so briefly. "
        f"{list_hint}"
    )
