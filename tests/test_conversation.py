"""Conversation follow-up resolution regression tests."""

from rag.query.conversation import resolve_conversation


def test_arabic_program_followup_inherits_engineering_faculty():
    history = [
        {
            "role": "user",
            "content": "مين عميد كلية الهندسة في جامعة المنصورة الجديدة؟",
        },
        {
            "role": "assistant",
            "content": "عميدها وفق بيانات الجامعة هو أ.د وائل صديق.",
        },
    ]

    context = resolve_conversation("وإيه البرامج الموجودة فيها؟", history)

    assert context.active
    assert context.understanding is not None
    assert context.understanding.intent == "PROGRAM"
    assert context.understanding.faculty == "engineering"
    assert "برامج" in context.retrieval_question
    assert "الهندسة" in context.retrieval_question


def test_fresh_explicit_question_does_not_inherit_previous_faculty():
    history = [
        {"role": "user", "content": "ما هي برامج كلية الهندسة؟"},
        {"role": "assistant", "content": "إجابة سابقة"},
    ]

    context = resolve_conversation("ما هي برامج كلية الطب؟", history)

    assert not context.active
