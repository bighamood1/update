"""Unit tests for query intent classification (no models, no services)."""

from __future__ import annotations

from rag.retrieval.intents import classify_intent, is_list_intent


def test_faculty_list_english():
    assert classify_intent("What faculties does NMU have?") == "FACULTY"
    assert is_list_intent("FACULTY")


def test_faculty_list_arabic():
    assert classify_intent("ما هي كليات جامعة المنصورة الجديدة؟") == "FACULTY"


def test_program_list():
    assert classify_intent("What programs does the faculty of engineering offer?") == "PROGRAM"
    assert is_list_intent("PROGRAM")


def test_explicit_list_marker():
    assert classify_intent("List all available services at NMU.") == "LIST"
    assert classify_intent("اذكر جميع خدمات الجامعة") == "LIST"


def test_location_intent():
    assert classify_intent("Where is New Mansoura University located?") == "LOCATION"
    assert classify_intent("أين تقع جامعة المنصورة الجديدة؟") == "LOCATION"


def test_person_intent():
    assert classify_intent("Who is the president of NMU?") == "PERSON"
    assert classify_intent("من هو رئيس الجامعة؟") == "PERSON"


def test_comparison_intent():
    assert classify_intent("Compare medicine and dentistry at NMU.") == "COMPARISON"


def test_admission_intent():
    assert classify_intent("What are the admission requirements for NMU?") == "ADMISSION"
    assert classify_intent("ما هي شروط القبول؟") == "ADMISSION"


def test_unknown_world_knowledge():
    assert classify_intent("What is the population of Egypt?") == "UNKNOWN"


def test_fact_default():
    assert classify_intent("When was New Mansoura University established?") == "FACT"
    assert not is_list_intent("FACT")


def test_empty_question_is_unknown():
    assert classify_intent("   ") == "UNKNOWN"