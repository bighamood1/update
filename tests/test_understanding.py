"""Unit tests for the Phase-2 upgrade: query understanding, expansion,
multi-intent splitting, feedback validation, the runtime SQLite store, and the
soft quality validator. All tests are pure Python (no scipy, no models)."""

from __future__ import annotations

import json

import numpy as np

from rag.cache.store import RuntimeStore
from rag.feedback.store import FeedbackStore
from rag.generation.prompts import SYSTEM_PROMPT
from rag.quality.validator import evaluate_answer
from rag.query.expansion import retrieval_variants
from rag.query.multi_intent import split_question
from rag.query.understanding import QueryUnderstanding, understand


# --- query understanding -------------------------------------------------------


def test_understand_tuition_arabic():
    u = understand("ما هي رسوم الدراسة؟")
    assert u.language == "ar"
    assert u.intent == "TUITION"
    assert u.category == "tuition"
    assert u.normalized_question
    assert u.route is not None


def test_understand_location_english():
    u = understand("Where is New Mansoura University located?")
    assert u.language == "en"
    assert u.intent == "LOCATION"
    assert u.normalized_question


def test_understand_empty_never_raises():
    u = understand("")
    assert u.original_question == ""
    assert u.normalized_question == ""


def test_understand_detects_multi_intent():
    u = understand("ما هي مصاريف كلية الطب وما هي شروط القبول؟")
    assert u.is_multi_intent is True
    assert u.intent in ("TUITION", "ADMISSION", "FACT")


def test_understand_mixed_language_is_reachable():
    u = understand("ايه programs in NMU؟")
    assert u.language == "mixed"


def test_understand_colloquial_fee_question():
    u = understand("مصاريف كلية الطب كام؟")
    assert u.intent == "TUITION"
    assert u.faculty == "medicine"


def test_understanding_to_dict_safe():
    u = understand("ما هي رسوم الدراسة؟")
    d = u.to_dict()
    assert d["intent"] == "TUITION"
    assert d["language"] == "ar"


# --- query expansion -----------------------------------------------------------


def test_retrieval_variants_simple_is_single():
    u = understand("ماهي شروط القبول بكلية الهندسة؟")
    assert u.confidence >= 0.8 and not u.is_multi_intent
    variants = retrieval_variants(u, u.original_question)
    assert len(variants) == 1
    assert variants[0]


def test_retrieval_variants_bounded_and_deduped():
    u = understand("ما هي مصاريف كلية الطب وما هي شروط القبول؟")
    variants = retrieval_variants(u, u.original_question)
    assert 1 <= len(variants) <= 3
    assert len(set(variants)) == len(variants)
    assert all(v.strip() for v in variants)


def test_retrieval_variants_scholarship_paraphrase_high_recall():
    u = understand("هل الجامعة بتقدم منح؟")
    variants = retrieval_variants(u, u.original_question)
    joined = "\n".join(variants)
    assert u.intent == "SCHOLARSHIP"
    assert len(variants) >= 3
    assert "المنح الدراسية" in joined
    assert "financial aid" in joined


def test_retrieval_variants_cse_mixed_query_high_recall():
    u = understand("ما هي برامج Computer Science and Engineering في NMU؟")
    variants = retrieval_variants(u, u.original_question)
    joined = "\n".join(variants)
    assert u.faculty == "computer-science-and-engineering"
    assert "Computer Science" in joined
    assert "كلية علوم وهندسة الحاسب" in joined


def test_understanding_refines_faculty_specialization_to_program():
    u = understand("ما هي تخصصات كلية الحاسب؟")
    assert u.intent == "PROGRAM"
    assert u.faculty == "computer-science-and-engineering"


def test_available_programs_in_named_faculty_is_not_location():
    u = understand("ايه البرامج الموجودة في كلية علوم وهندسة الحاسب؟")
    assert u.intent == "PROGRAM"
    assert u.faculty == "computer-science-and-engineering"


def test_list_word_does_not_override_named_faculty_program_intent():
    u = understand("اذكر برامج كلية الحاسبات")
    assert u.intent == "PROGRAM"
    assert u.route.intent == "PROGRAM"
    assert u.faculty == "computer-science-and-engineering"


def test_named_faculty_program_paraphrases_are_confident_single_intent():
    for question in (
        "كلية الحاسب فيها برامج ايه؟",
        "اذكر برامج كلية الحاسبات",
        "ما أقسام كلية علوم وهندسة الحاسب؟",
    ):
        u = understand(question)
        assert u.intent == "PROGRAM"
        assert u.is_multi_intent is False
        assert u.confidence >= 0.8


# --- multi-intent splitting ----------------------------------------------------


def test_split_question_two_parts():
    u = understand("ما هي مصاريف كلية الطب وما هي شروط القبول؟")
    parts = split_question(u.original_question, u)
    assert len(parts) == 2
    assert "مصاريف" in parts[0]
    assert "شروط" in parts[1]


def test_split_question_unchanged_when_not_multi():
    u = understand("Where is the university located?")
    parts = split_question(u.original_question, u)
    assert parts == [u.original_question]


def test_split_question_returns_original_when_parts_unquestionlike():
    u = understand("اقرأ و اكتب ملاحظات")
    parts = split_question(u.original_question, u)
    assert len(parts) == 1


def test_split_question_english_and_what():
    u = understand("What is the tuition and what are the requirements?")
    parts = split_question(u.original_question, u)
    assert len(parts) == 2
    assert "tuition" in parts[0]
    assert "requirements" in parts[1]


# --- feedback validation -------------------------------------------------------


class _FakeStore:
    def __init__(self):
        self.calls: list[tuple] = []

    def add_feedback(self, question_id, rating, reason=None):
        self.calls.append((question_id, rating, reason))
        return "fb-1"


def test_feedback_requires_question_id():
    rec = FeedbackStore(store=_FakeStore()).submit("", "useful")
    assert rec.ok is False
    assert "question_id" in rec.message


def test_feedback_rejects_unknown_rating():
    rec = FeedbackStore(store=_FakeStore()).submit("q1", "great")
    assert rec.ok is False


def test_feedback_validates_reason_per_rating():
    store = FeedbackStore(store=_FakeStore())
    bad = store.submit("q1", "not_useful", reason="made_up_reason")
    assert bad.ok is False
    good = store.submit("q1", "not_useful", reason="incorrect_answer")
    assert good.ok is True


def test_feedback_store_stores_record():
    fake = _FakeStore()
    rec = FeedbackStore(store=fake).submit("q1", "useful")
    assert rec.ok is True and rec.feedback_id == "fb-1"
    assert fake.calls == [("q1", "useful", None)]


def test_feedback_accepts_somewhat_and_legacy_medium_alias():
    fake = _FakeStore()
    store = FeedbackStore(store=fake)
    partial = store.submit("q1", "somewhat", reason="incomplete")
    legacy = store.submit("q2", "medium", reason="unclear")
    assert partial.ok is True
    assert legacy.ok is True
    assert fake.calls == [
        ("q1", "somewhat", "incomplete"),
        ("q2", "somewhat", "unclear"),
    ]


# --- runtime SQLite store ------------------------------------------------------


def test_runtime_store_event_feedback_stats_export(tmp_path):
    store = RuntimeStore(db_path=str(tmp_path / "runtime.db"), enabled=True)
    store.record_question_event(
        question_id="q-test-1", kb_version="kb-1",
        question="ما مصاريف الطب؟", normalized_question="ما مصاريف الطب",
        language="ar", intent="TUITION", category="tuition",
        faculty="medicine", is_multi_intent=False, answer="اجابة موثقة",
        sources=[{"title": "t", "url": "https://nmu.edu.eg/en"}],
        latency_ms=12.3, cache_hit=False, cache_entry_id=None,
        retrieval_meta={"strategy": "routed"},
    )
    fb_id = store.add_feedback("q-test-1", "useful")
    assert fb_id is not None
    assert store.get_feedback("q-test-1")["rating"] == "useful"

    stats = store.stats()
    assert stats["questions"] == 1
    assert stats["feedback"] == 1
    assert stats["ratings"].get("useful") == 1

    rows = store.export_training_rows()
    assert len(rows) == 1
    assert rows[0]["question"] == "ما مصاريف الطب؟"
    assert rows[0]["rating"] == "useful"


def test_runtime_feedback_updates_cache_status_and_strategy_learning(tmp_path):
    store = RuntimeStore(db_path=str(tmp_path / "runtime.db"), enabled=True)
    cache_id = store.upsert_cache_entry(
        kb_version="kb-1",
        embedding=np.asarray([1.0, 0.0], dtype=np.float32),
        question="What are medicine fees?",
        normalized_question="what are medicine fees",
        language="en",
        intent="TUITION",
        category="tuition",
        faculty="medicine",
        answer="Cached answer",
        sources=[{"title": "Fees", "url": "https://nmu.edu.eg/fees", "chunk_id": "c1"}],
        quality_score=0.8,
    )
    semantic_group = store.semantic_group(
        intent="TUITION",
        language="en",
        category="tuition",
        faculty="medicine",
    )
    meta = {
        "strategy": "routed",
        "strategy_signature": "strategy_test_bad",
        "semantic_group": semantic_group,
        "query_variants": ["medicine fees"],
        "retrieved_chunk_ids": ["c1"],
        "retrieval_scores": [{"chunk_id": "c1", "score": 0.81}],
        "reranker_used": True,
        "generation_route": "llm",
        "context_strategy": "standard",
        "answer_format": "freeform",
        "coverage": {"ok": True},
    }
    store.record_question_event(
        question_id="q-bad",
        kb_version="kb-1",
        question="What are medicine fees?",
        normalized_question="what are medicine fees",
        language="en",
        intent="TUITION",
        category="tuition",
        faculty="medicine",
        is_multi_intent=False,
        answer="Cached answer",
        sources=[{"title": "Fees", "url": "https://nmu.edu.eg/fees", "chunk_id": "c1"}],
        latency_ms=3.0,
        cache_hit=True,
        cache_entry_id=cache_id,
        retrieval_meta=meta,
    )

    assert store.add_feedback("q-bad", "not_useful", "incorrect_answer")

    cache_rows = store.find_cache_hits("kb-1")
    assert cache_rows[0]["feedback_status"] == "NOT_USEFUL"
    assert cache_rows[0]["quality_score"] == 0.0

    hints = store.get_strategy_hints(
        kb_version="kb-1",
        normalized_question="what are medicine fees",
        intent="TUITION",
        language="en",
        category="tuition",
        faculty="medicine",
    )
    assert hints and hints[0]["confidence"] < 0
    assert hints[0]["failure_count"] == 1
    assert hints[0]["failure_type"] == "incorrect_answer"
    assert json.loads(hints[0]["source_urls_json"]) == ["https://nmu.edu.eg/fees"]
    assert json.loads(hints[0]["chunk_ids_json"]) == ["c1"]
    paraphrase_hints = store.get_strategy_hints(
        kb_version="kb-1",
        normalized_question="medicine tuition cost",
        intent="TUITION",
        language="en",
        category="tuition",
        faculty="medicine",
    )
    assert paraphrase_hints and paraphrase_hints[0]["failure_count"] == 1


def test_runtime_feedback_records_somewhat_as_partial_strategy(tmp_path):
    store = RuntimeStore(db_path=str(tmp_path / "runtime.db"), enabled=True)
    store.record_question_event(
        question_id="q-partial",
        kb_version="kb-1",
        question="Tell me about admission",
        normalized_question="tell me about admission",
        language="en",
        intent="ADMISSION",
        category="admission",
        faculty=None,
        is_multi_intent=False,
        answer="Partial answer",
        sources=[{"title": "Admission", "url": "https://nmu.edu.eg/admission"}],
        latency_ms=4.0,
        cache_hit=False,
        cache_entry_id=None,
        retrieval_meta={
            "strategy": "routed",
            "strategy_signature": "strategy_test_partial",
            "generation_route": "llm",
            "context_strategy": "compressed",
            "validation_issues": ["too_short"],
        },
    )

    assert store.add_feedback("q-partial", "medium", "incomplete")
    assert store.get_feedback("q-partial")["rating"] == "somewhat"
    hints = store.get_strategy_hints(
        kb_version="kb-1",
        normalized_question="tell me about admission",
        intent="ADMISSION",
        language="en",
        category="admission",
        faculty=None,
    )
    assert hints[0]["partial_count"] == 1
    assert hints[0]["confidence"] > 0


def test_runtime_useful_strategy_hints_generalize_within_semantic_group(tmp_path):
    store = RuntimeStore(db_path=str(tmp_path / "runtime.db"), enabled=True)
    semantic_group = store.semantic_group(
        intent="LOCATION", language="ar", category="location", faculty=None
    )
    store.record_question_event(
        question_id="q-location-good",
        kb_version="kb-1",
        question="أين تقع الجامعة؟",
        normalized_question="اين تقع الجامعة",
        language="ar",
        intent="LOCATION",
        category="location",
        faculty=None,
        is_multi_intent=False,
        answer="تقع في مدينة المنصورة الجديدة.",
        sources=[{"title": "Contact", "url": "https://nmu.edu.eg/ar/contact-us"}],
        latency_ms=2.0,
        cache_hit=False,
        cache_entry_id=None,
        retrieval_meta={
            "semantic_group": semantic_group,
            "strategy_signature": "strategy_location_good",
            "strategy": "routed",
            "generation_route": "fast_path",
            "context_strategy": "deterministic",
        },
    )
    assert store.add_feedback("q-location-good", "useful")
    hints = store.get_strategy_hints(
        kb_version="kb-1",
        normalized_question="فين مكان الجامعة",
        intent="LOCATION",
        language="ar",
        category="location",
        faculty=None,
    )
    assert hints and hints[0]["success_count"] == 1
    assert json.loads(hints[0]["source_urls_json"]) == ["https://nmu.edu.eg/ar/contact-us"]


def test_runtime_latest_feedback_for_exact_query(tmp_path):
    store = RuntimeStore(db_path=str(tmp_path / "runtime.db"), enabled=True)
    semantic_group = store.semantic_group(
        intent="LOCATION",
        language="en",
        category="location",
        faculty=None,
    )
    store.record_question_event(
        question_id="q-approved",
        kb_version="kb-1",
        question="Where is NMU?",
        normalized_question="where is nmu",
        language="en",
        intent="LOCATION",
        category="location",
        faculty=None,
        is_multi_intent=False,
        answer="Approved answer",
        sources=[{"title": "About", "url": "https://nmu.edu.eg/en/about-us"}],
        latency_ms=2.0,
        cache_hit=False,
        cache_entry_id=None,
        retrieval_meta={"semantic_group": semantic_group},
    )
    assert store.add_feedback("q-approved", "useful")
    row = store.latest_feedback_for_query(
        kb_version="kb-1",
        normalized_question="where is nmu",
        semantic_group=semantic_group,
    )
    assert row["question_id"] == "q-approved"
    assert row["rating"] == "useful"
    assert row["answer"] == "Approved answer"


def test_runtime_store_rejects_feedback_for_unknown_question(tmp_path):
    store = RuntimeStore(db_path=str(tmp_path / "runtime.db"), enabled=True)
    assert store.add_feedback("unknown-id", "useful") is None


def test_runtime_store_clusters(tmp_path):
    store = RuntimeStore(db_path=str(tmp_path / "runtime.db"), enabled=True)
    store.record_cluster(kb_version="kb-1", cluster_key="ما مصاريف الطب",
                         question_id="q1", latency_ms=10.0, cache_hit=False)
    store.record_cluster(kb_version="kb-1", cluster_key="ما مصاريف الطب",
                         question_id="q2", latency_ms=9.0, cache_hit=False)
    faqs = store.top_faqs(limit=5)
    assert faqs and faqs[0]["frequency"] == 2


# --- quality validator ----------------------------------------------------------


def test_evaluate_answer_grounded():
    u = understand("What is the tuition?")
    out = evaluate_answer(
        "The tuition is 1000 EGP per year according to the official sources.",
        u, [], None,
    )
    assert out["score"] > 0.0
    assert out["ok"] is True


def test_evaluate_answer_removes_fabricated_url():
    u = understand("What is the tuition?")
    out = evaluate_answer("See https://evil.example.org/x for details.", u, [], None)
    assert "evil.example.org" not in out["cleaned"]


def test_prompts_preserve_grounding_rules():
    assert "Use only the provided evidence" in SYSTEM_PROMPT
    assert "Do not invent facts" in SYSTEM_PROMPT
    assert "Return only the final user-facing answer" in SYSTEM_PROMPT


# --- QueryUnderstanding importability (dataclass shape) -------------------------


def test_query_understanding_fields():
    u = QueryUnderstanding(original_question="q", normalized_question="q")
    assert u.intent == "FACT"
    assert u.is_multi_intent is False
