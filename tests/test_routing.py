"""Unit tests for the deterministic query router (no LLM involved)."""

from __future__ import annotations

from rag.retrieval.query_normalizer import apply_aliases, normalize_query
from rag.routing.router import QueryRouter
from rag.routing.rules import CATEGORY_MAP, FACULTY_ALIASES, INTENT_KEYWORDS

router = QueryRouter()


def test_route_english_faculty_list():
    r = router.route("What faculties does New Mansoura University have?")
    assert r.intent in ("FACULTY", "LIST", "PROGRAM")
    assert r.language == "en"
    assert r.confidence >= 0.5


def test_route_arabic_admission_faculty():
    r = router.route("ما شروط القبول في كلية الطب؟")
    assert r.intent == "ADMISSION"
    assert r.category == "admissions"
    assert r.faculty == "medicine"
    assert r.language == "ar"
    assert r.confidence >= 0.7


def test_route_specific_faculty_overview_is_confident():
    r = router.route("احكيلي عن كلية الطب")
    assert r.intent == "FACULTY"
    assert r.faculty == "medicine"
    assert r.faculty_id == "7"
    assert r.confidence >= 0.8


def test_dean_query_includes_faculty_pages_in_route_scope():
    r = router.route("مين عميد كلية الهندسة؟")
    assert r.intent == "PERSON"
    assert r.faculty == "engineering"
    assert "faculty" in r.category_types


def test_route_arabic_admission_health_sciences_longest_alias_wins():
    r = router.route("ما شروط القبول في كلية العلوم الصحية التطبيقية؟")
    # "العلوم الصحية التطبيقية" must beat the generic "العلوم" alias.
    assert r.faculty == "applied-health-sciences"


def test_route_mixed_language():
    r = router.route("ايه برامج Faculty of Artificial Intelligence؟")
    assert r.intent == "PROGRAM"
    assert r.language == "mixed"


def test_route_colloquial_arabic_tuition():
    r = router.route("مصاريف كلية الطب كام؟")
    assert r.intent == "TUITION"
    assert r.faculty == "medicine"
    assert r.language == "ar"


def test_route_english_how_much_faculty_tuition():
    r = router.route("how much is medicine?")
    assert r.intent == "TUITION"
    assert r.faculty == "medicine"


def test_route_location_english():
    r = router.route("Where is New Mansoura University located?")
    assert r.intent == "LOCATION"
    assert r.language == "en"


def test_route_arabic_variant_hamza():
    r1 = router.route("أين تقع جامعة المنصورة الجديدة؟")
    r2 = router.route("اين تقع جامعة المنصورة الجديدة؟")
    assert r1.intent == r2.intent
    assert r1.intent == "LOCATION"


def test_route_multi_intent_not_overly_confident():
    r = router.route("مين رئيس الجامعة وأين تقع الجامعة؟")
    assert r.is_multi
    assert r.confidence < 0.8


def test_route_multi_category_is_general():
    r = router.route("ما الكليات والبرامج المتاحة؟")
    assert r.is_multi
    assert r.category == "general" or len(r.category_types) > 1


def test_route_no_match_low_confidence_fact():
    r = router.route("Hello world example text.")
    assert r.intent == "FACT"
    assert r.confidence < 0.5


def test_route_faculty_aliases_cover_canonical_keys():
    for key in ("medicine", "engineering", "law", "business"):
        assert key in FACULTY_ALIASES


def test_category_map_covers_intents():
    for intent in INTENT_KEYWORDS:
        assert intent in CATEGORY_MAP, f"intent {intent} missing from category map"


# --- NMU name-variant aliases (perf: canonical BM25 matching) ------------------


def test_alias_university_name_variants():
    assert "New Mansoura University" in apply_aliases("Where is NMU located?")
    assert "New Mansoura University" in apply_aliases("newmansoura")
    assert "جامعة المنصورة الجديدة" in apply_aliases("جامعه المنصوره الجديده")
    assert "جامعة المنصورة" in apply_aliases("جامعه المنصوره")


def test_normalize_query_applies_aliases():
    n1 = normalize_query("جامعه المنصوره الجديده")
    n2 = normalize_query("جامعة المنصورة الجديدة")
    assert n1 == n2
    assert "جامعة المنصورة الجديدة" in n1


def test_normalize_query_english_abbreviation():
    n1 = normalize_query("Where is NMU located?")
    n2 = normalize_query("Where is New Mansoura University located?")
    assert n1 == n2


def test_route_computer_science_aliases():
    router = QueryRouter()
    for query in (
        "ما هي تخصصات كلية الحاسب؟",
        "ما هي برامج Computer Science and Engineering في NMU؟",
        "CSE programs",
    ):
        route = router.route(query)
        assert route.faculty == "computer-science-and-engineering"
