"""Unit tests for the boilerplate text filter."""

from __future__ import annotations

from rag.ingestion.text_filter import TextFilter


def _clean(text: str) -> str:
    return TextFilter().clean(text)


def test_arabic_footer_removed():
    text = (
        "هذا هو المحتوى الفعلي للصفحة.\n"
        "موقع الجامعة\n"
        "روابط سريعة\n"
        "جميع الحقوق محفوظة\n"
    )
    cleaned = _clean(text)
    assert "المحتوى الفعلي" in cleaned
    assert "موقع الجامعة" not in cleaned
    assert "روابط سريعة" not in cleaned
    assert "جميع الحقوق محفوظة" not in cleaned


def test_english_footer_removed():
    text = (
        "Welcome to New Mansoura University.\n"
        "Our university location\n"
        "All Rights Reserved\n"
    )
    cleaned = _clean(text)
    assert "Welcome to New Mansoura University" in cleaned
    assert "All Rights Reserved" not in cleaned


def test_header_noise_removed():
    text = (
        "01070004148 - 01070004149\n"
        "Sat - Thu: 9 AM - 4 PM\n"
        "English\n"
        "العربية\n"
        "Home\n"
        "الرئيسية\n"
        "Welcome to the real page content here.\n"
    )
    cleaned = _clean(text)
    assert "Welcome to the real page content" in cleaned
    assert "01070004148" not in cleaned
    assert "Home" not in cleaned


def test_footer_marker_not_truncating_body():
    # "تابعنا على" appears near the start as nav; the real footer marker
    # must only cut in the trailing portion of the document.
    text = (
        "جامعة المنصورة الجديدة\n"
        + "\n".join(f"فقرة رقم {i} عن الجامعة وأهدافها وبرامجها." for i in range(40))
        + "\nموقع الجامعة\nتواصل معنا\nجميع الحقوق محفوظة © 2025\n"
    )
    cleaned = _clean(text)
    assert len(cleaned) > 800
    assert "فقرة رقم" in cleaned


def test_empty_text_returns_empty():
    assert _clean("") == ""
    assert _clean("   \n  ") == ""


def test_no_markers_keeps_text():
    text = "Hello world, this is just content without any boilerplate markers."
    assert _clean(text) == text.strip()