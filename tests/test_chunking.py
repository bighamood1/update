"""Unit tests for the chunker (pure logic, no embeddings)."""

from __future__ import annotations

from tests.conftest import sample_document


def test_home_chunks_respect_size(chunker):
    doc = sample_document(text="Word. " * 500, content_type="home")
    chunks = chunker.chunk_document(doc)
    assert chunks
    assert all(len(c.text) <= chunker.chunk_size + 5 for c in chunks)


def test_chunk_ids_are_stable(chunker):
    doc = sample_document(text="Sentence one. Sentence two. Sentence three. " * 30)
    a = [c.chunk_id for c in chunker.chunk_document(doc)]
    b = [c.chunk_id for c in chunker.chunk_document(doc)]
    assert a == b


def test_faq_keeps_qa_units(chunker):
    text = (
        "ما هي كليات جامعة المنصورة الجديدة؟\n"
        "يمكنك الاطلاع على كليات الجامعة من خلال الموقع الرسمي.\n"
        "ما هي مصروفات الدراسة؟\n"
        "تختلف المصروفات حسب الكلية.\n"
    )
    doc = sample_document(content_type="faq", language="ar", text=text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) >= 2


def test_nav_menu_list_is_preserved(chunker):
    # A run of heading-like short lines (a faculty menu) must be kept as one
    # chunk instead of being dropped as sub-40-char fragments.
    menu = "\n".join(
        ["Faculties & Programs", "Business", "Law", "Engineering", "Medicine"]
    )
    doc = sample_document(content_type="home", text=menu)
    chunks = chunker.chunk_document(doc)
    assert chunks
    assert any("Business" in c.text and "Engineering" in c.text for c in chunks)


def test_empty_document_yields_no_chunks(chunker):
    doc = sample_document(text="   ")
    assert chunker.chunk_document(doc) == []


def test_provenance_carried_into_chunks(chunker):
    doc = sample_document(
        content_type="tuition",
        url="https://nmu.edu.eg/en/students/tuition",
        title="Tuition fees | NMU",
        faculty="Engineering",
        program="Mechatronics",
    )
    chunk = chunker.chunk_document(doc)[0]
    assert chunk.source_url == "https://nmu.edu.eg/en/students/tuition"
    assert chunk.title == "Tuition fees | NMU"
    assert chunk.content_type == "tuition"
    assert chunk.faculty == "Engineering"
    assert chunk.document_hash == "abc123"


def test_section_and_ordering_metadata(chunker):
    text = (
        "Overview paragraph with a few words.\n\n"
        "Heading One\n"
        "Body under heading one.\n\n"
        "Heading Two\n"
        "Body under heading two.\n"
    )
    doc = sample_document(content_type="about", text=text)
    chunks = chunker.chunk_document(doc)
    assert all(c.parent_document_id == doc.id for c in chunks)
    assert all(c.section_id for c in chunks)
    assert all(c.section_index is not None for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert all(c.chunk_count == len(chunks) for c in chunks)


def test_colon_intro_list_stays_in_one_section(chunker):
    # A list introducer ending with ":" followed by a heading-like run of list
    # items must stay in the same section (so expansion recovers the list).
    text = (
        "The University is composed of the following faculties:\n"
        "Faculty Of Business\n"
        "Faculty Of Law\n"
        "Faculty Of Engineering\n"
    )
    doc = sample_document(content_type="about", text=text)
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert "Faculty Of Business" in chunks[0].text
    assert "Faculty Of Engineering" in chunks[0].text
    assert chunks[0].section_id == chunks[0].section_id