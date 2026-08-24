"""Unit tests for the JSONL loader and validator (no external services)."""

from __future__ import annotations

import pytest

from rag.ingestion.loader import JsonlLoader
from rag.ingestion.validator import DatasetValidator
from tests.conftest import jsonl_records


def _write_jsonl(tmp_path, records) -> str:
    path = tmp_path / "docs.jsonl"
    path.write_text(jsonl_records(records), encoding="utf-8")
    return str(path)


def test_loader_iter_textual_skips_gallery(tmp_path):
    records = [
        {"id": "a", "content_type": "home", "language": "en", "text": "Hello NMU"},
        {"id": "b", "content_type": "gallery", "text": None},
        {"id": "c", "content_type": "news", "language": "en", "text": "News item"},
    ]
    loader = JsonlLoader(_write_jsonl(tmp_path, records))
    textual = list(loader.iter_textual())
    assert [d.id for d in textual] == ["a", "c"]


def test_loader_dedupes_by_id(tmp_path):
    records = [
        {"id": "dup", "content_type": "home", "language": "en", "text": "first"},
        {"id": "dup", "content_type": "home", "language": "en", "text": "second"},
    ]
    loader = JsonlLoader(_write_jsonl(tmp_path, records))
    docs = list(loader.iter_textual())
    assert len(docs) == 1
    assert docs[0].text == "first"


def test_loader_preserves_extra_fields(tmp_path):
    records = [
        {"id": "x", "content_type": "about", "language": "en", "text": "t", "custom": 42}
    ]
    loader = JsonlLoader(_write_jsonl(tmp_path, records))
    raw = next(loader.iter_raw())
    assert raw.extra.get("custom") == 42


def test_loader_malformed_line_raises(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"id": "ok", "text": "fine"}\n{"broken json\n', encoding="utf-8")
    loader = JsonlLoader(str(path))
    with pytest.raises(ValueError):
        list(loader.iter_raw())


def test_validator_flags_duplicate_ids(tmp_path):
    records = [
        {"id": "a", "content_type": "home", "language": "en", "text": "x" * 100},
        {"id": "a", "content_type": "home", "language": "en", "text": "y" * 100},
    ]
    validator = DatasetValidator(_write_jsonl(tmp_path, records))
    result = validator.validate()
    assert result.errors
    assert "a" in result.duplicate_ids


def test_validator_gallery_without_text_is_ok(tmp_path):
    records = [
        {"id": "img", "content_type": "gallery", "text": None, "url": "https://nmu.edu.eg/img"}
    ]
    validator = DatasetValidator(_write_jsonl(tmp_path, records))
    result = validator.validate()
    assert not result.errors


def test_validator_empty_file(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text("", encoding="utf-8")
    result = DatasetValidator(str(path)).validate()
    assert result.total_records == 0
    assert result.is_valid()


def test_validator_rejects_missing_id(tmp_path):
    records = [{"content_type": "home", "text": "no id here"}]
    validator = DatasetValidator(_write_jsonl(tmp_path, records))
    result = validator.validate()
    assert result.errors


def test_validator_reports_malformed_lines(tmp_path):
    path = tmp_path / "docs.jsonl"
    path.write_text('{"id": "ok", "text": "fine"}\nnot-json\n', encoding="utf-8")
    result = DatasetValidator(str(path)).validate()
    assert result.malformed_lines == [2]
    assert not result.is_valid()