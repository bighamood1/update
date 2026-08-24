"""Dataset validation logic.

Validates the JSONL dataset without modifying it. Gallery/image records are
allowed to have no text — that is not an error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from ..schemas.documents import RawDocument
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

_GALLERY_TYPES = {"gallery", "gallery_album", "image", "images"}
_REQUIRED_FIELDS = ("id",)


@dataclass
class ValidationIssue:
    """A single issue discovered during validation."""

    record_id: str
    field: str
    message: str
    severity: str  # "error" or "warning"


@dataclass
class ValidationResult:
    """Aggregated validation result."""

    total_records: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    duplicate_hashes: list[str] = field(default_factory=list)
    malformed_lines: list[int] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    def is_valid(self) -> bool:
        return len(self.errors) == 0 and not self.malformed_lines


class DatasetValidator:
    """Validate a JSONL dataset and report structured issues."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._seen_ids: dict[str, int] = {}
        self._seen_hashes: dict[str, int] = {}

    def validate(self) -> ValidationResult:
        result = ValidationResult()
        malformed_seen = 0

        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = RawDocument.model_validate_json(line)
                except Exception as exc:
                    # Only report the first 10 malformed lines.
                    if malformed_seen < 10:
                        result.malformed_lines.append(line_no)
                        result.issues.append(
                            ValidationIssue(
                                record_id=f"line-{line_no}",
                                field="record",
                                message=f"Malformed JSONL: {exc}",
                                severity="error",
                            )
                        )
                    malformed_seen += 1
                    continue

                result.total_records += 1
                self._validate_record(record, line_no, result)

        # Duplicate ID / hash detection.
        id_counts = self._seen_ids
        result.duplicate_ids = [k for k, v in id_counts.items() if v > 1]
        hash_counts = self._seen_hashes
        result.duplicate_hashes = [k for k, v in hash_counts.items() if v > 1]

        return result

    def _validate_record(
        self,
        record: RawDocument,
        line_no: int,
        result: ValidationResult,
    ) -> None:
        # ID checks.
        if not record.id:
            result.issues.append(
                ValidationIssue(
                    record_id=f"line-{line_no}",
                    field="id",
                    message="Missing id",
                    severity="error",
                )
            )
        else:
            if record.id in self._seen_ids:
                result.issues.append(
                    ValidationIssue(
                        record_id=record.id,
                        field="id",
                        message="Duplicate id",
                        severity="error",
                    )
                )
            self._seen_ids[record.id] = self._seen_ids.get(record.id, 0) + 1

        # Text: only required for non-gallery records.
        is_gallery = (record.content_type or "").strip().lower() in _GALLERY_TYPES
        text = (record.text or "").strip()
        if not text and not is_gallery:
            result.issues.append(
                ValidationIssue(
                    record_id=record.id,
                    field="text",
                    message="Missing text (non-gallery record)",
                    severity="error",
                )
            )

        # URL checks.
        if not record.url:
            result.issues.append(
                ValidationIssue(
                    record_id=record.id,
                    field="url",
                    message="Missing url",
                    severity="error",
                )
            )

        # Language / content_type checks.
        if not record.language:
            result.issues.append(
                ValidationIssue(
                    record_id=record.id,
                    field="language",
                    message="Missing language",
                    severity="warning",
                )
            )
        if not record.content_type:
            result.issues.append(
                ValidationIssue(
                    record_id=record.id,
                    field="content_type",
                    message="Missing content_type",
                    severity="warning",
                )
            )

        # Metadata must be a dict if present.
        if record.metadata is not None and not isinstance(record.metadata, dict):
            result.issues.append(
                ValidationIssue(
                    record_id=record.id,
                    field="metadata",
                    message="Malformed metadata (not a dict)",
                    severity="error",
                )
            )

        # Hash dedup (warning: may be legitimately shared for identical pages).
        if record.content_hash:
            if record.content_hash in self._seen_hashes:
                result.issues.append(
                    ValidationIssue(
                        record_id=record.id,
                        field="content_hash",
                        message="Duplicate content hash",
                        severity="warning",
                    )
                )
            self._seen_hashes[record.content_hash] = (
                self._seen_hashes.get(record.content_hash, 0) + 1
            )
