from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, model_validator


class EntityType(str, Enum):
    PERSON = "person"
    COMPONENT = "component"
    PARAMETER = "parameter"
    FIRMWARE_VERSION = "firmware_version"
    HARDWARE_PLATFORM = "hardware_platform"


class Predicate(str, Enum):
    CAUSES = "causes"
    AFFECTS = "affects"
    FIXED_IN = "fixed_in"
    REGRESSED_IN = "regressed_in"
    HAS_PARAMETER = "has_parameter"
    DUPLICATE_OF = "duplicate_of"
    ALIAS_OF = "alias_of"


class SourceStatus(str, Enum):
    ACTIVE = "active"
    EDITED = "edited"
    DELETED = "deleted"


class MentionSource(str, Enum):
    LLM_EXTRACTED = "llm_extracted"
    STRUCTURED_RULE = "structured_rule"


class ExtractionMetadata(BaseModel):
    schema_version: str
    model: str
    prompt_version: str
    extracted_at: datetime


class IDGenerator:
    def __init__(self, doc_id: str):
        self._hash = hashlib.sha1(doc_id.encode()).hexdigest()[:8]
        self._counters: dict[str, int] = {"m": 0, "ev": 0, "cm": 0}

    def mention_id(self) -> str:
        idx = self._counters["m"]
        self._counters["m"] += 1
        return f"m_{self._hash}_{idx}"

    def evidence_id(self) -> str:
        idx = self._counters["ev"]
        self._counters["ev"] += 1
        return f"ev_{self._hash}_{idx}"

    def claim_mention_id(self) -> str:
        idx = self._counters["cm"]
        self._counters["cm"] += 1
        return f"cm_{self._hash}_{idx}"


class EvidenceRecord(BaseModel):
    evidence_id: str
    doc_id: str
    source_hash: str
    significance: str | None = None
    excerpt: str
    char_offset_start: int
    char_offset_end: int
    timestamp: datetime
    author: str
    source_status: SourceStatus


class EntityMention(BaseModel):
    mention_id: str
    doc_id: str
    entity_type: EntityType
    surface_form: str
    normalized_form: str
    attributes: dict[str, Any]
    extraction_metadata: ExtractionMetadata
    mention_source: MentionSource = MentionSource.LLM_EXTRACTED
    context_sentence: str | None = None

    @model_validator(mode="after")
    def check_normalized_form(self) -> EntityMention:
        expected = self.surface_form.lower().strip()
        if self.normalized_form != expected:
            raise ValueError(
                f"normalized_form must equal surface_form.lower().strip(): "
                f"expected {expected!r}, got {self.normalized_form!r}"
            )
        return self


class ClaimMention(BaseModel):
    claim_mention_id: str
    doc_id: str
    predicate: Predicate
    subject_mention_id: str
    object_mention_id: str | None = None
    object_freetext: str | None = None
    evidence_ids: list[str]
    confidence: float
    extraction_metadata: ExtractionMetadata

    @model_validator(mode="after")
    def check_claim_mention(self) -> ClaimMention:
        if not self.evidence_ids:
            raise ValueError("evidence_ids must be non-empty")

        if self.predicate in (
            Predicate.CAUSES,
            Predicate.FIXED_IN,
            Predicate.REGRESSED_IN,
            Predicate.HAS_PARAMETER,
            Predicate.DUPLICATE_OF,
            Predicate.ALIAS_OF,
        ):
            has_id = self.object_mention_id is not None
            has_text = self.object_freetext is not None
            if not (has_id ^ has_text):
                raise ValueError(
                    f"predicate {self.predicate.value!r} requires exactly one of "
                    "object_mention_id or object_freetext to be non-None"
                )

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

        return self


class MentionExtractionResult(BaseModel):
    doc_id: str
    entity_mentions: list[EntityMention]
    claim_mentions: list[ClaimMention]
    evidence_records: list[EvidenceRecord]
    extraction_metadata: ExtractionMetadata
