from __future__ import annotations

import hashlib
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from extraction.schema import (
    ClaimMention,
    EntityMention,
    EntityType,
    EvidenceRecord,
    ExtractionMetadata,
    IDGenerator,
    MentionExtractionResult,
    Predicate,
    SourceStatus,
)

_NOW = datetime(2024, 1, 1, tzinfo=timezone.utc)

_META = ExtractionMetadata(
    schema_version="2.0.0",
    model="test-model",
    prompt_version="v1",
    extracted_at=_NOW,
)


def _doc_hash(doc_id: str) -> str:
    return hashlib.sha1(doc_id.encode()).hexdigest()[:8]


# --- Test 1 ---

def test_id_generator_format_and_independence():
    doc_id = "issue-123"
    h = _doc_hash(doc_id)
    gen = IDGenerator(doc_id)

    m0 = gen.mention_id()
    m1 = gen.mention_id()
    ev0 = gen.evidence_id()
    cm0 = gen.claim_mention_id()

    assert m0 == f"m_{h}_0"
    assert m1 == f"m_{h}_1"
    assert ev0 == f"ev_{h}_0"
    assert cm0 == f"cm_{h}_0"

    # Counters are independent — further calls increment their own counter only
    assert gen.mention_id() == f"m_{h}_2"
    assert gen.evidence_id() == f"ev_{h}_1"
    assert gen.claim_mention_id() == f"cm_{h}_1"


# --- Test 2 ---

def test_entity_mention_rejects_mismatched_normalized_form():
    with pytest.raises(ValidationError):
        EntityMention(
            mention_id="m_abc_0",
            doc_id="issue-1",
            entity_type=EntityType.COMPONENT,
            surface_form="DShot ESC",
            normalized_form="DShot ESC",  # not lowercased
            attributes={},
            extraction_metadata=_META,
        )


# --- Test 3 ---

def test_claim_mention_rejects_empty_evidence_ids():
    with pytest.raises(ValidationError):
        ClaimMention(
            claim_mention_id="cm_abc_0",
            doc_id="issue-1",
            predicate=Predicate.CAUSES,
            subject_mention_id="m_abc_0",
            object_freetext="motor stops",
            evidence_ids=[],
            confidence=0.9,
            extraction_metadata=_META,
        )


# --- Test 4 ---

def test_claim_mention_rejects_causes_with_no_object():
    with pytest.raises(ValidationError):
        ClaimMention(
            claim_mention_id="cm_abc_0",
            doc_id="issue-1",
            predicate=Predicate.CAUSES,
            subject_mention_id="m_abc_0",
            object_mention_id=None,
            object_freetext=None,
            evidence_ids=["ev_abc_0"],
            confidence=0.9,
            extraction_metadata=_META,
        )


# --- Test 5 ---

def test_mention_extraction_result_round_trip():
    em = EntityMention(
        mention_id="m_aabbccdd_0",
        doc_id="issue-42",
        entity_type=EntityType.COMPONENT,
        surface_form="EKF3",
        normalized_form="ekf3",
        attributes={"note": "test"},
        extraction_metadata=_META,
    )
    er = EvidenceRecord(
        evidence_id="ev_aabbccdd_0",
        doc_id="issue-42",
        source_hash="abc123",
        excerpt="EKF3 caused the crash",
        char_offset_start=0,
        char_offset_end=22,
        timestamp=_NOW,
        author="test-user",
        source_status=SourceStatus.ACTIVE,
    )
    cm = ClaimMention(
        claim_mention_id="cm_aabbccdd_0",
        doc_id="issue-42",
        predicate=Predicate.CAUSES,
        subject_mention_id="m_aabbccdd_0",
        object_freetext="crash",
        evidence_ids=["ev_aabbccdd_0"],
        confidence=0.95,
        extraction_metadata=_META,
    )
    result = MentionExtractionResult(
        doc_id="issue-42",
        entity_mentions=[em],
        claim_mentions=[cm],
        evidence_records=[er],
        extraction_metadata=_META,
    )

    dumped = result.model_dump()
    restored = MentionExtractionResult.model_validate(dumped)

    assert restored.doc_id == "issue-42"
    assert restored.entity_mentions[0].mention_id == "m_aabbccdd_0"
    assert restored.evidence_records[0].evidence_id == "ev_aabbccdd_0"
    assert restored.claim_mentions[0].claim_mention_id == "cm_aabbccdd_0"
    assert restored.claim_mentions[0].evidence_ids == ["ev_aabbccdd_0"]
