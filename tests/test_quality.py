from __future__ import annotations

from datetime import datetime, timezone

from extraction.schema import (
    ClaimMention,
    EvidenceRecord,
    ExtractionMetadata,
    Predicate,
    SourceStatus,
)
from extraction.quality import get_stale_claims, get_ungrounded_claims


# ---------------------------------------------------------------------------
# Helpers (mirrors test_extraction.py patterns)
# ---------------------------------------------------------------------------

def _meta(schema_version: str = "2.0.0") -> ExtractionMetadata:
    return ExtractionMetadata(
        schema_version=schema_version,
        model="test-model",
        prompt_version="v3",
        extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _claim_mention(
    claim_mention_id: str = "cm_test_0",
    predicate: Predicate = Predicate.AFFECTS,
    subject_mention_id: str = "m_test_0",
    evidence_ids: list[str] | None = None,
    confidence: float = 0.9,
    doc_id: str = "ardupilot/ardupilot#1",
    schema_version: str = "2.0.0",
) -> ClaimMention:
    return ClaimMention(
        claim_mention_id=claim_mention_id,
        doc_id=doc_id,
        predicate=predicate,
        subject_mention_id=subject_mention_id,
        evidence_ids=evidence_ids or ["ev_test_0"],
        confidence=confidence,
        extraction_metadata=_meta(schema_version),
    )


def _evidence_record(
    evidence_id: str = "ev_test_0",
    doc_id: str = "ardupilot/ardupilot#1",
    source_status: SourceStatus = SourceStatus.ACTIVE,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        doc_id=doc_id,
        source_hash="abc123",
        significance="test significance",
        excerpt="EKF lost GPS lock",
        char_offset_start=0,
        char_offset_end=18,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        author="tridge",
        source_status=source_status,
    )


# ---------------------------------------------------------------------------
# get_stale_claims
# ---------------------------------------------------------------------------

def test_get_stale_claims_empty_when_current(tmp_path):
    claims_path = tmp_path / "claim_mentions.jsonl"
    cm = _claim_mention(schema_version="2.0.0")
    claims_path.write_text(cm.model_dump_json() + "\n")

    result = get_stale_claims(claims_path, current_schema_version="2.0.0")
    assert result == []


def test_get_stale_claims_returns_doc_id_for_old_version(tmp_path):
    claims_path = tmp_path / "claim_mentions.jsonl"
    cm = _claim_mention(schema_version="1.0.0", doc_id="ardupilot/ardupilot#42")
    claims_path.write_text(cm.model_dump_json() + "\n")

    result = get_stale_claims(claims_path, current_schema_version="2.0.0")
    assert result == ["ardupilot/ardupilot#42"]


# ---------------------------------------------------------------------------
# get_ungrounded_claims
# ---------------------------------------------------------------------------

def test_get_ungrounded_claims_empty_when_all_active(tmp_path):
    claims_path = tmp_path / "claim_mentions.jsonl"
    evidence_path = tmp_path / "evidence_records.jsonl"

    ev = _evidence_record(evidence_id="ev_test_0", source_status=SourceStatus.ACTIVE)
    cm = _claim_mention(evidence_ids=["ev_test_0"])

    evidence_path.write_text(ev.model_dump_json() + "\n")
    claims_path.write_text(cm.model_dump_json() + "\n")

    result = get_ungrounded_claims(claims_path, evidence_path)
    assert result == []


def test_get_ungrounded_claims_returns_dict_for_edited(tmp_path):
    claims_path = tmp_path / "claim_mentions.jsonl"
    evidence_path = tmp_path / "evidence_records.jsonl"

    ev = _evidence_record(evidence_id="ev_test_0", source_status=SourceStatus.EDITED)
    cm = _claim_mention(evidence_ids=["ev_test_0"])

    evidence_path.write_text(ev.model_dump_json() + "\n")
    claims_path.write_text(cm.model_dump_json() + "\n")

    result = get_ungrounded_claims(claims_path, evidence_path)
    assert len(result) == 1
    assert result[0]["doc_id"] == "ardupilot/ardupilot#1"
    assert result[0]["predicate"] == "affects"
    assert result[0]["evidence_excerpt"] == "EKF lost GPS lock"
