from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from extraction.schema import Predicate
from resolution.claim_resolver import CanonicalClaim, ClaimProvenance
from resolution.freetext_deduper import (
    ClaimMergeResolver,
    Embedder,
    FreetextClaimDeduper,
    FreetextMergeRecord,
)


def _prov(doc_id: str = "doc1") -> ClaimProvenance:
    return ClaimProvenance(
        doc_id=doc_id,
        source_hash="abc",
        predicate="causes",
        extraction_schema_version="2.0.0",
        extraction_prompt_version="v1",
        extracted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _causes_claim(
    claim_id: str,
    subject_id: str,
    freetext: str,
    evidence_ids: list[str] | None = None,
    confidence: float = 0.9,
) -> CanonicalClaim:
    return CanonicalClaim(
        canonical_claim_id=claim_id,
        predicate=Predicate.CAUSES,
        subject_canonical_id=subject_id,
        object_canonical_id=None,
        object_freetext=freetext,
        object_freetext_variants=[freetext],
        evidence_ids=evidence_ids or [f"ev_{claim_id}"],
        provenance=[_prov()],
        confidence=confidence,
    )


def _regressed_claim(
    claim_id: str,
    subject_id: str,
    freetext: str,
    version_id: str | None = None,
) -> CanonicalClaim:
    return CanonicalClaim(
        canonical_claim_id=claim_id,
        predicate=Predicate.REGRESSED_IN,
        subject_canonical_id=subject_id,
        object_canonical_id=version_id,
        object_freetext=freetext,
        object_freetext_variants=[freetext],
        evidence_ids=[f"ev_{claim_id}"],
        provenance=[_prov()],
        confidence=0.85,
    )


def _affects_claim(claim_id: str, subject_id: str) -> CanonicalClaim:
    return CanonicalClaim(
        canonical_claim_id=claim_id,
        predicate=Predicate.AFFECTS,
        subject_canonical_id=subject_id,
        object_canonical_id="ce_fw_1",
        object_freetext=None,
        evidence_ids=[f"ev_{claim_id}"],
        provenance=[_prov()],
        confidence=0.8,
    )


# ── Test 1: passthrough — non-CAUSES/REGRESSED_IN claims are not deduplicated ──

def test_passthrough_non_candidate_claims():
    claims = [
        _affects_claim("cc1", "ce_comp_a"),
        _affects_claim("cc2", "ce_comp_b"),
    ]
    deduper = FreetextClaimDeduper()
    with patch.object(deduper, "_embed_group", return_value=None):
        result, records = deduper.deduplicate(claims, {})
    assert len(result) == 2
    assert len(records) == 0


# ── Test 2: singleton group — single CAUSES claim passes through unchanged ──

def test_singleton_group_passthrough():
    claims = [_causes_claim("cc1", "ce_comp_a", "motor failure")]
    deduper = FreetextClaimDeduper()
    result, records = deduper.deduplicate(claims, {})
    assert len(result) == 1
    assert result[0].canonical_claim_id == "cc1"
    assert len(records) == 0


# ── Test 3: auto-merge at score >= 0.92 ──

def test_auto_merge_high_similarity():
    claims = [
        _causes_claim("cc1", "ce_comp_a", "motor stops responding"),
        _causes_claim("cc2", "ce_comp_a", "motor stops responding under load"),
    ]

    # Return high similarity embeddings
    embeddings = np.array([[1.0, 0.0], [0.98, 0.2]])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    deduper = FreetextClaimDeduper()
    with patch.object(deduper.embedder, "embed", return_value=embeddings):
        result, records = deduper.deduplicate(claims, {"ce_comp_a": MagicMock(canonical_name="EKF")})

    # Should merge into one claim
    assert len(result) == 1
    assert len(records) == 1
    assert records[0].strategy == "embedding_auto_merge"


# ── Test 4: keep distinct below threshold (< 0.75) ──

def test_keep_distinct_below_threshold():
    claims = [
        _causes_claim("cc1", "ce_comp_a", "motor failure"),
        _causes_claim("cc2", "ce_comp_a", "GPS lock lost"),
    ]

    # Return low similarity embeddings
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])

    deduper = FreetextClaimDeduper()
    with patch.object(deduper.embedder, "embed", return_value=embeddings):
        result, records = deduper.deduplicate(claims, {"ce_comp_a": MagicMock(canonical_name="EKF")})

    assert len(result) == 2
    assert len(records) == 1
    assert records[0].strategy == "embedding_below_threshold"


# ── Test 5: LLM merge decision in borderline band (0.75 <= score < 0.92) ──

def test_llm_merge_decision_borderline():
    claims = [
        _causes_claim("cc1", "ce_comp_a", "motor shutoff"),
        _causes_claim("cc2", "ce_comp_a", "motor power cut"),
    ]

    sim = 0.83
    cos = np.cos(np.arccos(sim))
    e1 = np.array([1.0, 0.0])
    e2 = np.array([cos, np.sqrt(1 - cos**2)])
    embeddings = np.array([e1, e2])

    mock_decision = MagicMock()
    mock_decision.decision = "merge"
    mock_decision.confidence = 0.85
    mock_decision.reasoning = "near-synonym"
    mock_decision.strategy = "llm_claim_merge"

    deduper = FreetextClaimDeduper()
    with patch.object(deduper.embedder, "embed", return_value=embeddings), \
         patch.object(deduper.merge_resolver, "should_merge", return_value=mock_decision):
        result, records = deduper.deduplicate(claims, {"ce_comp_a": MagicMock(canonical_name="EKF")})

    assert len(result) == 1
    assert records[0].strategy == "llm_claim_merge"
    assert records[0].llm_decision == "merge"


# ── Test 6: LLM keep_distinct decision in borderline band ──

def test_llm_keep_distinct_borderline():
    claims = [
        _causes_claim("cc1", "ce_comp_a", "motor shutoff"),
        _causes_claim("cc2", "ce_comp_a", "altitude hold failure"),
    ]

    sim = 0.80
    e1 = np.array([1.0, 0.0])
    e2 = np.array([sim, np.sqrt(1 - sim**2)])
    embeddings = np.array([e1, e2])

    mock_decision = MagicMock()
    mock_decision.decision = "keep_distinct"
    mock_decision.confidence = 0.9
    mock_decision.reasoning = "different subsystems"
    mock_decision.strategy = "llm_claim_distinct"

    deduper = FreetextClaimDeduper()
    with patch.object(deduper.embedder, "embed", return_value=embeddings), \
         patch.object(deduper.merge_resolver, "should_merge", return_value=mock_decision):
        result, records = deduper.deduplicate(claims, {"ce_comp_a": MagicMock(canonical_name="EKF")})

    assert len(result) == 2
    assert records[0].strategy == "llm_claim_distinct"


# ── Test 7: merged claim keeps most evidence, unions evidence_ids ──

def test_merge_keeps_most_evidence_union():
    claims = [
        _causes_claim("cc1", "ce_comp_a", "motor failure", evidence_ids=["ev1", "ev2", "ev3"]),
        _causes_claim("cc2", "ce_comp_a", "motor stops", evidence_ids=["ev4"]),
    ]

    embeddings = np.array([[1.0, 0.0], [0.99, 0.141]])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    deduper = FreetextClaimDeduper()
    with patch.object(deduper.embedder, "embed", return_value=embeddings):
        result, records = deduper.deduplicate(claims, {"ce_comp_a": MagicMock(canonical_name="EKF")})

    assert len(result) == 1
    merged = result[0]
    # Base claim is cc1 (most evidence)
    assert set(merged.evidence_ids) == {"ev1", "ev2", "ev3", "ev4"}
    # Freetext variants unioned
    assert set(merged.object_freetext_variants) == {"motor failure", "motor stops"}


# ── Test 8: warning logged when CAUSES claim has >1 freetext_variants (old grouping) ──

def test_warning_on_multi_variant_claim(caplog):
    claim = CanonicalClaim(
        canonical_claim_id="cc1",
        predicate=Predicate.CAUSES,
        subject_canonical_id="ce_comp_a",
        object_canonical_id=None,
        object_freetext="motor failure",
        object_freetext_variants=["motor failure", "motor stops"],  # >1 variant
        evidence_ids=["ev1"],
        provenance=[_prov()],
        confidence=0.9,
    )
    deduper = FreetextClaimDeduper()
    with caplog.at_level(logging.WARNING):
        result, records = deduper.deduplicate([claim], {})
    assert "object_freetext_variants" in caplog.text or "old grouping" in caplog.text.lower() or len(caplog.records) >= 1


# ── Test 9: REGRESSED_IN claims are also deduplicated ──

def test_regressed_in_deduplication():
    claims = [
        _regressed_claim("cc1", "ce_comp_a", "battery failsafe triggered"),
        _regressed_claim("cc2", "ce_comp_a", "battery safety triggered"),
    ]

    embeddings = np.array([[1.0, 0.0], [0.97, 0.243]])
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    deduper = FreetextClaimDeduper()
    with patch.object(deduper.embedder, "embed", return_value=embeddings):
        result, records = deduper.deduplicate(claims, {"ce_comp_a": MagicMock(canonical_name="EKF")})

    assert len(result) == 1
    assert records[0].strategy == "embedding_auto_merge"


# ── Test 10: FreetextMergeRecord has all required fields ──

def test_freetext_merge_record_fields():
    record = FreetextMergeRecord(
        canonical_claim_id="cc_merged",
        merged_claim_ids=["cc1", "cc2"],
        subject_canonical_id="ce_comp_a",
        subject_name="EKF",
        predicate="causes",
        similarity_score=0.95,
        strategy="embedding_auto_merge",
        llm_decision=None,
        llm_confidence=None,
        llm_reasoning=None,
        decided_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert record.canonical_claim_id == "cc_merged"
    assert record.strategy == "embedding_auto_merge"
    assert record.llm_decision is None
    assert record.similarity_score == 0.95
