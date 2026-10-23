from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from extraction.schema import (
    ClaimMention,
    EntityType,
    EvidenceRecord,
    ExtractionMetadata,
    Predicate,
    SourceStatus,
)
from resolution.claim_resolver import CanonicalClaim, ClaimProvenance, ClaimResolver
from resolution.entity_resolver import CanonicalEntity
from resolution.temporal_resolver import SupersedingResolver, TemporalResolver


def _meta() -> ExtractionMetadata:
    return ExtractionMetadata(
        schema_version="2.0.0",
        model="test-model",
        prompt_version="v1",
        extracted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _claim(
    claim_mention_id: str,
    predicate: str,
    subject_mention_id: str,
    *,
    object_mention_id: str | None = None,
    object_freetext: str | None = None,
    evidence_ids: list[str] | None = None,
    confidence: float = 0.9,
    doc_id: str = "doc1",
) -> ClaimMention:
    return ClaimMention(
        claim_mention_id=claim_mention_id,
        doc_id=doc_id,
        predicate=Predicate(predicate),
        subject_mention_id=subject_mention_id,
        object_mention_id=object_mention_id,
        object_freetext=object_freetext,
        evidence_ids=evidence_ids or ["ev_test_0"],
        confidence=confidence,
        extraction_metadata=_meta(),
    )


def _entity(
    canonical_id: str,
    entity_type: str,
    name: str,
    mention_ids: list[str],
) -> CanonicalEntity:
    return CanonicalEntity(
        canonical_id=canonical_id,
        entity_type=EntityType(entity_type),
        canonical_name=name,
        aliases=[],
        attributes={},
        mention_ids=mention_ids,
        first_seen=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_seen=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _mapping(*entities: CanonicalEntity) -> dict[str, str]:
    """Build mention_to_canonical mapping from entities."""
    m = {}
    for e in entities:
        for mid in e.mention_ids:
            m[mid] = e.canonical_id
    return m


def _evidence(
    evidence_id: str,
    source_hash: str,
    timestamp: datetime,
    doc_id: str = "doc1",
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        doc_id=doc_id,
        source_hash=source_hash,
        excerpt="test excerpt",
        char_offset_start=0,
        char_offset_end=10,
        timestamp=timestamp,
        author="test_author",
        source_status=SourceStatus.ACTIVE,
    )


# Test 1: Subject resolution — valid mention_id → correct canonical_id
def test_subject_resolution():
    mapping = {"m1": "ce_comp_abc", "m2": "ce_fw_def"}
    claims = [
        _claim("cm1", "affects", "m1", object_mention_id="m2"),
    ]
    resolver = ClaimResolver(claims, mapping, [])
    canonical, skipped = resolver.resolve()
    assert len(canonical) == 1
    assert canonical[0].subject_canonical_id == "ce_comp_abc"
    assert canonical[0].object_canonical_id == "ce_fw_def"
    assert skipped == 0


# Test 2: Unknown subject — missing mention → claim skipped, warning logged
def test_unknown_subject_skipped(caplog):
    mapping = {"m1": "ce_comp_abc"}  # m999 not in mapping
    claims = [
        _claim("cm1", "affects", "m999", object_mention_id="m1"),
    ]
    resolver = ClaimResolver(claims, mapping, [])
    with caplog.at_level(logging.WARNING):
        canonical, skipped = resolver.resolve()
    assert len(canonical) == 0
    assert skipped == 1
    assert "m999" in caplog.text


# Test 3: CAUSES grouping — same subject+predicate, different freetext → 2 separate claims
# (new behavior: each distinct symptom is its own canonical claim)
def test_causes_grouping_variants():
    mapping = {"m1": "ce_comp_abc"}
    claims = [
        _claim("cm1", "causes", "m1", object_freetext="motor failure"),
        _claim("cm2", "causes", "m1", object_freetext="motor stops"),
    ]
    resolver = ClaimResolver(claims, mapping, [])
    canonical, skipped = resolver.resolve()
    assert len(canonical) == 2
    freetexts = {c.object_freetext for c in canonical}
    assert freetexts == {"motor failure", "motor stops"}
    for c in canonical:
        assert c.predicate == Predicate.CAUSES
        assert len(c.provenance) == 1


# Test 4: HAS_PARAMETER grouping — same (subject, predicate, object) → merged evidence, max confidence
def test_has_parameter_grouping():
    mapping = {"m1": "ce_comp_abc", "m2": "ce_param_xyz"}
    claims = [
        _claim(
            "cm1", "has_parameter", "m1",
            object_mention_id="m2",
            evidence_ids=["ev1"],
            confidence=0.8,
        ),
        _claim(
            "cm2", "has_parameter", "m1",
            object_mention_id="m2",
            evidence_ids=["ev2"],
            confidence=0.95,
        ),
    ]
    resolver = ClaimResolver(claims, mapping, [])
    canonical, skipped = resolver.resolve()
    assert len(canonical) == 1
    c = canonical[0]
    assert c.confidence == 0.95
    assert set(c.evidence_ids) == {"ev1", "ev2"}
    assert len(c.provenance) == 2


# Test 5: Evidence dedup — no duplicate evidence_ids
def test_evidence_dedup():
    mapping = {"m1": "ce_comp_abc", "m2": "ce_fw_def"}
    claims = [
        _claim(
            "cm1", "affects", "m1",
            object_mention_id="m2",
            evidence_ids=["ev1", "ev2"],
        ),
        _claim(
            "cm2", "affects", "m1",
            object_mention_id="m2",
            evidence_ids=["ev2", "ev3"],
        ),
    ]
    resolver = ClaimResolver(claims, mapping, [])
    canonical, skipped = resolver.resolve()
    assert len(canonical) == 1
    assert canonical[0].evidence_ids == ["ev1", "ev2", "ev3"]


# Test 6: FIXED_IN temporal — matching AFFECTS gets is_current=False + valid_until
def test_fixed_in_temporal():
    entities = [
        _entity("ce_comp_abc", "component", "EKF", ["m1"]),
        _entity("ce_fw_451", "firmware_version", "4.5.1", ["m2"]),
    ]
    entity_dict = {e.canonical_id: e for e in entities}
    entity_dict["ce_fw_451"].attributes = {"version_string": "4.5.1"}

    affects = CanonicalClaim(
        canonical_claim_id="cc_affects_1",
        predicate=Predicate.AFFECTS,
        subject_canonical_id="ce_comp_abc",
        object_canonical_id="ce_fw_451",
        object_freetext=None,
        object_freetext_variants=[],
        evidence_ids=["ev1"],
        provenance=[],
        confidence=0.9,
        is_current=True,
        valid_from=None,
        valid_until=None,
    )
    fixed_in = CanonicalClaim(
        canonical_claim_id="cc_fixed_1",
        predicate=Predicate.FIXED_IN,
        subject_canonical_id="ce_comp_abc",
        object_canonical_id="ce_fw_451",
        object_freetext=None,
        object_freetext_variants=[],
        evidence_ids=["ev2"],
        provenance=[],
        confidence=0.9,
        is_current=True,
        valid_from=None,
        valid_until=None,
    )
    claims = [affects, fixed_in]
    TemporalResolver().resolve(claims, entity_dict)
    assert affects.is_current is False
    assert affects.valid_until == "4.5.1"


# Test 7: Unaffected claims — AFFECTS on different subject not touched by FIXED_IN
def test_unrelated_affects_not_touched():
    entities = [
        _entity("ce_comp_abc", "component", "EKF", ["m1"]),
        _entity("ce_comp_xyz", "component", "GPS", ["m3"]),
        _entity("ce_fw_451", "firmware_version", "4.5.1", ["m2"]),
    ]
    entity_dict = {e.canonical_id: e for e in entities}
    entity_dict["ce_fw_451"].attributes = {"version_string": "4.5.1"}

    affects_ekf = CanonicalClaim(
        canonical_claim_id="cc_affects_1",
        predicate=Predicate.AFFECTS,
        subject_canonical_id="ce_comp_abc",
        object_canonical_id="ce_fw_451",
        object_freetext=None,
        object_freetext_variants=[],
        evidence_ids=["ev1"],
        provenance=[],
        confidence=0.9,
        is_current=True,
        valid_from=None,
        valid_until=None,
    )
    affects_gps = CanonicalClaim(
        canonical_claim_id="cc_affects_2",
        predicate=Predicate.AFFECTS,
        subject_canonical_id="ce_comp_xyz",
        object_canonical_id="ce_fw_451",
        object_freetext=None,
        object_freetext_variants=[],
        evidence_ids=["ev3"],
        provenance=[],
        confidence=0.85,
        is_current=True,
        valid_from=None,
        valid_until=None,
    )
    fixed_in = CanonicalClaim(
        canonical_claim_id="cc_fixed_1",
        predicate=Predicate.FIXED_IN,
        subject_canonical_id="ce_comp_abc",
        object_canonical_id="ce_fw_451",
        object_freetext=None,
        object_freetext_variants=[],
        evidence_ids=["ev2"],
        provenance=[],
        confidence=0.9,
        is_current=True,
        valid_from=None,
        valid_until=None,
    )
    claims = [affects_ekf, affects_gps, fixed_in]
    TemporalResolver().resolve(claims, entity_dict)
    # EKF AFFECTS should be marked historical
    assert affects_ekf.is_current is False
    assert affects_ekf.valid_until == "4.5.1"
    # GPS AFFECTS should remain current (different subject)
    assert affects_gps.is_current is True
    assert affects_gps.valid_until is None


# ---------------------------------------------------------------------------
# New tests — CHANGE 2: ClaimProvenance
# ---------------------------------------------------------------------------

# Test 8: ClaimMention with valid evidence_ids produces ClaimProvenance with correct fields
def test_claim_provenance_fields():
    er = _evidence("ev1", "abc123", datetime(2026, 1, 1, tzinfo=timezone.utc))
    mapping = {"m1": "ce_comp_abc", "m2": "ce_fw_def"}
    claims = [_claim("cm1", "affects", "m1", object_mention_id="m2", evidence_ids=["ev1"])]
    resolver = ClaimResolver(claims, mapping, [er])
    canonical, _ = resolver.resolve()
    assert len(canonical) == 1
    assert len(canonical[0].provenance) == 1
    p = canonical[0].provenance[0]
    assert p.doc_id == "doc1"
    assert p.source_hash == "abc123"
    assert p.predicate == "affects"
    assert p.extraction_schema_version == "2.0.0"
    assert p.extraction_prompt_version == "v1"


# Test 9: Merged canonical claim has one provenance entry per source ClaimMention
# (same freetext → merges; different freetext → separate claims under new grouping)
def test_claim_provenance_one_per_mention():
    er1 = _evidence("ev1", "hash1", datetime(2026, 1, 1, tzinfo=timezone.utc))
    er2 = _evidence("ev2", "hash2", datetime(2026, 1, 2, tzinfo=timezone.utc))
    mapping = {"m1": "ce_comp_abc"}
    # Same freetext → same group → merged into 1 canonical claim with 2 provenance entries
    claims = [
        _claim("cm1", "causes", "m1", object_freetext="crash", evidence_ids=["ev1"]),
        _claim("cm2", "causes", "m1", object_freetext="crash", evidence_ids=["ev2"]),
    ]
    resolver = ClaimResolver(claims, mapping, [er1, er2])
    canonical, _ = resolver.resolve()
    assert len(canonical) == 1
    assert len(canonical[0].provenance) == 2


# Test 10: ClaimProvenance survives model_dump/model_validate round-trip
def test_claim_provenance_round_trip():
    p = ClaimProvenance(
        doc_id="doc1",
        source_hash="abc",
        predicate="affects",
        extraction_schema_version="2.0.0",
        extraction_prompt_version="v1",
        extracted_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    dumped = p.model_dump()
    restored = ClaimProvenance.model_validate(dumped)
    assert restored.doc_id == p.doc_id
    assert restored.source_hash == p.source_hash
    assert restored.predicate == p.predicate
    assert restored.extraction_schema_version == p.extraction_schema_version
    assert restored.extracted_at == p.extracted_at


# ---------------------------------------------------------------------------
# New tests — CHANGE 3: Bitemporal fields
# ---------------------------------------------------------------------------

# Test 11: assertion_time equals the max timestamp across evidence records
def test_assertion_time_max_timestamp():
    t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 2, 1, tzinfo=timezone.utc)
    er1 = _evidence("ev1", "h1", t1)
    er2 = _evidence("ev2", "h2", t2)
    mapping = {"m1": "ce_comp_abc", "m2": "ce_fw_def"}
    claims = [
        _claim("cm1", "affects", "m1", object_mention_id="m2", evidence_ids=["ev1", "ev2"]),
    ]
    resolver = ClaimResolver(claims, mapping, [er1, er2])
    canonical, _ = resolver.resolve()
    assert canonical[0].assertion_time == t2


# Test 12: FIXED_IN claim gets event_time set to the version string of the object entity
def test_event_time_fixed_in():
    entities = [
        _entity("ce_comp_abc", "component", "EKF", ["m1"]),
        _entity("ce_fw_451", "firmware_version", "4.5.1", ["m2"]),
    ]
    entity_dict = {e.canonical_id: e for e in entities}
    entity_dict["ce_fw_451"].attributes = {"version_string": "4.5.1"}

    fixed_in = CanonicalClaim(
        canonical_claim_id="cc_fixed_1",
        predicate=Predicate.FIXED_IN,
        subject_canonical_id="ce_comp_abc",
        object_canonical_id="ce_fw_451",
        object_freetext=None,
    )
    TemporalResolver().resolve([fixed_in], entity_dict)
    assert fixed_in.event_time == "4.5.1"


# Test 13: AFFECTS claim has event_time = None
def test_event_time_affects_is_none():
    entity_dict: dict = {}
    affects = CanonicalClaim(
        canonical_claim_id="cc_affects_1",
        predicate=Predicate.AFFECTS,
        subject_canonical_id="ce_comp_abc",
        object_canonical_id=None,
        object_freetext="test",
    )
    TemporalResolver().resolve([affects], entity_dict)
    assert affects.event_time is None


# Test 14: Claim with no evidence records has assertion_time = None, no exception raised
def test_assertion_time_no_evidence():
    mapping = {"m1": "ce_comp_abc", "m2": "ce_fw_def"}
    claims = [
        _claim("cm1", "affects", "m1", object_mention_id="m2", evidence_ids=["ev_unknown"]),
    ]
    resolver = ClaimResolver(claims, mapping, [])
    canonical, _ = resolver.resolve()
    assert canonical[0].assertion_time is None


# ---------------------------------------------------------------------------
# Tests for _make_group_key
# ---------------------------------------------------------------------------

def _resolver_for_key() -> ClaimResolver:
    """Minimal resolver for testing _make_group_key only."""
    return ClaimResolver([], {}, [])


# Test 15: CAUSES key separates claims by freetext symptom
def test_make_group_key_causes_distinct_symptoms():
    r = _resolver_for_key()
    cm1 = _claim("cm1", "causes", "m1", object_freetext="motor failure")
    cm2 = _claim("cm2", "causes", "m1", object_freetext="esc stall")
    key1 = r._make_group_key(cm1, "ce_comp_abc", None)
    key2 = r._make_group_key(cm2, "ce_comp_abc", None)
    assert key1 != key2
    assert key1 == ("ce_comp_abc", Predicate.CAUSES, "motor failure")
    assert key2 == ("ce_comp_abc", Predicate.CAUSES, "esc stall")


# Test 16: CAUSES key merges claims with same freetext
def test_make_group_key_causes_same_symptom():
    r = _resolver_for_key()
    cm1 = _claim("cm1", "causes", "m1", object_freetext="motor failure")
    cm2 = _claim("cm2", "causes", "m1", object_freetext="motor failure")
    key1 = r._make_group_key(cm1, "ce_comp_abc", None)
    key2 = r._make_group_key(cm2, "ce_comp_abc", None)
    assert key1 == key2


# Test 17: AFFECTS key ignores object identity
def test_make_group_key_affects_ignores_object():
    r = _resolver_for_key()
    cm1 = _claim("cm1", "affects", "m1")
    cm2 = _claim("cm2", "affects", "m1")
    key1 = r._make_group_key(cm1, "ce_comp_abc", "ce_fw_111")
    key2 = r._make_group_key(cm2, "ce_comp_abc", "ce_fw_222")
    assert key1 == key2
    assert key1 == ("ce_comp_abc", Predicate.AFFECTS, None)


# Test 18: FIXED_IN key includes object_canonical_id
def test_make_group_key_fixed_in_includes_object():
    r = _resolver_for_key()
    cm = _claim("cm1", "fixed_in", "m1", object_mention_id="m2")
    key = r._make_group_key(cm, "ce_comp_abc", "ce_fw_451")
    assert key == ("ce_comp_abc", Predicate.FIXED_IN, "ce_fw_451")


# Test 19: REGRESSED_IN key groups by version (object_canonical_id)
def test_make_group_key_regressed_in_by_version():
    r = _resolver_for_key()
    cm1 = _claim("cm1", "regressed_in", "m1", object_freetext="something broke")
    cm2 = _claim("cm2", "regressed_in", "m1", object_freetext="different description")
    # Same version → same key
    key1 = r._make_group_key(cm1, "ce_comp_abc", "ce_fw_430")
    key2 = r._make_group_key(cm2, "ce_comp_abc", "ce_fw_430")
    assert key1 == key2
    # Different versions → different keys
    key3 = r._make_group_key(cm1, "ce_comp_abc", "ce_fw_440")
    assert key1 != key3


# Test 20: HAS_PARAMETER key includes context_hash (use only object_freetext — schema allows one)
def test_make_group_key_has_parameter_context_hash():
    r = _resolver_for_key()
    # HAS_PARAMETER: exactly one of object_mention_id or object_freetext must be set
    cm_a = _claim("cm1", "has_parameter", "m1", object_freetext="value=100")
    cm_b = _claim("cm2", "has_parameter", "m1", object_freetext="value=200")
    key_a = r._make_group_key(cm_a, "ce_comp_abc", None)
    key_b = r._make_group_key(cm_b, "ce_comp_abc", None)
    # Different freetext context → different context_hash → different keys
    assert key_a != key_b
    assert len(key_a) == 4  # (subj, predicate, obj, context_hash)


# Test 21: HAS_PARAMETER key merges identical context
def test_make_group_key_has_parameter_same_context():
    r = _resolver_for_key()
    cm1 = _claim("cm1", "has_parameter", "m1", object_freetext="value=100")
    cm2 = _claim("cm2", "has_parameter", "m1", object_freetext="value=100")
    key1 = r._make_group_key(cm1, "ce_comp_abc", None)
    key2 = r._make_group_key(cm2, "ce_comp_abc", None)
    assert key1 == key2


# Test 22: DUPLICATE_OF key excludes subject
def test_make_group_key_duplicate_of_excludes_subject():
    r = _resolver_for_key()
    cm1 = _claim("cm1", "duplicate_of", "m1", object_freetext="issue #123 is duplicate")
    cm2 = _claim("cm2", "duplicate_of", "m2", object_freetext="issue #123 is duplicate")
    key1 = r._make_group_key(cm1, "ce_comp_abc", None)
    key2 = r._make_group_key(cm2, "ce_comp_xyz", None)  # different subject
    assert key1 == key2
    assert key1 == (Predicate.DUPLICATE_OF, "issue #123 is duplicate")


# Test 23: CAUSES results in separate CanonicalClaims per distinct symptom
def test_causes_separate_canonical_claims_per_symptom():
    mapping = {"m1": "ce_comp_abc"}
    claims = [
        _claim("cm1", "causes", "m1", object_freetext="motor failure"),
        _claim("cm2", "causes", "m1", object_freetext="esc stall"),
        _claim("cm3", "causes", "m1", object_freetext="motor failure"),  # duplicate of cm1
    ]
    resolver = ClaimResolver(claims, mapping, [])
    canonical, skipped = resolver.resolve()
    # Two distinct symptoms → two canonical claims
    assert len(canonical) == 2
    freetexts = {c.object_freetext for c in canonical}
    assert freetexts == {"motor failure", "esc stall"}


# ---------------------------------------------------------------------------
# Tests for SupersedingResolver
# ---------------------------------------------------------------------------


def _regressed_claim(
    claim_id: str,
    subject_id: str,
    assertion_time: datetime | None = None,
) -> CanonicalClaim:
    return CanonicalClaim(
        canonical_claim_id=claim_id,
        predicate=Predicate.REGRESSED_IN,
        subject_canonical_id=subject_id,
        object_canonical_id=None,
        object_freetext="something broke",
        assertion_time=assertion_time,
    )


# Test 24: Single REGRESSED_IN claim — is_current=True, superseded_by=None
def test_superseding_resolver_single_claim():
    claim = _regressed_claim("cc1", "ce_comp_a", datetime(2026, 1, 1, tzinfo=timezone.utc))
    SupersedingResolver().resolve([claim])
    assert claim.is_current is True
    assert claim.superseded_by is None


# Test 25: Two REGRESSED_IN claims — newer is_current=True, older is_current=False + superseded_by
def test_superseding_resolver_two_claims():
    old = _regressed_claim("cc_old", "ce_comp_a", datetime(2025, 6, 1, tzinfo=timezone.utc))
    new = _regressed_claim("cc_new", "ce_comp_a", datetime(2026, 1, 1, tzinfo=timezone.utc))
    SupersedingResolver().resolve([old, new])
    assert new.is_current is True
    assert new.superseded_by is None
    assert old.is_current is False
    assert old.superseded_by == "cc_new"


# Test 26: Different subjects — no cross-subject superseding
def test_superseding_resolver_different_subjects():
    c_a = _regressed_claim("cc_a", "ce_comp_a", datetime(2026, 1, 1, tzinfo=timezone.utc))
    c_b = _regressed_claim("cc_b", "ce_comp_b", datetime(2025, 6, 1, tzinfo=timezone.utc))
    SupersedingResolver().resolve([c_a, c_b])
    # Neither supersedes the other (different subjects)
    assert c_a.superseded_by is None
    assert c_b.superseded_by is None


# Test 27: CAUSES claims are not auto-superseded
def test_superseding_resolver_ignores_causes():
    c1 = CanonicalClaim(
        canonical_claim_id="cc1",
        predicate=Predicate.CAUSES,
        subject_canonical_id="ce_comp_a",
        object_canonical_id=None,
        object_freetext="motor failure",
        assertion_time=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    c2 = CanonicalClaim(
        canonical_claim_id="cc2",
        predicate=Predicate.CAUSES,
        subject_canonical_id="ce_comp_a",
        object_canonical_id=None,
        object_freetext="motor failure",
        assertion_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    SupersedingResolver().resolve([c1, c2])
    # Both remain current — CAUSES not auto-superseded
    assert c1.is_current is True
    assert c2.is_current is True
    assert c1.superseded_by is None
    assert c2.superseded_by is None
