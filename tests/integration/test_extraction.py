from __future__ import annotations

import re

import pytest

from extraction.schema import EntityType, Predicate
from tests.integration.fixtures import (
    bug_report_doc,
    duplicate_issue_doc,
    empty_body_doc,
    fix_confirmation_doc,
    parameter_discussion_doc,
)

pytestmark = pytest.mark.integration


def _retry_extraction(extractor, doc_factory, predicate, max_attempts=3):
    """Retry extraction up to max_attempts times, returning first result that
    contains the expected predicate. Falls back to last result."""
    result = None
    for _ in range(max_attempts):
        result = extractor.extract(doc_factory())
        preds = {cm.predicate for cm in result.claim_mentions}
        if predicate in preds:
            return result
    return result


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _entity_types(result) -> set[EntityType]:
    return {em.entity_type for em in result.entity_mentions}


def _predicates(result) -> set[Predicate]:
    return {cm.predicate for cm in result.claim_mentions}


def _entity_ids(result) -> set[str]:
    return {em.mention_id for em in result.entity_mentions}


def _evidence_ids(result) -> set[str]:
    return {ev.evidence_id for ev in result.evidence_records}


# --------------------------------------------------------------------------- #
# Bug report extraction
# --------------------------------------------------------------------------- #

class TestBugReportExtraction:
    @pytest.fixture(autouse=True)
    def _extract(self, extractor):
        self.result = extractor.extract(bug_report_doc())

    def test_extracts_expected_entity_types(self):
        types = _entity_types(self.result)
        assert EntityType.COMPONENT in types, f"Expected COMPONENT, got {types}"
        assert EntityType.FIRMWARE_VERSION in types, f"Expected FIRMWARE_VERSION, got {types}"

    def test_entity_mention_fields_valid(self):
        for em in self.result.entity_mentions:
            assert re.fullmatch(r"m_[0-9a-f]{8}_\d+", em.mention_id), em.mention_id
            assert em.normalized_form == em.surface_form.lower().strip()

    def test_extracts_evidence(self):
        assert len(self.result.evidence_records) > 0, "Expected at least one evidence record"
        body = bug_report_doc().body
        for ev in self.result.evidence_records:
            if ev.char_offset_start >= 0:
                actual = body[ev.char_offset_start : ev.char_offset_end]
                assert actual == ev.excerpt, (
                    f"Excerpt mismatch at [{ev.char_offset_start}:{ev.char_offset_end}]: "
                    f"{actual!r} != {ev.excerpt!r}"
                )

    def test_extracts_claims(self):
        preds = _predicates(self.result)
        assert preds & {Predicate.AFFECTS, Predicate.CAUSES}, (
            f"Expected AFFECTS or CAUSES, got {preds}"
        )
        ent_ids = _entity_ids(self.result)
        ev_ids = _evidence_ids(self.result)
        for cm in self.result.claim_mentions:
            assert cm.subject_mention_id in ent_ids, (
                f"Claim {cm.claim_mention_id} subject {cm.subject_mention_id} not in entities"
            )
            assert 0.5 <= cm.confidence <= 1.0, (
                f"Confidence {cm.confidence} out of range"
            )
            for eid in cm.evidence_ids:
                assert eid in ev_ids, (
                    f"Claim {cm.claim_mention_id} evidence {eid} not in evidence records"
                )

    def test_entity_ids_consistent_across_passes(self):
        ent_ids = _entity_ids(self.result)
        ev_ids = _evidence_ids(self.result)
        for cm in self.result.claim_mentions:
            assert cm.subject_mention_id in ent_ids
            if cm.object_mention_id:
                assert cm.object_mention_id in ent_ids
            for eid in cm.evidence_ids:
                assert eid in ev_ids


# --------------------------------------------------------------------------- #
# Fix confirmation
# --------------------------------------------------------------------------- #

class TestFixConfirmation:
    def test_extracts_fixed_in(self, extractor):
        result = _retry_extraction(extractor, fix_confirmation_doc, Predicate.FIXED_IN)
        preds = _predicates(result)
        assert Predicate.FIXED_IN in preds, f"Expected FIXED_IN, got {preds}"

        fixed_claims = [c for c in result.claim_mentions if c.predicate == Predicate.FIXED_IN]
        for fc in fixed_claims:
            # Object should reference a firmware version or have freetext with a version
            if fc.object_mention_id:
                obj = next(
                    (e for e in result.entity_mentions if e.mention_id == fc.object_mention_id),
                    None,
                )
                assert obj is not None
                assert obj.entity_type == EntityType.FIRMWARE_VERSION
            else:
                assert fc.object_freetext, "FIXED_IN claim needs object_mention_id or freetext"


# --------------------------------------------------------------------------- #
# Duplicate issue
# --------------------------------------------------------------------------- #

class TestDuplicateIssue:
    def test_extracts_duplicate_of(self, extractor):
        result = extractor.extract(duplicate_issue_doc())
        preds = _predicates(result)
        assert Predicate.DUPLICATE_OF in preds, f"Expected DUPLICATE_OF, got {preds}"


# --------------------------------------------------------------------------- #
# Parameter discussion
# --------------------------------------------------------------------------- #

class TestParameterDiscussion:
    def test_extracts_has_parameter(self, extractor):
        result = extractor.extract(parameter_discussion_doc())
        types = _entity_types(result)
        assert EntityType.PARAMETER in types, f"Expected PARAMETER, got {types}"

        preds = _predicates(result)
        assert Predicate.HAS_PARAMETER in preds, f"Expected HAS_PARAMETER, got {preds}"


# --------------------------------------------------------------------------- #
# Empty body
# --------------------------------------------------------------------------- #

class TestEmptyBody:
    def test_returns_empty(self, extractor):
        # The AUTHOR RULE guarantees the document author is always extracted as a PERSON
        # entity even when the body is empty, so entity_mentions may contain exactly one
        # entry (the author). No claims or evidence should be extracted from an empty body.
        result = extractor.extract(empty_body_doc())
        assert len(result.entity_mentions) <= 1
        if result.entity_mentions:
            assert result.entity_mentions[0].entity_type.value == "person"
        assert len(result.claim_mentions) == 0
        assert len(result.evidence_records) == 0
