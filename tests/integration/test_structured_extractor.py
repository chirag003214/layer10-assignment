"""Integration tests for StructuredExtractor.

Marked @pytest.mark.no_llm — these are fully deterministic and run without
a real API key. They validate realistic ArduPilot-style fixture inputs,
metadata correctness, and JSON round-trip fidelity.
"""
from __future__ import annotations

import re

import pytest

from extraction.schema import EntityType, MentionExtractionResult
from ingestion.preprocessor import RawDocument

pytestmark = [pytest.mark.integration, pytest.mark.no_llm]

MENTION_ID_RE = re.compile(r"m_[0-9a-f]{8}_\d+")


# ---------------------------------------------------------------------------
# Fixture docs with realistic ArduPilot data
# ---------------------------------------------------------------------------

def _assigned_doc() -> RawDocument:
    return RawDocument(
        doc_id="ardupilot/ardupilot#27431_ev3",
        doc_type="timeline_event",
        issue_number=27431,
        source_url="https://github.com/ArduPilot/ardupilot/issues/27431",
        author="github-actions[bot]",
        created_at="2024-09-12T08:14:22Z",
        event_type="assigned",
        event_detail={"assignee": "Ryanf55"},
    )


def _labeled_rover_doc() -> RawDocument:
    return RawDocument(
        doc_id="ardupilot/ardupilot#27431_ev5",
        doc_type="timeline_event",
        issue_number=27431,
        source_url="https://github.com/ArduPilot/ardupilot/issues/27431",
        author="github-actions[bot]",
        created_at="2024-09-12T08:15:00Z",
        event_type="labeled",
        event_detail={"label": "Rover"},
    )


def _unlabeled_rover_doc() -> RawDocument:
    return RawDocument(
        doc_id="ardupilot/ardupilot#27431_ev6",
        doc_type="timeline_event",
        issue_number=27431,
        source_url="https://github.com/ArduPilot/ardupilot/issues/27431",
        author="github-actions[bot]",
        created_at="2024-09-12T09:00:00Z",
        event_type="unlabeled",
        event_detail={"label": "Rover"},
    )


def _milestoned_doc() -> RawDocument:
    return RawDocument(
        doc_id="ardupilot/ardupilot#27431_ev7",
        doc_type="timeline_event",
        issue_number=27431,
        source_url="https://github.com/ArduPilot/ardupilot/issues/27431",
        author="github-actions[bot]",
        created_at="2024-09-14T10:00:00Z",
        event_type="milestoned",
        event_detail={"milestone": "Copter-4.5.8"},
    )


def _demilestoned_doc() -> RawDocument:
    return RawDocument(
        doc_id="ardupilot/ardupilot#27431_ev8",
        doc_type="timeline_event",
        issue_number=27431,
        source_url="https://github.com/ArduPilot/ardupilot/issues/27431",
        author="github-actions[bot]",
        created_at="2024-09-15T10:00:00Z",
        event_type="demilestoned",
        event_detail={"milestone": "Copter-4.5.7"},
    )


def _referenced_doc() -> RawDocument:
    return RawDocument(
        doc_id="ardupilot/ardupilot#27431_ev9",
        doc_type="timeline_event",
        issue_number=27431,
        source_url="https://github.com/ArduPilot/ardupilot/issues/27431",
        author="tridge",
        created_at="2024-09-16T11:00:00Z",
        event_type="referenced",
        event_detail={},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAssignedEvent:
    @pytest.fixture(autouse=True)
    def _extract(self, structured_extractor):
        self.result = structured_extractor.extract(_assigned_doc())

    def test_produces_one_person_mention(self):
        assert len(self.result.entity_mentions) == 1
        assert self.result.entity_mentions[0].entity_type == EntityType.PERSON

    def test_mention_id_format(self):
        for em in self.result.entity_mentions:
            assert MENTION_ID_RE.fullmatch(em.mention_id), em.mention_id

    def test_normalized_form_invariant(self):
        for em in self.result.entity_mentions:
            assert em.normalized_form == em.surface_form.lower().strip()

    def test_github_handle_attribute(self):
        em = self.result.entity_mentions[0]
        assert em.attributes["github_handle"] == "Ryanf55"

    def test_no_claims_or_evidence(self):
        assert self.result.claim_mentions == []
        assert self.result.evidence_records == []

    def test_metadata_model_is_structured_rules(self):
        assert self.result.extraction_metadata.model == "structured_rules_v1"

    def test_json_round_trip(self):
        rt = MentionExtractionResult.model_validate(self.result.model_dump(mode="json"))
        assert rt.doc_id == self.result.doc_id
        assert len(rt.entity_mentions) == len(self.result.entity_mentions)


class TestLabeledEvent:
    @pytest.fixture(autouse=True)
    def _extract(self, structured_extractor):
        self.result = structured_extractor.extract(_labeled_rover_doc())

    def test_produces_one_component_mention(self):
        assert len(self.result.entity_mentions) == 1
        assert self.result.entity_mentions[0].entity_type == EntityType.COMPONENT

    def test_surface_form_preserved(self):
        assert self.result.entity_mentions[0].surface_form == "Rover"

    def test_normalized_form(self):
        assert self.result.entity_mentions[0].normalized_form == "rover"

    def test_unlabeled_same_result(self, structured_extractor):
        """unlabeled produces the same entity as labeled for the same label."""
        unlabeled_result = structured_extractor.extract(_unlabeled_rover_doc())
        assert len(unlabeled_result.entity_mentions) == 1
        assert unlabeled_result.entity_mentions[0].entity_type == EntityType.COMPONENT


class TestMilestonedEvent:
    @pytest.fixture(autouse=True)
    def _extract(self, structured_extractor):
        self.result = structured_extractor.extract(_milestoned_doc())

    def test_produces_one_firmware_version_mention(self):
        assert len(self.result.entity_mentions) == 1
        assert self.result.entity_mentions[0].entity_type == EntityType.FIRMWARE_VERSION

    def test_version_string_extracted(self):
        em = self.result.entity_mentions[0]
        assert em.attributes["version_string"] == "4.5.8"

    def test_vehicle_type_extracted(self):
        em = self.result.entity_mentions[0]
        assert em.attributes["vehicle_type"] == "Copter"

    def test_surface_form_is_full_milestone(self):
        em = self.result.entity_mentions[0]
        assert em.surface_form == "Copter-4.5.8"


class TestUnhandledEventTypes:
    @pytest.mark.parametrize("doc_factory", [_demilestoned_doc, _referenced_doc])
    def test_returns_empty(self, structured_extractor, doc_factory):
        result = structured_extractor.extract(doc_factory())
        assert result.entity_mentions == []
        assert result.claim_mentions == []
        assert result.evidence_records == []

    @pytest.mark.parametrize("doc_factory", [_demilestoned_doc, _referenced_doc])
    def test_doc_id_preserved(self, structured_extractor, doc_factory):
        doc = doc_factory()
        result = structured_extractor.extract(doc)
        assert result.doc_id == doc.doc_id


class TestBatchRoundTrip:
    def test_all_fixture_docs_round_trip(self, structured_extractor):
        """Every fixture doc round-trips through JSON without validation errors."""
        docs = [
            _assigned_doc(),
            _labeled_rover_doc(),
            _unlabeled_rover_doc(),
            _milestoned_doc(),
            _demilestoned_doc(),
            _referenced_doc(),
        ]
        for doc in docs:
            result = structured_extractor.extract(doc)
            rt = MentionExtractionResult.model_validate(result.model_dump(mode="json"))
            assert rt.doc_id == doc.doc_id

    def test_schema_version_consistent(self, structured_extractor):
        from extraction.extractor import SCHEMA_VERSION
        docs = [_assigned_doc(), _labeled_rover_doc(), _milestoned_doc()]
        for doc in docs:
            result = structured_extractor.extract(doc)
            assert result.extraction_metadata.schema_version == SCHEMA_VERSION
