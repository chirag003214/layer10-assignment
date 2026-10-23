from __future__ import annotations

from extraction.schema import EntityType, MentionExtractionResult
from extraction.structured_extractor import StructuredExtractor
from ingestion.preprocessor import RawDocument

SCHEMA_VERSION = "2.0.0"
PROMPT_VERSION = "v3"


def _timeline_doc(**kwargs) -> RawDocument:
    defaults = {
        "doc_id": "ardupilot/ardupilot#42_ev1",
        "doc_type": "timeline_event",
        "issue_number": 42,
        "source_url": "https://github.com/ardupilot/ardupilot/issues/42",
        "author": "github-actions",
        "created_at": "2024-01-01T00:00:00Z",
        "event_type": "assigned",
        "event_detail": {},
    }
    defaults.update(kwargs)
    return RawDocument(**defaults)


def _extractor() -> StructuredExtractor:
    return StructuredExtractor(schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_assigned_valid_assignee():
    doc = _timeline_doc(event_type="assigned", event_detail={"assignee": "Ryanf55"})
    result = _extractor().extract(doc)

    assert len(result.entity_mentions) == 1
    em = result.entity_mentions[0]
    assert em.entity_type == EntityType.PERSON
    assert em.attributes["github_handle"] == "Ryanf55"
    assert em.normalized_form == "ryanf55"
    assert result.claim_mentions == []
    assert result.evidence_records == []


def test_assigned_empty_detail():
    doc = _timeline_doc(event_type="assigned", event_detail={})
    result = _extractor().extract(doc)

    assert result.entity_mentions == []
    assert result.claim_mentions == []
    assert result.evidence_records == []


def test_labeled_known_component_gps():
    doc = _timeline_doc(event_type="labeled", event_detail={"label": "GPS"})
    result = _extractor().extract(doc)

    assert len(result.entity_mentions) == 1
    em = result.entity_mentions[0]
    assert em.entity_type == EntityType.COMPONENT
    assert em.surface_form == "GPS"
    assert em.normalized_form == "gps"


def test_labeled_duplicate_returns_empty():
    doc = _timeline_doc(event_type="labeled", event_detail={"label": "Duplicate"})
    result = _extractor().extract(doc)

    assert result.entity_mentions == []
    assert result.claim_mentions == []


def test_labeled_unknown_label_returns_empty():
    doc = _timeline_doc(event_type="labeled", event_detail={"label": "NeedAnswer"})
    result = _extractor().extract(doc)

    assert result.entity_mentions == []
    assert result.claim_mentions == []


def test_milestoned_copter_version():
    doc = _timeline_doc(event_type="milestoned", event_detail={"milestone": "Copter-4.5.8"})
    result = _extractor().extract(doc)

    assert len(result.entity_mentions) == 1
    em = result.entity_mentions[0]
    assert em.entity_type == EntityType.FIRMWARE_VERSION
    assert em.attributes["version_string"] == "4.5.8"
    assert em.attributes["vehicle_type"] == "Copter"
    assert em.surface_form == "Copter-4.5.8"
    assert em.normalized_form == "copter-4.5.8"


def test_milestoned_dds_no_vehicle():
    doc = _timeline_doc(event_type="milestoned", event_detail={"milestone": "DDS 4.6"})
    result = _extractor().extract(doc)

    assert len(result.entity_mentions) == 1
    em = result.entity_mentions[0]
    assert em.entity_type == EntityType.FIRMWARE_VERSION
    assert em.attributes["version_string"] == "4.6"
    assert em.attributes["vehicle_type"] is None


def test_milestoned_empty_detail():
    doc = _timeline_doc(event_type="milestoned", event_detail={})
    result = _extractor().extract(doc)

    assert result.entity_mentions == []


def test_unknown_event_type_locked():
    doc = _timeline_doc(event_type="locked", event_detail={})
    result = _extractor().extract(doc)

    assert result.entity_mentions == []
    assert result.claim_mentions == []
    assert result.evidence_records == []


def test_all_results_round_trip_validation():
    extractor = _extractor()
    cases = [
        _timeline_doc(event_type="assigned", event_detail={"assignee": "tridge"}),
        _timeline_doc(event_type="labeled", event_detail={"label": "Rover"}),
        _timeline_doc(event_type="milestoned", event_detail={"milestone": "Plane-4.4.0"}),
        _timeline_doc(event_type="locked", event_detail={}),
    ]
    for doc in cases:
        result = extractor.extract(doc)
        roundtripped = MentionExtractionResult.model_validate(result.model_dump(mode="json"))
        assert roundtripped.doc_id == result.doc_id
        assert len(roundtripped.entity_mentions) == len(result.entity_mentions)
