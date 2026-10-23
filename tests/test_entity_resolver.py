from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from extraction.schema import EntityMention, EntityType, ExtractionMetadata, MentionSource
from resolution.entity_resolver import (
    CanonicalEntity,
    ComponentResolver,
    EntityResolver,
    FirmwareVersionResolver,
    HardwarePlatformResolver,
    ParameterResolver,
    PersonResolver,
    SemanticEntityResolver,
    resolve_aliases,
)


def _mock_llm_response(decision: str, confidence: float, reasoning: str) -> MagicMock:
    tool_call = MagicMock()
    tool_call.function.arguments = json.dumps(
        {"decision": decision, "confidence": confidence, "reasoning": reasoning}
    )
    message = MagicMock()
    message.tool_calls = [tool_call]
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def _meta(ts: str = "2026-01-01T00:00:00Z") -> ExtractionMetadata:
    return ExtractionMetadata(
        schema_version="2.0.0",
        model="test-model",
        prompt_version="v1",
        extracted_at=datetime.fromisoformat(ts.replace("Z", "+00:00")),
    )


def _mention(
    mention_id: str,
    entity_type: str,
    surface_form: str,
    attributes: dict | None = None,
    doc_id: str = "doc1",
    ts: str = "2026-01-01T00:00:00Z",
) -> EntityMention:
    return EntityMention(
        mention_id=mention_id,
        doc_id=doc_id,
        entity_type=EntityType(entity_type),
        surface_form=surface_form,
        normalized_form=surface_form.lower().strip(),
        attributes=attributes or {},
        extraction_metadata=_meta(ts),
    )


# ---------------------------------------------------------------------------
# TestPersonResolver
# ---------------------------------------------------------------------------

class TestPersonResolver:
    def test_same_github_handle_merged(self):
        m1 = _mention("m1", "person", "Alice", {"github_handle": "alice123"})
        m2 = _mention("m2", "person", "alice123", {"github_handle": "alice123"})
        resolver = PersonResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert len(records) == 1
        assert set(entities[0].mention_ids) == {"m1", "m2"}

    def test_different_handles_distinct(self):
        m1 = _mention("m1", "person", "Alice", {"github_handle": "alice"})
        m2 = _mention("m2", "person", "Bob", {"github_handle": "bob"})
        resolver = PersonResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2
        assert len(records) == 0

    def test_no_handle_falls_back_to_normalized_form(self):
        m1 = _mention("m1", "person", "Alice Smith", {})
        m2 = _mention("m2", "person", "alice smith", {})
        resolver = PersonResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert len(records) == 1


# ---------------------------------------------------------------------------
# TestParameterResolver
# ---------------------------------------------------------------------------

class TestParameterResolver:
    def test_exact_same_case_merged(self):
        m1 = _mention("m1", "parameter", "EK3_ENABLE", {"value": "1"})
        m2 = _mention("m2", "parameter", "EK3_ENABLE", {"value": "1"})
        resolver = ParameterResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert len(records) == 1
        assert records[0].strategy == "exact_key_match"

    def test_space_vs_underscore_merged(self):
        m1 = _mention("m1", "parameter", "AHRS EKF TYPE", {"value": "3"})
        m2 = _mention("m2", "parameter", "AHRS_EKF_TYPE", {"value": "3"})
        resolver = ParameterResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert len(records) == 1
        assert records[0].strategy == "exact_key_match"

    def test_case_insensitive_merged(self):
        m1 = _mention("m1", "parameter", "EK3_ENABLE", {"value": "1"})
        m2 = _mention("m2", "parameter", "ek3_enable", {"value": "0"})
        resolver = ParameterResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert len(records) == 1
        assert records[0].strategy == "exact_key_match"

    def test_different_names_distinct(self):
        m1 = _mention("m1", "parameter", "EK3_ENABLE", {"value": "1"})
        m2 = _mention("m2", "parameter", "ATC_RATE_P", {"value": "0.1"})
        resolver = ParameterResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2
        assert len(records) == 0


# ---------------------------------------------------------------------------
# TestFirmwareVersionResolver
# ---------------------------------------------------------------------------

class TestFirmwareVersionResolver:
    def test_exact_same_version_and_vehicle_merged(self):
        m1 = _mention("m1", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": "Copter"})
        m2 = _mention("m2", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": "Copter"})
        resolver = FirmwareVersionResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "exact_key_match"

    def test_same_version_one_vehicle_none_merged(self):
        m1 = _mention("m1", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": None})
        m2 = _mention("m2", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": "Copter"})
        resolver = FirmwareVersionResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "firmware_null_vehicle_merge"

    def test_same_version_both_none_merged(self):
        m1 = _mention("m1", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": None})
        m2 = _mention("m2", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": None})
        resolver = FirmwareVersionResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "exact_key_match"

    def test_same_version_different_vehicle_types_distinct(self):
        m1 = _mention("m1", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": "Copter"})
        m2 = _mention("m2", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": "Plane"})
        resolver = FirmwareVersionResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2
        assert len(records) == 0

    def test_different_version_strings_distinct(self):
        m1 = _mention("m1", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": "Copter"})
        m2 = _mention("m2", "firmware_version", "4.5.2",
                      {"version_string": "4.5.2", "vehicle_type": "Copter"})
        resolver = FirmwareVersionResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2
        assert len(records) == 0

    def test_prefix_strip_merged(self):
        m1 = _mention("m1", "firmware_version", "Copter-4.5.1",
                      {"version_string": "Copter-4.5.1", "vehicle_type": "Copter"})
        m2 = _mention("m2", "firmware_version", "4.5.1",
                      {"version_string": "4.5.1", "vehicle_type": "Copter"})
        resolver = FirmwareVersionResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "exact_key_match"


# ---------------------------------------------------------------------------
# TestSemanticEntityResolver — Phase 1: Structural merging
# ---------------------------------------------------------------------------

class TestSemanticEntityResolver:
    """Tests for the 5-step structural decision tree."""

    # Test 1: Exact normalized match merges
    def test_exact_normalized_match(self):
        m1 = _mention("m1", "component", "EKF", {})
        m2 = _mention("m2", "component", "ekf", {})
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "exact_key_match"

    # Test 2: Structural key match via suffix strip
    def test_structural_key_via_suffix_strip(self):
        m1 = _mention("m1", "component", "GPS Module", {})
        m2 = _mention("m2", "component", "GPS", {})
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "structural_key_match"

    # Test 3: Structural key match via namespace prefix
    def test_structural_key_via_namespace_prefix(self):
        m1 = _mention("m1", "component", "AP_GPS", {})
        m2 = _mention("m2", "component", "GPS", {})
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "structural_key_match"

    # Test 4: Numeric variant keeps distinct
    def test_numeric_variant_distinct(self):
        m1 = _mention("m1", "component", "EKF", {})
        m2 = _mention("m2", "component", "EKF3", {})
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2
        assert len(records) == 0

    # Test 5: Levenshtein-1 same document → llm_candidate
    def test_levenshtein_1_same_doc_llm_candidate(self):
        m1 = _mention("m1", "component", "Compas", {}, doc_id="doc1")
        m2 = _mention("m2", "component", "Compass", {}, doc_id="doc1")
        resolver = ComponentResolver()
        # Without LLM resolver, stays distinct (no merge happens)
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2
        assert len(records) == 0
        # With an LLM resolver that merges them
        mock_resp = _mock_llm_response("merge", 0.9, "Typo of compass.")
        from resolution.merge_resolver import MergeResolver
        with patch("resolution.merge_resolver.litellm.completion", return_value=mock_resp):
            entities, records, decisions = resolver.resolve_group(
                [m1, m2], llm_resolver=MergeResolver()
            )
        assert len(entities) == 1
        assert records[0].strategy == "llm_merge_decision"

    # Test 6: Levenshtein-1 different documents → distinct
    def test_levenshtein_1_different_docs_distinct(self):
        m1 = _mention("m1", "component", "Compas", {}, doc_id="doc1")
        m2 = _mention("m2", "component", "Compass", {}, doc_id="doc2")
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2
        assert len(records) == 0

    # Test 7: SITL and SITL Rover → distinct
    def test_sitl_and_sitl_rover_distinct(self):
        m1 = _mention("m1", "component", "SITL", {})
        m2 = _mention("m2", "component", "SITL Rover", {})
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2

    # Test 8: Rover and SITL Rover → distinct
    def test_rover_and_sitl_rover_distinct(self):
        m1 = _mention("m1", "component", "Rover", {})
        m2 = _mention("m2", "component", "SITL Rover", {})
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2

    # Test 9: MAVLink and Camera → distinct
    def test_mavlink_and_camera_distinct(self):
        m1 = _mention("m1", "component", "MAVLink", {})
        m2 = _mention("m2", "component", "Camera", {})
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2

    # Test 10: AP_GPS and AP_Baro → distinct
    def test_ap_gps_and_ap_baro_distinct(self):
        m1 = _mention("m1", "component", "AP_GPS", {})
        m2 = _mention("m2", "component", "AP_Baro", {})
        resolver = ComponentResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 2


class TestEditDistance:
    def test_identical_strings(self):
        assert SemanticEntityResolver._edit_distance("hello", "hello") == 0

    def test_single_char_difference(self):
        assert SemanticEntityResolver._edit_distance("compas", "compass") == 1

    def test_completely_different(self):
        assert SemanticEntityResolver._edit_distance("abc", "xyz") == 3

    def test_empty_strings(self):
        assert SemanticEntityResolver._edit_distance("", "") == 0
        assert SemanticEntityResolver._edit_distance("abc", "") == 3
        assert SemanticEntityResolver._edit_distance("", "abc") == 3


# ---------------------------------------------------------------------------
# TestHardwarePlatformResolver
# ---------------------------------------------------------------------------

class TestHardwarePlatformResolver:
    def test_exact_normalized_merged(self):
        m1 = _mention("m1", "hardware_platform", "Pixhawk", {})
        m2 = _mention("m2", "hardware_platform", "pixhawk", {})
        resolver = HardwarePlatformResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "exact_key_match"

    def test_structural_key_merge(self):
        """'pixhawk module' strips suffix → 'pixhawk', matches 'pixhawk'."""
        m1 = _mention("m1", "hardware_platform", "Pixhawk Module", {})
        m2 = _mention("m2", "hardware_platform", "Pixhawk", {})
        resolver = HardwarePlatformResolver()
        entities, records, decisions = resolver.resolve_group([m1, m2])
        assert len(entities) == 1
        assert records[0].strategy == "structural_key_match"


# ---------------------------------------------------------------------------
# TestEntityResolverOrchestrator
# ---------------------------------------------------------------------------

class TestEntityResolverOrchestrator:
    def test_mixed_types_resolve_independently(self):
        mentions = [
            _mention("m1", "component", "DShot", {}),
            _mention("m2", "person", "Alice", {"github_handle": "alice"}),
            _mention("m3", "firmware_version", "4.5.1",
                     {"version_string": "4.5.1", "vehicle_type": "Copter"}),
            _mention("m4", "hardware_platform", "Pixhawk 6C", {}),
        ]
        resolver = EntityResolver()
        entities, _, _ = resolver.resolve(mentions)
        pattern = re.compile(r"^ce_[a-z_]+_[0-9a-f]{8}$")
        for entity in entities:
            assert pattern.match(entity.canonical_id), (
                f"canonical_id {entity.canonical_id!r} does not match expected pattern"
            )

    def test_all_mentions_mapped(self):
        mentions = [
            _mention("m1", "person", "Alice", {"github_handle": "alice"}),
            _mention("m2", "person", "Bob", {"github_handle": "bob"}),
            _mention("m3", "parameter", "EK3_ENABLE", {"value": "1"}),
            _mention("m4", "parameter", "ATC_RATE_P", {"value": "0.1"}),
            _mention("m5", "component", "DShot", {}),
            _mention("m6", "component", "DShot ESC", {}),
            _mention("m7", "firmware_version", "Copter-4.5.1",
                     {"version_string": "Copter-4.5.1", "vehicle_type": "Copter"}),
        ]
        resolver = EntityResolver()
        entities, records, candidates = resolver.resolve(mentions)

        mention_to_canonical: dict[str, str] = {}
        for entity in entities:
            for mid in entity.mention_ids:
                mention_to_canonical[mid] = entity.canonical_id

        all_ids = {m.mention_id for m in mentions}
        assert all_ids == set(mention_to_canonical.keys())
        total_mapped = sum(len(e.mention_ids) for e in entities)
        assert total_mapped == len(mentions)

    def test_merge_records_have_correct_strategies(self):
        mentions = [
            _mention("m1", "component", "GPS Module", {}),
            _mention("m2", "component", "GPS", {}),
            _mention("m3", "parameter", "EK3_ENABLE", {"value": "1"}),
            _mention("m4", "parameter", "ek3_enable", {"value": "0"}),
        ]
        resolver = EntityResolver()
        entities, records, _ = resolver.resolve(mentions)
        strategies = {r.strategy for r in records}
        assert "structural_key_match" in strategies
        assert "exact_key_match" in strategies


# ---------------------------------------------------------------------------
# TestMentionSource — CHANGE 1
# ---------------------------------------------------------------------------

class TestMentionSource:
    def test_default_mention_source_is_llm_extracted(self):
        m = _mention("m1", "component", "EKF", {})
        assert m.mention_source == MentionSource.LLM_EXTRACTED

    def test_structured_extractor_sets_structured_rule(self):
        from extraction.structured_extractor import StructuredExtractor
        from ingestion.preprocessor import RawDocument

        extractor = StructuredExtractor("1.0.0", "v1")

        assigned_doc = RawDocument(
            doc_id="test_doc_1",
            doc_type="timeline_event",
            issue_number=1,
            source_url="https://github.com/ArduPilot/ardupilot/issues/1",
            author="bot",
            created_at="2024-01-01T00:00:00Z",
            event_type="assigned",
            event_detail={"assignee": "alice"},
        )
        result = extractor.extract(assigned_doc)
        assert len(result.entity_mentions) == 1
        assert result.entity_mentions[0].mention_source == MentionSource.STRUCTURED_RULE

        labeled_doc = RawDocument(
            doc_id="test_doc_2",
            doc_type="timeline_event",
            issue_number=1,
            source_url="https://github.com/ArduPilot/ardupilot/issues/1",
            author="bot",
            created_at="2024-01-01T00:00:00Z",
            event_type="labeled",
            event_detail={"label": "gps"},
        )
        result = extractor.extract(labeled_doc)
        assert len(result.entity_mentions) == 1
        assert result.entity_mentions[0].mention_source == MentionSource.STRUCTURED_RULE

        milestoned_doc = RawDocument(
            doc_id="test_doc_3",
            doc_type="timeline_event",
            issue_number=1,
            source_url="https://github.com/ArduPilot/ardupilot/issues/1",
            author="bot",
            created_at="2024-01-01T00:00:00Z",
            event_type="milestoned",
            event_detail={"milestone": "Copter-4.5.1"},
        )
        result = extractor.extract(milestoned_doc)
        assert len(result.entity_mentions) == 1
        assert result.entity_mentions[0].mention_source == MentionSource.STRUCTURED_RULE

    def test_mention_source_round_trip(self):
        m = _mention("m1", "component", "EKF", {})
        dumped = m.model_dump()
        restored = EntityMention.model_validate(dumped)
        assert restored.mention_source == MentionSource.LLM_EXTRACTED

        m2 = EntityMention(
            mention_id="m2",
            doc_id="doc1",
            entity_type=EntityType.COMPONENT,
            surface_form="GPS",
            normalized_form="gps",
            attributes={},
            extraction_metadata=_meta(),
            mention_source=MentionSource.STRUCTURED_RULE,
        )
        dumped2 = m2.model_dump()
        restored2 = EntityMention.model_validate(dumped2)
        assert restored2.mention_source == MentionSource.STRUCTURED_RULE


# ---------------------------------------------------------------------------
# Phase 2 tests: Alias resolution
# ---------------------------------------------------------------------------

def _make_canonical_entity(
    canonical_id: str,
    name: str,
    entity_type: str = "component",
    aliases: list[str] | None = None,
    mention_ids: list[str] | None = None,
) -> CanonicalEntity:
    return CanonicalEntity(
        canonical_id=canonical_id,
        entity_type=EntityType(entity_type),
        canonical_name=name,
        aliases=aliases or [],
        attributes={},
        mention_ids=mention_ids or [f"m_{canonical_id}"],
        first_seen=datetime(2026, 1, 1, tzinfo=timezone.utc),
        last_seen=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


def _make_alias_claim(
    claim_id: str,
    subject_mention_id: str,
    alias_text: str,
    doc_id: str = "doc1",
) -> "ClaimMention":
    from extraction.schema import ClaimMention, Predicate
    return ClaimMention(
        claim_mention_id=claim_id,
        doc_id=doc_id,
        predicate=Predicate.ALIAS_OF,
        subject_mention_id=subject_mention_id,
        object_freetext=alias_text,
        evidence_ids=["ev_test_0"],
        confidence=0.9,
        extraction_metadata=_meta(),
    )


class TestAliasResolution:
    # Test 11: ALIAS_OF claim triggers merge of two canonical entities
    def test_alias_claim_merges_entities(self):
        entity_a = _make_canonical_entity("ce_a", "EKF", mention_ids=["m1"])
        entity_b = _make_canonical_entity("ce_b", "Extended Kalman Filter", mention_ids=["m2"])
        mention_to_canonical = {"m1": "ce_a", "m2": "ce_b"}

        alias_claim = _make_alias_claim("cm1", "m1", "Extended Kalman Filter")

        entities, mtc, records = resolve_aliases(
            [entity_a, entity_b], mention_to_canonical, [alias_claim]
        )

        assert len(entities) == 1
        assert entities[0].canonical_name == "EKF"
        assert "Extended Kalman Filter" in entities[0].aliases
        assert mtc["m2"] == "ce_a"
        assert len(records) == 1
        assert records[0].strategy == "corpus_alias_evidence"

    # Test 12: ALIAS_OF with no matching entity for object_freetext → no merge
    def test_alias_no_matching_entity_no_error(self):
        entity_a = _make_canonical_entity("ce_a", "EKF", mention_ids=["m1"])
        mention_to_canonical = {"m1": "ce_a"}

        alias_claim = _make_alias_claim("cm1", "m1", "nonexistent entity")

        entities, mtc, records = resolve_aliases(
            [entity_a], mention_to_canonical, [alias_claim]
        )

        assert len(entities) == 1
        assert len(records) == 0


# ---------------------------------------------------------------------------
# Phase 3 tests: LLM context
# ---------------------------------------------------------------------------

class TestLLMContext:
    # Test 14: LLM candidate receives context_sentence
    def test_llm_candidate_receives_context(self):
        from resolution.merge_resolver import MergeResolver, MergeCandidate

        m1 = EntityMention(
            mention_id="m1",
            doc_id="doc1",
            entity_type=EntityType.COMPONENT,
            surface_form="Compas",
            normalized_form="compas",
            attributes={},
            extraction_metadata=_meta(),
            context_sentence="The Compas driver reported an error.",
        )
        m2 = EntityMention(
            mention_id="m2",
            doc_id="doc1",
            entity_type=EntityType.COMPONENT,
            surface_form="Compass",
            normalized_form="compass",
            attributes={},
            extraction_metadata=_meta(),
            context_sentence="Check the Compass calibration settings.",
        )

        candidate = MergeCandidate(mention_a=m1, mention_b=m2, similarity_score=0.85)

        mock_resp = _mock_llm_response("merge", 0.9, "Typo for compass.")
        with patch("resolution.merge_resolver.litellm.completion", return_value=mock_resp) as mock_call:
            resolver = MergeResolver()
            decision = resolver._resolve_one(candidate)

        # Verify context was passed in prompt
        call_args = mock_call.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        user_msg = messages[1]["content"]
        assert "The Compas driver reported an error." in user_msg
        assert "Check the Compass calibration settings." in user_msg
        assert decision.decision == "merge"

    # Test 15: LLM merge decision records reasoning
    def test_llm_merge_records_reasoning(self):
        from resolution.merge_resolver import MergeResolver, MergeCandidate

        m1 = _mention("m1", "component", "Compas", {}, doc_id="doc1")
        m2 = _mention("m2", "component", "Compass", {}, doc_id="doc1")

        candidate = MergeCandidate(mention_a=m1, mention_b=m2, similarity_score=0.85)
        mock_resp = _mock_llm_response("merge", 0.95, "Same component, typo.")
        with patch("resolution.merge_resolver.litellm.completion", return_value=mock_resp):
            resolver = MergeResolver()
            decision = resolver._resolve_one(candidate)

        assert decision.decision == "merge"
        assert decision.reasoning == "Same component, typo."
        assert decision.confidence == 0.95

    # Test 16: LLM failure returns distinct, no exception
    def test_llm_failure_returns_distinct(self):
        from resolution.merge_resolver import MergeResolver, MergeCandidate

        m1 = _mention("m1", "component", "Compas", {}, doc_id="doc1")
        m2 = _mention("m2", "component", "Compass", {}, doc_id="doc1")

        candidate = MergeCandidate(mention_a=m1, mention_b=m2, similarity_score=0.85)
        with patch("resolution.merge_resolver.litellm.completion", side_effect=Exception("LLM down")):
            resolver = MergeResolver()
            decision = resolver._resolve_one(candidate)

        assert decision.decision == "keep_distinct"
        assert decision.confidence == 0.0


# ---------------------------------------------------------------------------
# Phase 4 tests: Coherence splitter
# ---------------------------------------------------------------------------

class TestCoherenceSplitter:
    def _make_claim(
        self, claim_id: str, subject_id: str, predicate: str, freetext: str
    ) -> "CanonicalClaim":
        from resolution.claim_resolver import CanonicalClaim
        from extraction.schema import Predicate
        return CanonicalClaim(
            canonical_claim_id=claim_id,
            predicate=Predicate(predicate),
            subject_canonical_id=subject_id,
            object_canonical_id=None,
            object_freetext=freetext,
        )

    # Test 17: Two clearly separate claim clusters → flagged
    def test_separate_clusters_flagged(self):
        from resolution.coherence_splitter import CoherenceSplitter

        entity = _make_canonical_entity("ce_1", "GPS", mention_ids=["m1"])

        # Cluster A: GPS navigation claims
        # Cluster B: temperature sensor claims (completely unrelated)
        claims = [
            self._make_claim("c1", "ce_1", "causes", "GPS position drift in hover mode"),
            self._make_claim("c2", "ce_1", "causes", "GPS satellite count drops during flight"),
            self._make_claim("c3", "ce_1", "causes", "GPS fix lost after takeoff"),
            self._make_claim("c4", "ce_1", "causes", "motor temperature exceeds safe operating range"),
            self._make_claim("c5", "ce_1", "causes", "battery overheating during high current draw"),
        ]

        splitter = CoherenceSplitter()
        decisions = splitter.find_splits([entity], claims)

        assert len(decisions) >= 1
        assert decisions[0].canonical_id == "ce_1"

    # Test 18: Coherent claims → not flagged
    def test_coherent_claims_not_flagged(self):
        from resolution.coherence_splitter import CoherenceSplitter

        entity = _make_canonical_entity("ce_1", "GPS", mention_ids=["m1"])

        claims = [
            self._make_claim("c1", "ce_1", "causes", "GPS position drift in hover"),
            self._make_claim("c2", "ce_1", "causes", "GPS satellite lock lost"),
            self._make_claim("c3", "ce_1", "causes", "GPS fix takes too long"),
            self._make_claim("c4", "ce_1", "causes", "GPS accuracy degrades in flight"),
            self._make_claim("c5", "ce_1", "causes", "GPS coordinates jump erratically"),
        ]

        splitter = CoherenceSplitter()
        decisions = splitter.find_splits([entity], claims)

        assert len(decisions) == 0

    # Test 19: Entity with < 5 claims → skipped
    def test_few_claims_skipped(self):
        from resolution.coherence_splitter import CoherenceSplitter

        entity = _make_canonical_entity("ce_1", "GPS", mention_ids=["m1"])
        claims = [
            self._make_claim("c1", "ce_1", "causes", "GPS drift"),
            self._make_claim("c2", "ce_1", "causes", "motor overheat"),
        ]

        splitter = CoherenceSplitter()
        decisions = splitter.find_splits([entity], claims)

        assert len(decisions) == 0

    # Test 20: Split candidates have correct structure
    def test_split_decision_structure(self):
        from resolution.coherence_splitter import CoherenceSplitter

        entity = _make_canonical_entity("ce_1", "GPS", mention_ids=["m1"])

        claims = [
            self._make_claim("c1", "ce_1", "causes", "GPS position drift in hover mode"),
            self._make_claim("c2", "ce_1", "causes", "GPS satellite count drops during flight"),
            self._make_claim("c3", "ce_1", "causes", "GPS fix lost after takeoff"),
            self._make_claim("c4", "ce_1", "causes", "motor temperature exceeds safe operating range"),
            self._make_claim("c5", "ce_1", "causes", "battery overheating during high current draw"),
        ]

        splitter = CoherenceSplitter()
        decisions = splitter.find_splits([entity], claims)

        if decisions:
            d = decisions[0]
            assert d.canonical_id == "ce_1"
            assert d.canonical_name == "GPS"
            assert len(d.proposed_group_a) > 0
            assert len(d.proposed_group_b) > 0
            assert isinstance(d.group_a_sample, str)
            assert isinstance(d.group_b_sample, str)
            assert 0 <= d.confidence <= 1
