from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from graph.retrieval import (
    Conflict,
    ContextPack,
    EnrichedEvidence,
    EvidenceSnippet,
    MAINTAINER_HANDLES,
    MemoryRetriever,
    QueryIntent,
    RankedClaim,
    RelatedEntity,
    SituationPack,
)


# ── helpers ────────────────────────────────────────────────────────

def _make_claim(
    *,
    claim_id: str = "cc_1",
    predicate: str = "causes",
    subject_name: str = "EKF3",
    object_name: str | None = None,
    object_freetext: str | None = "crash on boot",
    confidence: float = 0.9,
    support_level: str = "strong",
    is_current: bool = True,
    valid_until: str | None = None,
    assertion_time: str | None = "2026-01-15T00:00:00Z",
) -> dict:
    """Build a raw claim row dict as returned by Kuzu queries."""
    return {
        "canonical_claim_id": claim_id,
        "predicate": predicate,
        "subject_name": subject_name,
        "object_name": object_name,
        "object_freetext": object_freetext,
        "confidence": confidence,
        "support_level": support_level,
        "is_current": is_current,
        "valid_until": valid_until,
        "assertion_time": assertion_time,
    }


def _mock_kuzu_result(rows: list[list]):
    """Create a mock Kuzu query result that yields rows via get_next/has_next."""
    result = MagicMock()
    idx = {"i": 0}

    def has_next():
        return idx["i"] < len(rows)

    def get_next():
        row = rows[idx["i"]]
        idx["i"] += 1
        return row

    result.has_next = has_next
    result.get_next = get_next
    return result


@pytest.fixture()
def retriever():
    """Create a MemoryRetriever with mocked kuzu database."""
    with patch("graph.retrieval.kuzu") as mock_kuzu:
        mock_db = MagicMock()
        mock_conn = MagicMock()
        mock_kuzu.Database.return_value = mock_db
        mock_kuzu.Connection.return_value = mock_conn
        # Default: execute returns empty result
        mock_conn.execute.return_value = _mock_kuzu_result([])

        r = MemoryRetriever("fake/path.kuzu")
        # Expose mocks for test manipulation
        r._mock_conn = mock_conn
        yield r


# ── _score_claim tests ─────────────────────────────────────────────


class TestScoreClaim:
    def test_current_strong_beats_historical_single(self, retriever):
        """A current claim with strong support should outscore a historical one with single support."""
        current_strong = _make_claim(
            confidence=0.9, is_current=True, support_level="strong",
            assertion_time="2026-01-15T00:00:00Z",
        )
        historical_single = _make_claim(
            confidence=0.7, is_current=False, support_level="single",
            assertion_time="2025-06-01T00:00:00Z",
        )
        score_a = retriever._score_claim(current_strong)
        score_b = retriever._score_claim(historical_single)
        assert score_a > score_b

    def test_confidence_weight_is_0_4(self, retriever):
        """Confidence contributes 40% of the score."""
        high_conf = _make_claim(confidence=1.0, is_current=False, support_level="unsupported", assertion_time=None)
        low_conf = _make_claim(confidence=0.0, is_current=False, support_level="unsupported", assertion_time=None)
        diff = retriever._score_claim(high_conf) - retriever._score_claim(low_conf)
        # confidence weight = 0.4, so diff should be 0.4 * (1.0 - 0.0) = 0.4
        assert abs(diff - 0.4) < 0.01


# ── _recency tests ─────────────────────────────────────────────────


class TestRecency:
    def test_recent_beats_old(self, retriever):
        recent = "2026-03-01T00:00:00Z"
        old = "2024-01-01T00:00:00Z"
        assert retriever._recency(recent) > retriever._recency(old)

    def test_none_returns_0_5(self, retriever):
        assert retriever._recency(None) == 0.5


# ── _match_entities tests ──────────────────────────────────────────


class TestMatchEntities:
    def test_deduplication_by_canonical_id(self, retriever):
        """If the same entity is found in multiple tables, deduplicate by canonical_id."""
        # The LLM returns one entity name
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.tool_calls = [MagicMock()]
        llm_response.choices[0].message.tool_calls[0].function.arguments = json.dumps(
            {"entities": [{"name": "EKF3", "type": "component"}]}
        )

        # Graph returns same entity twice (e.g. matched in two queries)
        def mock_execute(query, parameters=None):
            if "CONTAINS" in query or "contains" in query:
                return _mock_kuzu_result([
                    ["ce_comp_ekf3", "EKF3", "Component", 5],
                    ["ce_comp_ekf3", "EKF3", "Component", 5],  # duplicate
                ])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.return_value = llm_response
            entities = retriever._match_entities("What's wrong with EKF3?")

        # Should be deduplicated to 1
        ids = [e["canonical_id"] for e in entities]
        assert len(set(ids)) == len(ids)


# ── strategy selection tests ───────────────────────────────────────


class TestStrategySelection:
    def test_whats_wrong_selects_component_deep_dive(self, retriever):
        entities = [{"canonical_id": "ce_comp_ekf3", "name": "EKF3", "type": "Component"}]
        assert retriever._select_strategy("What's wrong with EKF3?", entities) == "component_deep_dive"

    def test_who_contributors_selects_person_ranking(self, retriever):
        assert retriever._select_strategy("Who are the main contributors?", []) == "person_ranking"


# ── ContextPack validation ─────────────────────────────────────────


class TestContextPack:
    def test_validates_with_all_fields(self):
        pack = ContextPack(
            question="test?",
            strategy="component_deep_dive",
            matched_entities=[{"canonical_id": "ce_1", "name": "EKF3", "type": "Component"}],
            claims=[
                RankedClaim(
                    canonical_claim_id="cc_1",
                    predicate="causes",
                    subject_name="EKF3",
                    object_name=None,
                    object_freetext="crash",
                    confidence=0.9,
                    support_level="strong",
                    is_current=True,
                    valid_until=None,
                    assertion_time="2026-01-15T00:00:00Z",
                    score=0.85,
                    evidence=[],
                )
            ],
            generated_at="2026-03-05T00:00:00Z",
            query_time_ms=150.0,
        )
        assert pack.question == "test?"
        assert len(pack.claims) == 1


# ── query() integration tests ─────────────────────────────────────


class TestQuery:
    def test_returns_max_10_claims(self, retriever):
        """query() should truncate results to top 10 claims."""
        # Mock LLM intent classification response
        intent_response = MagicMock()
        intent_response.choices = [MagicMock()]
        intent_response.choices[0].message.tool_calls = [MagicMock()]
        intent_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "strategy": "component_deep_dive",
            "entity_names": ["GPS"],
        })

        # Mock LLM synthesis response
        synth_response = MagicMock()
        synth_response.choices = [MagicMock()]
        synth_response.choices[0].message.content = "GPS has many issues."

        # Return 15 claims from the graph
        claim_rows = []
        for i in range(15):
            claim_rows.append([
                f"cc_{i}", "causes", f"symptom {i}", None, 0.8,
                "strong", True, None, "2026-01-01T00:00:00Z",
                "GPS", None,
            ])
        entity_rows = [["ce_comp_gps", "GPS", "Component", 10]]

        def mock_execute(query, parameters=None):
            if "CONTAINS" in query or "contains" in query:
                return _mock_kuzu_result(entity_rows)
            if "HAS_SUBJECT" in query:
                return _mock_kuzu_result(claim_rows)
            if "SUPPORTED_BY" in query:
                return _mock_kuzu_result([])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        with patch("graph.retrieval.litellm") as mock_litellm:
            # First call = intent classification, second = synthesis
            mock_litellm.completion.side_effect = [intent_response, synth_response]
            pack = retriever.query("What's wrong with GPS?")

        assert len(pack.claims) <= 10

    def test_query_time_ms_positive(self, retriever):
        """query() should report positive query_time_ms."""
        # Mock intent classification
        intent_response = MagicMock()
        intent_response.choices = [MagicMock()]
        intent_response.choices[0].message.tool_calls = [MagicMock()]
        intent_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "strategy": "component_deep_dive",
            "entity_names": ["GPS"],
        })

        def mock_execute(query, parameters=None):
            if "CONTAINS" in query or "contains" in query:
                return _mock_kuzu_result([["ce_comp_gps", "GPS", "Component", 5]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.return_value = intent_response
            pack = retriever.query("What's wrong with GPS?")

        assert pack.query_time_ms > 0


# ── EnrichedEvidence tests ────────────────────────────────────────


class TestEnrichedEvidence:
    def test_model_has_document_context_fields(self):
        """EnrichedEvidence should include issue_title, issue_number, issue_state, doc_type."""
        ev = EnrichedEvidence(
            evidence_id="ev_1",
            excerpt="some text",
            author="rmackay9",
            doc_id="ArduPilot/ardupilot#1234",
            issue_title="GPS glitch on boot",
            issue_number=1234,
            issue_state="open",
            doc_type="issue",
            author_is_maintainer=True,
        )
        assert ev.issue_title == "GPS glitch on boot"
        assert ev.issue_number == 1234
        assert ev.issue_state == "open"
        assert ev.doc_type == "issue"
        assert ev.author_is_maintainer is True

    def test_maintainer_flag_default_false(self):
        ev = EnrichedEvidence(evidence_id="ev_2", excerpt="x")
        assert ev.author_is_maintainer is False

    def test_maintainer_handles_constant(self):
        """Known maintainer handles should be in the set."""
        assert "rmackay9" in MAINTAINER_HANDLES
        assert "tridge" in MAINTAINER_HANDLES
        assert "random_user" not in MAINTAINER_HANDLES


class TestFetchEvidenceEnriched:
    def test_returns_enriched_evidence_with_doc_context(self, retriever):
        """_fetch_evidence returns EnrichedEvidence with document context fields."""
        # evidence row: evidence_id, excerpt, author, timestamp, doc_id, source_url,
        #               source_status, doc_type, issue_number, title, state
        evidence_rows = [[
            "ev_1", "crash log here", "rmackay9", "2026-01-15T00:00:00Z",
            "ArduPilot/ardupilot#1234", "https://github.com/1234",
            "confirmed", "issue", 1234, "GPS glitch", "open",
        ]]

        def mock_execute(query, parameters=None):
            if "SUPPORTED_BY" in query:
                return _mock_kuzu_result(evidence_rows)
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        result = retriever._fetch_evidence(["cc_1"])
        assert "cc_1" in result
        ev = result["cc_1"][0]
        assert isinstance(ev, EnrichedEvidence)
        assert ev.issue_title == "GPS glitch"
        assert ev.issue_number == 1234
        assert ev.issue_state == "open"
        assert ev.doc_type == "issue"
        assert ev.author_is_maintainer is True

    def test_comment_doc_gets_parent_issue_context(self, retriever):
        """Comment docs should get parent issue title/state via _doc_cache."""
        # Evidence row for a comment doc — doc_type=comment, no issue_number/title/state
        comment_rows = [[
            "ev_2", "fix applied", "bnsgeyer", "2026-02-01T00:00:00Z",
            "ArduPilot/ardupilot#5678/c/999", "https://github.com/5678#c999",
            "confirmed", "comment", None, None, None,
        ]]

        # Prepopulate the doc cache with the parent issue
        retriever._doc_cache = {
            "ArduPilot/ardupilot#5678": {
                "title": "Motor stall bug",
                "issue_number": 5678,
                "state": "closed",
            }
        }

        def mock_execute(query, parameters=None):
            if "SUPPORTED_BY" in query:
                return _mock_kuzu_result(comment_rows)
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        result = retriever._fetch_evidence(["cc_2"])
        ev = result["cc_2"][0]
        assert ev.issue_title == "Motor stall bug"
        assert ev.issue_number == 5678
        assert ev.issue_state == "closed"
        assert ev.author_is_maintainer is True  # bnsgeyer is a maintainer


# ── SituationPack tests ───────────────────────────────────────────


class TestRelatedEntityModel:
    def test_related_entity_fields(self):
        re = RelatedEntity(
            canonical_id="ce_param_1",
            canonical_name="GPS_TYPE",
            entity_type="Parameter",
            relationship="has_parameter",
            direction="object",
        )
        assert re.canonical_id == "ce_param_1"
        assert re.direction == "object"


class TestSituationPack:
    def test_situation_pack_fields(self):
        pack = SituationPack(
            entity={"canonical_id": "ce_comp_1", "canonical_name": "EKF3"},
            claims_as_subject=[],
            claims_as_object=[],
            related_entities=[],
            mentioned_in_docs=[],
            timeline=[],
            conflicts=[],
        )
        assert pack.entity["canonical_name"] == "EKF3"
        assert pack.conflicts == []


class TestGetEntitySituation:
    def test_returns_situation_with_claims(self, retriever):
        """get_entity_situation returns a SituationPack with claims_as_subject."""
        # We need to handle multiple different queries
        # 1. Entity lookup query
        # 2. Claims as subject query (for each entity type)
        # 3. Claims as object query (for each entity type and object type)
        # 4. Mentioned-in docs query
        # 5. Evidence queries

        entity_row = ["ce_comp_1", "DDFP", "Component", [], 5]
        # Columns: canonical_claim_id, predicate, object_freetext,
        #          confidence, support_level, is_current, valid_until,
        #          assertion_time, e.canonical_name, superseded_by
        claim_row = [
            "cc_1", "causes", "crash on boot", 0.9,
            "strong", True, None, "2026-01-15T00:00:00Z",
            "DDFP", None,
        ]

        def mock_execute(query, parameters=None):
            # Entity node lookup
            if "RETURN n.canonical_id, n.canonical_name" in query:
                return _mock_kuzu_result([entity_row])
            # Claims as subject
            if "HAS_SUBJECT" in query and "Claim_HAS_OBJECT" not in query:
                return _mock_kuzu_result([claim_row])
            # Evidence
            if "SUPPORTED_BY" in query:
                return _mock_kuzu_result([])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        situation = retriever.get_entity_situation("ce_comp_1")
        assert isinstance(situation, SituationPack)
        assert situation.entity["canonical_name"] == "DDFP"
        assert len(situation.claims_as_subject) >= 1
        assert situation.claims_as_subject[0].predicate == "causes"

    def test_returns_empty_situation_for_unknown_entity(self, retriever):
        """get_entity_situation returns empty SituationPack for unknown entity."""
        retriever._mock_conn.execute.return_value = _mock_kuzu_result([])
        situation = retriever.get_entity_situation("ce_unknown_xyz")
        assert isinstance(situation, SituationPack)
        assert situation.claims_as_subject == []
        assert situation.claims_as_object == []


# ── synthesize_answer tests ───────────────────────────────────────


class TestSynthesizeAnswer:
    def test_returns_synthesized_text(self, retriever):
        """synthesize_answer returns a string when LLM succeeds."""
        claims = [
            RankedClaim(
                canonical_claim_id="cc_1", predicate="causes",
                subject_name="DDFP", object_freetext="crash on boot",
                confidence=0.9, support_level="strong",
                is_current=True, score=0.85,
                evidence=[EnrichedEvidence(evidence_id="ev_1", excerpt="crash log")],
            )
        ]
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = "DDFP causes crashes [cc_1]."

        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.return_value = llm_response
            answer = retriever.synthesize_answer("What's wrong with DDFP?", claims)

        assert answer is not None
        assert "DDFP" in answer

    def test_returns_none_on_failure(self, retriever):
        """synthesize_answer returns None when LLM call fails."""
        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = Exception("API error")
            answer = retriever.synthesize_answer("test?", [])

        assert answer is None

    def test_context_pack_has_synthesized_answer(self):
        """ContextPack should have optional synthesized_answer field."""
        pack = ContextPack(
            question="test?", strategy="component_deep_dive",
            matched_entities=[], claims=[],
            generated_at="2026-03-05T00:00:00Z", query_time_ms=100.0,
            synthesized_answer="This is the answer.",
        )
        assert pack.synthesized_answer == "This is the answer."

    def test_context_pack_synthesized_answer_defaults_none(self):
        pack = ContextPack(
            question="test?", strategy="component_deep_dive",
            matched_entities=[], claims=[],
            generated_at="2026-03-05T00:00:00Z", query_time_ms=100.0,
        )
        assert pack.synthesized_answer is None


# ── Conflict detection tests ──────────────────────────────────────


class TestConflictModel:
    def test_conflict_model_fields(self):
        older = RankedClaim(
            canonical_claim_id="cc_old", predicate="causes",
            confidence=0.8, score=0.7, superseded_by="cc_new",
        )
        newer = RankedClaim(
            canonical_claim_id="cc_new", predicate="causes",
            confidence=0.9, score=0.85,
        )
        conflict = Conflict(
            older_claim=older, newer_claim=newer,
            conflict_type="superseded",
            explanation="cc_old was superseded by cc_new",
        )
        assert conflict.conflict_type == "superseded"
        assert conflict.older_claim.superseded_by == "cc_new"


class TestDetectConflicts:
    def test_detects_superseded_claims(self, retriever):
        """detect_conflicts finds superseded claims."""
        older = RankedClaim(
            canonical_claim_id="cc_old", predicate="causes",
            subject_name="DDFP", object_freetext="old bug",
            confidence=0.8, score=0.7, superseded_by="cc_new",
        )
        newer = RankedClaim(
            canonical_claim_id="cc_new", predicate="causes",
            subject_name="DDFP", object_freetext="updated bug",
            confidence=0.9, score=0.85,
        )

        # The newer claim needs to be fetched from kuzu
        def mock_execute(query, parameters=None):
            if "canonical_claim_id" in query and parameters and parameters.get("cid") == "cc_new":
                return _mock_kuzu_result([[
                    "cc_new", "causes", "updated bug", 0.9,
                    "strong", True, None, "2026-02-01T00:00:00Z", None,
                ]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        conflicts = retriever.detect_conflicts([older, newer])
        assert len(conflicts) == 1
        assert conflicts[0].conflict_type == "superseded"
        assert conflicts[0].older_claim.canonical_claim_id == "cc_old"

    def test_no_conflicts_when_no_superseded(self, retriever):
        """detect_conflicts returns empty list when no superseded claims."""
        claim = RankedClaim(
            canonical_claim_id="cc_1", predicate="causes",
            confidence=0.9, score=0.85,
        )
        conflicts = retriever.detect_conflicts([claim])
        assert conflicts == []

    def test_ranked_claim_has_superseded_by(self):
        """RankedClaim should have optional superseded_by field."""
        claim = RankedClaim(
            canonical_claim_id="cc_1", predicate="causes",
            confidence=0.9, score=0.85, superseded_by="cc_2",
        )
        assert claim.superseded_by == "cc_2"


# ── Intent classification tests ───────────────────────────────────


class TestQueryIntentModel:
    def test_query_intent_fields(self):
        intent = QueryIntent(
            strategy="component_deep_dive",
            entity_names=["DDFP", "EKF3"],
            temporal_filter=None,
            confidence_filter=None,
        )
        assert intent.strategy == "component_deep_dive"
        assert len(intent.entity_names) == 2

    def test_query_intent_with_temporal_filter(self):
        intent = QueryIntent(
            strategy="temporal_query",
            entity_names=["DShot"],
            temporal_filter="4.5",
        )
        assert intent.temporal_filter == "4.5"


class TestClassifyIntent:
    def test_classify_returns_intent_from_llm(self, retriever):
        """classify_intent returns QueryIntent from LLM tool call."""
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.tool_calls = [MagicMock()]
        llm_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "strategy": "component_deep_dive",
            "entity_names": ["DDFP"],
            "temporal_filter": None,
            "confidence_filter": None,
        })

        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.return_value = llm_response
            intent = retriever.classify_intent("What's wrong with DDFP?")

        assert isinstance(intent, QueryIntent)
        assert intent.strategy == "component_deep_dive"
        assert "DDFP" in intent.entity_names

    def test_classify_falls_back_on_failure(self, retriever):
        """classify_intent falls back to keyword strategy on LLM failure."""
        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = Exception("timeout")
            intent = retriever.classify_intent("Who are the contributors?")

        assert isinstance(intent, QueryIntent)
        assert intent.strategy == "person_ranking"

    def test_match_entities_by_names(self, retriever):
        """_match_entities_by_names looks up pre-extracted names in graph."""
        def mock_execute(query, parameters=None):
            if "CONTAINS" in query or "contains" in query:
                return _mock_kuzu_result([
                    ["ce_comp_ddfp", "DDFP", "Component", 10],
                ])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute
        entities = retriever._match_entities_by_names(["DDFP"])
        assert len(entities) >= 1
        assert entities[0]["name"] == "DDFP"


# ── temporal_query strategy tests ─────────────────────────────────


class TestTemporalQueryStrategy:
    def test_version_specific_returns_claims(self, retriever):
        """temporal_query with version filter returns version-related claims."""
        intent = QueryIntent(
            strategy="temporal_query",
            entity_names=["DShot"],
            temporal_filter="4.5",
        )
        # Kuzu returns fixed_in claims for that version
        claim_rows = [[
            "cc_fix_1", "fixed_in", "DShot arming bug fixed", 0.9,
            "strong", True, None, "2026-01-15T00:00:00Z",
            "DShot", "4.5.0",
        ]]

        def mock_execute(query, parameters=None):
            if "Claim_HAS_OBJECT_FirmwareVersion" in query:
                return _mock_kuzu_result(claim_rows)
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_temporal_query([], intent)
        assert len(claims) >= 1
        assert claims[0]["predicate"] == "fixed_in"

    def test_general_temporal_returns_current_claims(self, retriever):
        """temporal_query without version returns current fixed_in/regressed_in."""
        intent = QueryIntent(
            strategy="temporal_query",
            entity_names=[],
            temporal_filter=None,
        )
        claim_rows = [[
            "cc_fix_2", "fixed_in", "GPS bug fixed", 0.8,
            "strong", True, None, "2026-02-01T00:00:00Z",
            "GPS",
        ]]

        def mock_execute(query, parameters=None):
            # General temporal uses $pred parameter for fixed_in/regressed_in
            if parameters and parameters.get("pred") in ("fixed_in", "regressed_in"):
                return _mock_kuzu_result(claim_rows)
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_temporal_query([], intent)
        assert len(claims) >= 1

    def test_temporal_query_in_strategy_routing(self, retriever):
        """temporal_query strategy should be available in query() dispatch."""
        intent_response = MagicMock()
        intent_response.choices = [MagicMock()]
        intent_response.choices[0].message.tool_calls = [MagicMock()]
        intent_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "strategy": "temporal_query",
            "entity_names": ["DShot"],
            "temporal_filter": "4.5",
        })

        def mock_execute(query, parameters=None):
            if "CONTAINS" in query or "contains" in query:
                return _mock_kuzu_result([["ce_comp_dshot", "DShot", "Component", 5]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.return_value = intent_response
            pack = retriever.query("Has the DShot arming bug been fixed?")

        assert pack.strategy == "temporal_query"


# ── Fix 1: parent entity resolution tests ─────────────────────


class TestResolveParentEntities:
    def test_resolves_parameter_to_parent_component(self, retriever):
        """When only Parameters are matched, resolve parent Components via has_parameter claims."""
        # Parameter entity only — no Component
        entities = [{"canonical_id": "ce_param_ddfp_spin", "name": "DDFP_SPIN_MAX", "type": "Parameter"}]

        # Simulate graph: Component DDFP → has_parameter → DDFP_SPIN_MAX
        def mock_execute(query, parameters=None):
            if "Claim_HAS_OBJECT_Parameter" in query and "Component" in query:
                return _mock_kuzu_result([
                    ["ce_comp_ddfp", "DDFP", "Component", 10],
                ])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        result = retriever._resolve_parent_entities(entities)
        component_ids = [e["canonical_id"] for e in result if e["type"] == "Component"]
        assert "ce_comp_ddfp" in component_ids

    def test_skips_resolution_when_components_present(self, retriever):
        """If entities already include Components, don't resolve parents."""
        entities = [
            {"canonical_id": "ce_comp_ekf3", "name": "EKF3", "type": "Component"},
            {"canonical_id": "ce_param_x", "name": "EKF3_MAG", "type": "Parameter"},
        ]
        # Should not call the graph at all
        retriever._mock_conn.execute.return_value = _mock_kuzu_result([])

        result = retriever._resolve_parent_entities(entities)
        # Entities unchanged — already has a Component
        assert result == entities

    def test_deduplicates_resolved_parents(self, retriever):
        """Multiple Parameters from the same Component should not duplicate."""
        entities = [
            {"canonical_id": "ce_param_a", "name": "DDFP_SPIN_MAX", "type": "Parameter"},
            {"canonical_id": "ce_param_b", "name": "DDFP_SPIN_MIN", "type": "Parameter"},
        ]

        def mock_execute(query, parameters=None):
            if "Claim_HAS_OBJECT_Parameter" in query:
                return _mock_kuzu_result([
                    ["ce_comp_ddfp", "DDFP", "Component", 10],
                ])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        result = retriever._resolve_parent_entities(entities)
        comp_ids = [e["canonical_id"] for e in result if e["type"] == "Component"]
        assert len(comp_ids) == 1  # Only one DDFP, not duplicated


# ── Fix 2: scoped component_bug_scan tests ────────────────────


class TestScopedComponentBugScan:
    def test_scoped_to_matched_entities(self, retriever):
        """When entities contain Components, only scan those components."""
        entities = [{"canonical_id": "ce_comp_ddfp", "name": "DDFP", "type": "Component"}]

        # Return a bug claim for DDFP
        def mock_execute(query, parameters=None):
            if parameters and parameters.get("eid") == "ce_comp_ddfp":
                return _mock_kuzu_result([[
                    "cc_bug_1", "causes", "motor stall", 0.9,
                    "strong", True, None, "2026-01-01T00:00:00Z",
                    "DDFP",
                ]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_component_bug_scan(entities)
        assert len(claims) >= 1
        assert claims[0]["subject_name"] == "DDFP"

    def test_global_scan_when_no_entities(self, retriever):
        """When no entities matched, do global bug scan."""
        def mock_execute(query, parameters=None):
            if "Component_HAS_SUBJECT" in query and parameters is None:
                return _mock_kuzu_result([[
                    "cc_bug_2", "affects", "GPS glitch", 0.8,
                    "strong", True, None, "2026-01-01T00:00:00Z",
                    "GPS",
                ]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_component_bug_scan([])
        assert len(claims) >= 1

    def test_scoped_includes_hardware_platform(self, retriever):
        """HardwarePlatform entities also scope the bug scan."""
        entities = [{"canonical_id": "ce_hw_pixhawk", "name": "Pixhawk", "type": "HardwarePlatform"}]

        def mock_execute(query, parameters=None):
            if parameters and parameters.get("eid") == "ce_hw_pixhawk":
                return _mock_kuzu_result([[
                    "cc_bug_3", "affects", "sensor failure", 0.85,
                    "strong", True, None, "2026-01-01T00:00:00Z",
                    "Pixhawk",
                ]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_component_bug_scan(entities)
        assert len(claims) >= 1
        assert claims[0]["subject_name"] == "Pixhawk"


# ── Fix 3: scoped temporal_query tests ─────────────────────────


class TestScopedTemporalQuery:
    def test_scoped_to_matched_entities_no_version(self, retriever):
        """temporal_query without version scopes to matched entities."""
        intent = QueryIntent(
            strategy="temporal_query",
            entity_names=["DShot"],
            temporal_filter=None,
        )
        entities = [{"canonical_id": "ce_comp_dshot", "name": "DShot", "type": "Component"}]

        def mock_execute(query, parameters=None):
            if parameters and parameters.get("eid") == "ce_comp_dshot":
                return _mock_kuzu_result([[
                    "cc_fix_scoped", "fixed_in", "DShot arming bug fixed", 0.9,
                    "strong", True, None, "2026-01-15T00:00:00Z",
                    "DShot",
                ]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_temporal_query(entities, intent)
        assert len(claims) >= 1
        assert claims[0]["subject_name"] == "DShot"

    def test_global_scan_when_no_entities_no_version(self, retriever):
        """temporal_query without entities or version does global scan."""
        intent = QueryIntent(
            strategy="temporal_query",
            entity_names=[],
            temporal_filter=None,
        )
        claim_rows = [[
            "cc_fix_global", "fixed_in", "GPS bug fixed", 0.8,
            "strong", True, None, "2026-02-01T00:00:00Z",
            "GPS",
        ]]

        def mock_execute(query, parameters=None):
            if parameters and parameters.get("pred") in ("fixed_in", "regressed_in"):
                return _mock_kuzu_result(claim_rows)
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_temporal_query([], intent)
        assert len(claims) >= 1


# ── Fix 4: evidence_chain accepts all entity types ─────────────


class TestEvidenceChainAllTypes:
    def test_works_with_parameter_entities(self, retriever):
        """evidence_chain should work with Parameter entities, not just Component."""
        entities = [{"canonical_id": "ce_param_ddfp_spin", "name": "DDFP_SPIN_MAX", "type": "Parameter"}]

        def mock_execute(query, parameters=None):
            if "Parameter_HAS_SUBJECT" in query:
                return _mock_kuzu_result([[
                    "cc_cause_1", "causes", "motor overheat", 0.85,
                    "strong", True, None, "2026-01-01T00:00:00Z",
                    "DDFP_SPIN_MAX",
                ]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_evidence_chain(entities)
        assert len(claims) >= 1
        assert claims[0]["subject_name"] == "DDFP_SPIN_MAX"

    def test_works_with_firmware_version(self, retriever):
        """evidence_chain should work with FirmwareVersion entities."""
        entities = [{"canonical_id": "ce_fw_45", "name": "4.5.0", "type": "FirmwareVersion"}]

        def mock_execute(query, parameters=None):
            if "FirmwareVersion_HAS_SUBJECT" in query:
                return _mock_kuzu_result([[
                    "cc_cause_2", "causes", "regression", 0.7,
                    "strong", True, None, "2026-01-01T00:00:00Z",
                    "4.5.0",
                ]])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        claims = retriever._strategy_evidence_chain(entities)
        assert len(claims) >= 1

    def test_skips_invalid_entity_type(self, retriever):
        """evidence_chain skips entities with unknown types."""
        entities = [{"canonical_id": "ce_x", "name": "X", "type": "InvalidType"}]
        retriever._mock_conn.execute.return_value = _mock_kuzu_result([])
        claims = retriever._strategy_evidence_chain(entities)
        assert claims == []


# ── Fix 5: alias search tests ─────────────────────────────────


class TestAliasSearch:
    def test_finds_entity_by_alias(self, retriever):
        """_search_entity_tables should match aliases, not just canonical_name."""
        def mock_execute(query, parameters=None):
            if "UNWIND" in query and "aliases" in query:
                return _mock_kuzu_result([
                    ["ce_comp_ddfp", "DDFP", "Component", 10],
                ])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        results = retriever._search_entity_tables("DirectDriveFP")
        # Should find DDFP via alias match
        assert len(results) >= 1

    def test_alias_query_includes_any_clause(self, retriever):
        """The generated query should include alias search via ANY()."""
        calls = []

        def mock_execute(query, parameters=None):
            calls.append(query)
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        retriever._search_entity_tables("SomeEntity")
        # At least one query should include alias search
        alias_queries = [q for q in calls if "aliases" in q]
        assert len(alias_queries) > 0, "No query searched aliases"


# ── Fix 6: junk name filtering tests ──────────────────────────


class TestJunkNameFiltering:
    def test_filters_unknown_names(self, retriever):
        """_match_entities_by_names should skip junk names like <UNKNOWN>."""
        retriever._mock_conn.execute.return_value = _mock_kuzu_result([])

        entities = retriever._match_entities_by_names(["<UNKNOWN>", "DDFP"])
        # Should have called search only for "DDFP", not "<UNKNOWN>"
        # We verify by checking what was searched
        calls = retriever._mock_conn.execute.call_args_list
        search_terms = [c[1].get("parameters", {}).get("term") for c in calls if c[1].get("parameters")]
        assert "<UNKNOWN>" not in search_terms

    def test_filters_empty_and_none_names(self, retriever):
        """_match_entities_by_names filters out empty strings."""
        retriever._mock_conn.execute.return_value = _mock_kuzu_result([])

        entities = retriever._match_entities_by_names(["", "N/A", "none"])
        # Should not search for any of these
        calls = retriever._mock_conn.execute.call_args_list
        assert len(calls) == 0

    def test_classify_intent_filters_junk_entities(self, retriever):
        """classify_intent should filter junk entity names from LLM response."""
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.tool_calls = [MagicMock()]
        llm_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "strategy": "component_deep_dive",
            "entity_names": ["DDFP", "<UNKNOWN>", "unknown"],
        })

        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.return_value = llm_response
            intent = retriever.classify_intent("What's wrong with DDFP?")

        assert "<UNKNOWN>" not in intent.entity_names
        assert "unknown" not in intent.entity_names
        assert "DDFP" in intent.entity_names


# ── Fix 7: claim deduplication tests ──────────────────────────


class TestClaimDeduplication:
    def test_deduplicates_claims_in_query(self, retriever):
        """query() should deduplicate claims by canonical_claim_id."""
        intent_response = MagicMock()
        intent_response.choices = [MagicMock()]
        intent_response.choices[0].message.tool_calls = [MagicMock()]
        intent_response.choices[0].message.tool_calls[0].function.arguments = json.dumps({
            "strategy": "component_deep_dive",
            "entity_names": ["GPS"],
        })

        synth_response = MagicMock()
        synth_response.choices = [MagicMock()]
        synth_response.choices[0].message.content = "GPS has issues."

        # Return duplicate claims from graph
        claim_rows = [
            ["cc_1", "causes", "GPS glitch", None, 0.9, "strong", True, None, "2026-01-01T00:00:00Z", "GPS", None],
            ["cc_1", "causes", "GPS glitch", None, 0.9, "strong", True, None, "2026-01-01T00:00:00Z", "GPS", None],  # duplicate
            ["cc_2", "affects", "position error", None, 0.8, "strong", True, None, "2026-01-01T00:00:00Z", "GPS", None],
        ]
        entity_rows = [["ce_comp_gps", "GPS", "Component", 10]]

        def mock_execute(query, parameters=None):
            if "CONTAINS" in query or "contains" in query or "ANY" in query:
                return _mock_kuzu_result(entity_rows)
            if "HAS_SUBJECT" in query:
                return _mock_kuzu_result(claim_rows)
            if "SUPPORTED_BY" in query:
                return _mock_kuzu_result([])
            return _mock_kuzu_result([])

        retriever._mock_conn.execute.side_effect = mock_execute

        with patch("graph.retrieval.litellm") as mock_litellm:
            mock_litellm.completion.side_effect = [intent_response, synth_response]
            pack = retriever.query("What's wrong with GPS?")

        # Should have 2 unique claims, not 3
        claim_ids = [c.canonical_claim_id for c in pack.claims]
        assert len(claim_ids) == len(set(claim_ids))
        assert len(pack.claims) == 2
