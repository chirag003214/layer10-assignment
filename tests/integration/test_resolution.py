from __future__ import annotations

import pytest

from extraction.schema import EntityType, Predicate
from resolution.claim_resolver import ClaimResolver
from resolution.temporal_resolver import TemporalResolver
from tests.integration.fixtures import (
    bug_report_doc,
    fix_confirmation_doc,
    fuzzy_merge_pair_docs,
    multi_component_doc,
)

pytestmark = pytest.mark.integration


# --------------------------------------------------------------------------- #
# Entity resolution
# --------------------------------------------------------------------------- #

class TestDistinctComponentsNotMerged:
    def test_ekf2_and_ekf3_stay_separate(self, extractor, entity_resolver):
        result = extractor.extract(multi_component_doc())
        entities, _, _ = entity_resolver.resolve(result.entity_mentions)

        component_entities = [e for e in entities if e.entity_type == EntityType.COMPONENT]
        all_names = set()
        for ce in component_entities:
            all_names.add(ce.canonical_name.lower())
            all_names.update(a.lower() for a in ce.aliases)

        ekf2_entities = [
            ce for ce in component_entities
            if "ekf2" in ce.canonical_name.lower()
            or any("ekf2" in a.lower() for a in ce.aliases)
        ]
        ekf3_entities = [
            ce for ce in component_entities
            if "ekf3" in ce.canonical_name.lower()
            or any("ekf3" in a.lower() for a in ce.aliases)
        ]

        # If both were extracted, they should be in distinct canonical entities
        if ekf2_entities and ekf3_entities:
            ekf2_ids = {ce.canonical_id for ce in ekf2_entities}
            ekf3_ids = {ce.canonical_id for ce in ekf3_entities}
            assert not ekf2_ids & ekf3_ids, (
                "EKF2 and EKF3 should not share a canonical entity"
            )


class TestFuzzyMergeSameComponent:
    def test_scripting_variants_merge(self, extractor, entity_resolver):
        doc_a, doc_b = fuzzy_merge_pair_docs()
        result_a = extractor.extract(doc_a)
        result_b = extractor.extract(doc_b)

        all_mentions = result_a.entity_mentions + result_b.entity_mentions
        entities, _, _ = entity_resolver.resolve(all_mentions)

        component_entities = [e for e in entities if e.entity_type == EntityType.COMPONENT]

        # Collect scripting-related canonical entities
        scripting_entities = [
            ce for ce in component_entities
            if "script" in ce.canonical_name.lower()
            or any("script" in a.lower() for a in ce.aliases)
        ]

        # At least some scripting mentions should have been merged together
        if scripting_entities:
            max_mentions = max(len(ce.mention_ids) for ce in scripting_entities)
            assert max_mentions >= 2, (
                f"Expected scripting mentions from both docs to merge; "
                f"max mention count per canonical entity: {max_mentions}"
            )


# --------------------------------------------------------------------------- #
# Claim resolution
# --------------------------------------------------------------------------- #

class TestClaimResolutionRewritesIds:
    def test_canonical_claims_use_canonical_entity_ids(self, extractor, entity_resolver):
        result = extractor.extract(bug_report_doc())
        if not result.claim_mentions:
            pytest.skip("No claims extracted from bug report")

        entities, _, _ = entity_resolver.resolve(result.entity_mentions)

        # Build mention -> canonical mapping
        mention_to_canonical = {}
        for ce in entities:
            for mid in ce.mention_ids:
                mention_to_canonical[mid] = ce.canonical_id

        resolver = ClaimResolver(
            result.claim_mentions, mention_to_canonical, result.evidence_records,
        )
        canonical_claims, _ = resolver.resolve()

        canonical_ids = {ce.canonical_id for ce in entities}
        for cc in canonical_claims:
            assert cc.subject_canonical_id in canonical_ids, (
                f"Claim subject {cc.subject_canonical_id} not in canonical entities"
            )
            assert cc.subject_canonical_id.startswith("ce_"), (
                f"Expected canonical ID format ce_*, got {cc.subject_canonical_id}"
            )
            if cc.object_canonical_id:
                assert cc.object_canonical_id in canonical_ids, (
                    f"Claim object {cc.object_canonical_id} not in canonical entities"
                )


# --------------------------------------------------------------------------- #
# Temporal resolution
# --------------------------------------------------------------------------- #

class TestTemporalResolution:
    def test_fixed_in_marks_affects_as_historical(self, extractor, entity_resolver):
        bug_result = extractor.extract(bug_report_doc())
        fix_result = extractor.extract(fix_confirmation_doc())

        all_mentions = bug_result.entity_mentions + fix_result.entity_mentions
        all_claims = bug_result.claim_mentions + fix_result.claim_mentions
        all_evidence = bug_result.evidence_records + fix_result.evidence_records

        entities, _, _ = entity_resolver.resolve(all_mentions)
        mention_to_canonical = {}
        for ce in entities:
            for mid in ce.mention_ids:
                mention_to_canonical[mid] = ce.canonical_id

        resolver = ClaimResolver(all_claims, mention_to_canonical, all_evidence)
        canonical_claims, _ = resolver.resolve()

        has_fixed = any(c.predicate == Predicate.FIXED_IN for c in canonical_claims)
        has_affects = any(
            c.predicate in (Predicate.AFFECTS, Predicate.CAUSES) for c in canonical_claims
        )

        if not has_fixed or not has_affects:
            pytest.skip("Need both FIXED_IN and AFFECTS/CAUSES claims for temporal test")

        entity_map = {ce.canonical_id: ce for ce in entities}
        temporal = TemporalResolver()
        temporal.resolve(canonical_claims, entity_map)

        # Find FIXED_IN subjects
        fixed_subjects = {
            c.subject_canonical_id
            for c in canonical_claims
            if c.predicate == Predicate.FIXED_IN
        }

        for c in canonical_claims:
            if c.predicate in (Predicate.AFFECTS, Predicate.CAUSES):
                if c.subject_canonical_id in fixed_subjects:
                    assert c.is_current is False, (
                        f"AFFECTS/CAUSES claim {c.canonical_claim_id} should be marked "
                        f"historical after FIXED_IN on same subject"
                    )
                    assert c.valid_until is not None
