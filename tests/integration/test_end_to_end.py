from __future__ import annotations

import pytest

from extraction.schema import Predicate
from resolution.claim_resolver import ClaimResolver
from resolution.temporal_resolver import TemporalResolver
from tests.integration.fixtures import (
    bug_report_doc,
    duplicate_issue_doc,
    fix_confirmation_doc,
    parameter_discussion_doc,
)

pytestmark = pytest.mark.integration


class TestFullPipeline:
    """Smoke test: extract from multiple docs, resolve, run temporal reasoning."""

    def test_produces_valid_graph_data(self, extractor, entity_resolver):
        docs = [
            bug_report_doc(),
            fix_confirmation_doc(),
            duplicate_issue_doc(),
            parameter_discussion_doc(),
        ]

        # --- Extract ---
        results = [extractor.extract(doc) for doc in docs]

        all_entity_mentions = []
        all_claim_mentions = []
        all_evidence_records = []
        for r in results:
            all_entity_mentions.extend(r.entity_mentions)
            all_claim_mentions.extend(r.claim_mentions)
            all_evidence_records.extend(r.evidence_records)

        assert len(all_entity_mentions) > 0, "Pipeline produced no entities"

        # --- Resolve entities ---
        entities, merge_records, decisions = entity_resolver.resolve(all_entity_mentions)
        assert len(entities) > 0, "Entity resolution produced no canonical entities"

        # No duplicate canonical_ids
        canonical_ids = [ce.canonical_id for ce in entities]
        assert len(canonical_ids) == len(set(canonical_ids)), (
            f"Duplicate canonical IDs: {canonical_ids}"
        )

        # Every mention appears in exactly one canonical entity
        all_mapped_mids = []
        for ce in entities:
            all_mapped_mids.extend(ce.mention_ids)
        assert len(all_mapped_mids) == len(set(all_mapped_mids)), "Mention mapped to >1 entity"
        assert set(all_mapped_mids) == {m.mention_id for m in all_entity_mentions}

        # --- Resolve claims ---
        mention_to_canonical = {}
        for ce in entities:
            for mid in ce.mention_ids:
                mention_to_canonical[mid] = ce.canonical_id

        if not all_claim_mentions:
            pytest.skip("No claims extracted — nothing to resolve")

        resolver = ClaimResolver(
            all_claim_mentions, mention_to_canonical, all_evidence_records,
        )
        canonical_claims, skipped = resolver.resolve()

        canonical_entity_ids = {ce.canonical_id for ce in entities}
        all_ev_ids = {ev.evidence_id for ev in all_evidence_records}

        for cc in canonical_claims:
            # Every canonical claim references a valid canonical entity
            assert cc.subject_canonical_id in canonical_entity_ids, (
                f"Claim {cc.canonical_claim_id} subject {cc.subject_canonical_id} "
                f"not in canonical entities"
            )
            if cc.object_canonical_id:
                assert cc.object_canonical_id in canonical_entity_ids

            # Evidence IDs in claims are non-empty and valid
            assert len(cc.evidence_ids) > 0, (
                f"Claim {cc.canonical_claim_id} has no evidence IDs"
            )
            for eid in cc.evidence_ids:
                assert eid in all_ev_ids, (
                    f"Evidence {eid} in claim {cc.canonical_claim_id} not found"
                )

        # No duplicate canonical_claim_ids
        claim_ids = [cc.canonical_claim_id for cc in canonical_claims]
        assert len(claim_ids) == len(set(claim_ids)), "Duplicate canonical claim IDs"

        # --- Temporal resolution ---
        entity_map = {ce.canonical_id: ce for ce in entities}
        temporal = TemporalResolver()
        temporal.resolve(canonical_claims, entity_map)

        # Temporal fields are consistent
        for cc in canonical_claims:
            if not cc.is_current:
                assert cc.valid_until is not None, (
                    f"Claim {cc.canonical_claim_id} is_current=False but valid_until is None"
                )

        # Check that at least some predicates were extracted across all docs
        found_predicates = {cc.predicate for cc in canonical_claims}
        assert len(found_predicates) >= 2, (
            f"Expected at least 2 distinct predicates across 4 docs, got {found_predicates}"
        )
