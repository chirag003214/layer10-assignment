from __future__ import annotations

import logging
from datetime import datetime

from extraction.schema import EntityType, Predicate
from resolution.claim_resolver import CanonicalClaim
from resolution.entity_resolver import CanonicalEntity

logger = logging.getLogger(__name__)


class TemporalResolver:
    def resolve(
        self,
        claims: list[CanonicalClaim],
        entities: dict[str, CanonicalEntity],
    ) -> list[CanonicalClaim]:
        self._apply_fixed_in(claims, entities)
        self._apply_regressed_in(claims, entities)
        return claims

    def _apply_fixed_in(
        self,
        claims: list[CanonicalClaim],
        entities: dict[str, CanonicalEntity],
    ) -> None:
        """For each FIXED_IN claim pointing to a FIRMWARE_VERSION, mark matching
        AFFECTS/CAUSES claims on the same subject as historical. Also set event_time."""
        fixed_claims = [c for c in claims if c.predicate == Predicate.FIXED_IN]

        for fc in fixed_claims:
            if fc.object_canonical_id is None:
                continue
            obj_entity = entities.get(fc.object_canonical_id)
            if obj_entity is None or obj_entity.entity_type != EntityType.FIRMWARE_VERSION:
                continue

            version_str = obj_entity.attributes.get(
                "version_string", obj_entity.canonical_name
            )

            fc.event_time = version_str

            for c in claims:
                if c.predicate not in (Predicate.AFFECTS, Predicate.CAUSES):
                    continue
                if c.subject_canonical_id != fc.subject_canonical_id:
                    continue
                c.is_current = False
                c.valid_until = version_str

    def _apply_regressed_in(
        self,
        claims: list[CanonicalClaim],
        entities: dict[str, CanonicalEntity],
    ) -> None:
        """For each REGRESSED_IN claim pointing to a FIRMWARE_VERSION, set
        valid_from on the claim itself and event_time."""
        regressed_claims = [c for c in claims if c.predicate == Predicate.REGRESSED_IN]

        for rc in regressed_claims:
            if rc.object_canonical_id is None:
                continue
            obj_entity = entities.get(rc.object_canonical_id)
            if obj_entity is None or obj_entity.entity_type != EntityType.FIRMWARE_VERSION:
                continue

            version_str = obj_entity.attributes.get(
                "version_string", obj_entity.canonical_name
            )
            rc.valid_from = version_str
            rc.event_time = version_str


class SupersedingResolver:
    """Marks older REGRESSED_IN claims as superseded by the most recent one per subject."""

    def resolve(self, claims: list[CanonicalClaim]) -> list[CanonicalClaim]:
        self._apply_regressed_in_superseding(claims)
        return claims

    def _apply_regressed_in_superseding(self, claims: list[CanonicalClaim]) -> None:
        """For each subject, sort REGRESSED_IN claims by assertion_time descending.
        Mark oldest as superseded by the newest."""
        regressed = [c for c in claims if c.predicate == Predicate.REGRESSED_IN]

        # Group by subject
        by_subject: dict[str, list[CanonicalClaim]] = {}
        for c in regressed:
            by_subject.setdefault(c.subject_canonical_id, []).append(c)

        for subj_id, group in by_subject.items():
            if len(group) <= 1:
                continue

            # Sort by assertion_time descending (None sorts last)
            sorted_group = sorted(
                group,
                key=lambda c: c.assertion_time or datetime.min.replace(tzinfo=None),
                reverse=True,
            )

            # Most recent is current
            sorted_group[0].is_current = True
            sorted_group[0].superseded_by = None

            # Older ones are superseded
            latest_id = sorted_group[0].canonical_claim_id
            for older in sorted_group[1:]:
                older.is_current = False
                older.superseded_by = latest_id
