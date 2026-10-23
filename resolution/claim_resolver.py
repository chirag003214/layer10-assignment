from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel

from extraction.schema import ClaimMention, EvidenceRecord, Predicate

logger = logging.getLogger(__name__)

# Predicates where grouping ignores object identity — collect freetext variants
_VARIANT_PREDICATES = {Predicate.CAUSES, Predicate.REGRESSED_IN}


class ClaimProvenance(BaseModel):
    doc_id: str
    source_hash: str
    predicate: str
    extraction_schema_version: str
    extraction_prompt_version: str
    extracted_at: datetime


@dataclass
class CanonicalClaim:
    canonical_claim_id: str
    predicate: Predicate
    subject_canonical_id: str
    object_canonical_id: str | None
    object_freetext: str | None
    object_freetext_variants: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    provenance: list[ClaimProvenance] = field(default_factory=list)
    confidence: float = 0.0
    is_current: bool = True
    valid_from: str | None = None
    valid_until: str | None = None
    assertion_time: datetime | None = None
    event_time: str | None = None
    superseded_by: str | None = None
    grounding_status: str = "fully_grounded"


class ClaimResolver:
    def __init__(
        self,
        claims: list[ClaimMention],
        mention_to_canonical: dict[str, str],
        evidence_records: list[EvidenceRecord],
    ) -> None:
        self._claims = claims
        self._mapping = mention_to_canonical
        self._evidence = {er.evidence_id: er for er in evidence_records}

    def resolve(self) -> tuple[list[CanonicalClaim], int]:
        rewritten, skipped = self._rewrite_ids()
        groups = self._group(rewritten)
        canonical = self._merge(groups)
        return canonical, skipped

    # ---- internal -----------------------------------------------------------

    def _make_group_key(
        self,
        cm: ClaimMention,
        subj: str,
        obj: str | None,
    ) -> tuple:
        """Return the grouping key for a claim. All grouping logic lives here."""
        p = cm.predicate

        if p == Predicate.CAUSES:
            # Each distinct symptom is a separate claim; group by (subject, predicate, symptom)
            return (subj, p, cm.object_freetext)

        elif p == Predicate.AFFECTS:
            return (subj, p, None)

        elif p == Predicate.FIXED_IN:
            return (subj, p, obj)

        elif p == Predicate.REGRESSED_IN:
            # Group by version (object_canonical_id), not description
            return (subj, p, obj)

        elif p == Predicate.HAS_PARAMETER:
            # Include context_hash to distinguish same (subject, predicate, object) with different values
            context_hash = hashlib.sha1(
                (cm.object_freetext or "").encode()
            ).hexdigest()[:8]
            return (subj, p, obj, context_hash)

        elif p == Predicate.DUPLICATE_OF:
            # Subject is not part of key — group all duplicate statements about same target
            return (p, cm.object_freetext)

        else:
            # Fallback for unknown predicates
            return (subj, p, obj, cm.object_freetext)

    def _rewrite_ids(
        self,
    ) -> tuple[list[tuple[ClaimMention, str, str | None]], int]:
        """Rewrite mention IDs to canonical IDs. Returns (claim, subj_canon, obj_canon) tuples."""
        results: list[tuple[ClaimMention, str, str | None]] = []
        skipped = 0
        for cm in self._claims:
            subj = self._mapping.get(cm.subject_mention_id)
            if subj is None:
                logger.warning(
                    "Skipping claim %s: subject mention %s not in canonical mapping",
                    cm.claim_mention_id,
                    cm.subject_mention_id,
                )
                skipped += 1
                continue

            obj: str | None = None
            if cm.object_mention_id is not None:
                obj = self._mapping.get(cm.object_mention_id)
                if obj is None:
                    logger.warning(
                        "Claim %s: object mention %s not in canonical mapping, setting to None",
                        cm.claim_mention_id,
                        cm.object_mention_id,
                    )

            results.append((cm, subj, obj))
        return results, skipped

    def _group(
        self,
        rewritten: list[tuple[ClaimMention, str, str | None]],
    ) -> dict[tuple, list[tuple[ClaimMention, str, str | None]]]:
        groups: dict[tuple, list[tuple[ClaimMention, str, str | None]]] = {}
        for item in rewritten:
            cm, subj, obj = item
            key = self._make_group_key(cm, subj, obj)
            groups.setdefault(key, []).append(item)
        return groups

    def _build_provenance(self, cm: ClaimMention) -> ClaimProvenance:
        source_hash = ""
        if cm.evidence_ids:
            er = self._evidence.get(cm.evidence_ids[0])
            if er is not None:
                source_hash = er.source_hash
        return ClaimProvenance(
            doc_id=cm.doc_id,
            source_hash=source_hash,
            predicate=cm.predicate.value,
            extraction_schema_version=cm.extraction_metadata.schema_version,
            extraction_prompt_version=cm.extraction_metadata.prompt_version,
            extracted_at=cm.extraction_metadata.extracted_at,
        )

    def _select_duplicate_of_subject(
        self, items: list[tuple[ClaimMention, str, str | None]]
    ) -> str:
        """Pick the subject from the claim with the earliest assertion_time."""
        best_cm, best_subj, _ = items[0]
        best_time: datetime | None = None

        for cm, subj, _ in items:
            # Get the min timestamp across evidence for this claim
            timestamps = [
                self._evidence[eid].timestamp
                for eid in cm.evidence_ids
                if eid in self._evidence
            ]
            if timestamps:
                claim_min = min(timestamps)
                if best_time is None or claim_min < best_time:
                    best_time = claim_min
                    best_cm = cm
                    best_subj = subj

        return best_subj

    def _merge(
        self,
        groups: dict[tuple, list[tuple[ClaimMention, str, str | None]]],
    ) -> list[CanonicalClaim]:
        canonical: list[CanonicalClaim] = []
        for key, items in groups.items():
            first_cm, first_subj, first_obj = items[0]
            predicate = first_cm.predicate

            # Collect all object canonical IDs and freetext values
            obj_canonical = first_obj
            freetext_variants: list[str] = []
            seen_freetext: set[str] = set()

            for cm, subj, obj in items:
                if obj is not None:
                    obj_canonical = obj
                if cm.object_freetext and cm.object_freetext not in seen_freetext:
                    freetext_variants.append(cm.object_freetext)
                    seen_freetext.add(cm.object_freetext)

            # For CAUSES: each group already has a single freetext (from key)
            # For other variant predicates, pick longest freetext as canonical
            if predicate in _VARIANT_PREDICATES and freetext_variants:
                canonical_freetext = max(freetext_variants, key=len)
            elif freetext_variants:
                canonical_freetext = freetext_variants[0]
            else:
                canonical_freetext = None

            # Union evidence_ids (deduplicated, insertion order)
            evidence_ids: list[str] = []
            seen_ev: set[str] = set()
            for cm, _, _ in items:
                for eid in cm.evidence_ids:
                    if eid not in seen_ev:
                        evidence_ids.append(eid)
                        seen_ev.add(eid)

            # Max confidence
            confidence = max(cm.confidence for cm, _, _ in items)

            # Build provenance (one entry per source ClaimMention)
            provenance = [self._build_provenance(cm) for cm, _, _ in items]

            # Compute assertion_time: max timestamp across known evidence records
            valid_timestamps = [
                self._evidence[eid].timestamp
                for eid in evidence_ids
                if eid in self._evidence
            ]
            assertion_time = max(valid_timestamps) if valid_timestamps else None

            # For DUPLICATE_OF: select subject from earliest claim
            if predicate == Predicate.DUPLICATE_OF:
                subject_id = self._select_duplicate_of_subject(items)
            else:
                subject_id = first_subj

            # Generate canonical_claim_id — include all key components to avoid collisions
            if predicate == Predicate.HAS_PARAMETER:
                # key is (subj, p, obj, context_hash)
                context_hash = key[3] if len(key) > 3 else ""
                obj_part = obj_canonical if obj_canonical else "freetext"
                hash_input = f"{subject_id}{predicate.value}{obj_part}{context_hash}"
            elif predicate == Predicate.CAUSES:
                # key is (subj, p, object_freetext)
                freetext_key = key[2] or ""
                hash_input = f"{subject_id}{predicate.value}{freetext_key}"
            elif predicate == Predicate.DUPLICATE_OF:
                freetext_key = key[1] or ""
                hash_input = f"{predicate.value}{freetext_key}"
            else:
                obj_part = obj_canonical if obj_canonical else "freetext"
                hash_input = f"{subject_id}{predicate.value}{obj_part}"

            claim_id = f"cc_{hashlib.sha1(hash_input.encode()).hexdigest()[:8]}"

            canonical.append(
                CanonicalClaim(
                    canonical_claim_id=claim_id,
                    predicate=predicate,
                    subject_canonical_id=subject_id,
                    object_canonical_id=obj_canonical,
                    object_freetext=canonical_freetext,
                    object_freetext_variants=freetext_variants,
                    evidence_ids=evidence_ids,
                    provenance=provenance,
                    confidence=confidence,
                    assertion_time=assertion_time,
                )
            )

        return canonical
