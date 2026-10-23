from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from extraction.schema import EntityMention, EntityType
from resolution.merge_resolver import MergeCandidate, MergeDecision, MergeResolver


@dataclass
class CanonicalEntity:
    canonical_id: str
    entity_type: EntityType
    canonical_name: str
    aliases: list[str]
    attributes: dict
    mention_ids: list[str]
    first_seen: datetime
    last_seen: datetime


@dataclass
class MergeRecord:
    canonical_id: str
    canonical_name: str
    merged_mention_ids: list[str]
    strategy: str
    similarity_score: float | None
    decided_at: datetime


ResolutionOutcome = Literal["merge", "keep_distinct", "llm_candidate"]


@dataclass
class ResolutionDecision:
    decision: ResolutionOutcome
    strategy: str
    similarity_score: float | None = None


class BaseEntityResolver(ABC):
    entity_type: EntityType

    @abstractmethod
    def make_key(self, mention: EntityMention) -> Any:
        """Deterministic identity key. Same key = same entity."""

    @abstractmethod
    def should_merge(
        self,
        a: EntityMention,
        b: EntityMention,
        **kwargs: Any,
    ) -> ResolutionDecision:
        """Called only when make_key(a) != make_key(b)."""

    def canonical_name(self, mentions: list[EntityMention]) -> str:
        counts = Counter(m.surface_form for m in mentions)
        max_count = max(counts.values())
        candidates = [sf for sf, c in counts.items() if c == max_count]
        return max(candidates, key=len)

    def canonical_attrs(self, mentions: list[EntityMention]) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        for m in mentions:
            attrs.update(m.attributes)
        return attrs

    def resolve_group(
        self,
        mentions: list[EntityMention],
        llm_resolver: MergeResolver | None = None,
    ) -> tuple[list[CanonicalEntity], list[MergeRecord], list[MergeDecision]]:
        if not mentions:
            return [], [], []

        mid_to_mention = {m.mention_id: m for m in mentions}

        # --- Union-Find ---
        parent: dict[str, str] = {m.mention_id: m.mention_id for m in mentions}
        group_strategy: dict[str, str] = {}
        group_score: dict[str, float | None] = {}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str, strategy: str, score: float | None = None) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[rx] = ry
                group_strategy[ry] = strategy
                group_score[ry] = score
                group_strategy.pop(rx, None)

        # Step 1: Exact key groups
        key_to_mids: dict[Any, list[str]] = defaultdict(list)
        for m in mentions:
            key_to_mids[self.make_key(m)].append(m.mention_id)

        # Build key → doc_ids mapping for co-document checks
        key_to_doc_ids: dict[Any, set[str]] = defaultdict(set)
        for m in mentions:
            key_to_doc_ids[self.make_key(m)].add(m.doc_id)

        for mids in key_to_mids.values():
            rep = mids[0]
            for mid in mids[1:]:
                union(rep, mid, strategy="exact_key_match")

        # Step 2: Cross-group comparisons using key representatives
        key_reps = [(k, mids[0]) for k, mids in key_to_mids.items()]
        llm_candidates: list[MergeCandidate] = []

        for i in range(len(key_reps)):
            for j in range(i + 1, len(key_reps)):
                ki, ri = key_reps[i]
                kj, rj = key_reps[j]
                if find(ri) == find(rj):
                    continue
                ma = mid_to_mention[ri]
                mb = mid_to_mention[rj]
                dec = self.should_merge(
                    ma, mb,
                    group_doc_ids_a=key_to_doc_ids.get(ki),
                    group_doc_ids_b=key_to_doc_ids.get(kj),
                )
                if dec.decision == "merge":
                    union(ri, rj, strategy=dec.strategy, score=dec.similarity_score)
                elif dec.decision == "llm_candidate":
                    llm_candidates.append(
                        MergeCandidate(
                            mention_a=ma,
                            mention_b=mb,
                            similarity_score=dec.similarity_score or 0.0,
                        )
                    )
                # "keep_distinct" → nothing

        # Step 3: LLM decisions for candidates
        llm_decisions: list[MergeDecision] = []
        if llm_candidates and llm_resolver is not None:
            llm_decisions = llm_resolver.resolve(llm_candidates)
            for dec in llm_decisions:
                if dec.decision == "merge":
                    union(
                        dec.mention_id_a,
                        dec.mention_id_b,
                        strategy="llm_merge_decision",
                        score=dec.similarity_score,
                    )

        # Step 4: Build final groups
        root_to_mids: dict[str, list[str]] = defaultdict(list)
        for m in mentions:
            root_to_mids[find(m.mention_id)].append(m.mention_id)

        # Step 5: Build CanonicalEntity + MergeRecord per group
        entities: list[CanonicalEntity] = []
        records: list[MergeRecord] = []

        for root, mids in root_to_mids.items():
            group_mentions = [mid_to_mention[mid] for mid in mids]
            name = self.canonical_name(group_mentions)
            aliases = sorted({m.surface_form for m in group_mentions} - {name})
            attrs = self.canonical_attrs(group_mentions)
            first_seen = min(m.extraction_metadata.extracted_at for m in group_mentions)
            last_seen = max(m.extraction_metadata.extracted_at for m in group_mentions)
            canonical_id = (
                f"ce_{self.entity_type.value}_"
                f"{hashlib.sha1(name.encode()).hexdigest()[:8]}"
            )
            entities.append(
                CanonicalEntity(
                    canonical_id=canonical_id,
                    entity_type=self.entity_type,
                    canonical_name=name,
                    aliases=aliases,
                    attributes=attrs,
                    mention_ids=mids,
                    first_seen=first_seen,
                    last_seen=last_seen,
                )
            )
            if len(mids) > 1:
                strategy = group_strategy.get(root, "exact_key_match")
                records.append(
                    MergeRecord(
                        canonical_id=canonical_id,
                        canonical_name=name,
                        merged_mention_ids=mids,
                        strategy=strategy,
                        similarity_score=group_score.get(root),
                        decided_at=datetime.now(),
                    )
                )

        return entities, records, llm_decisions


class PersonResolver(BaseEntityResolver):
    entity_type = EntityType.PERSON

    def make_key(self, mention: EntityMention) -> Any:
        gh = mention.attributes.get("github_handle", "")
        if gh:
            return ("handle", gh.lower())
        return ("no_handle", mention.normalized_form)

    def should_merge(self, a: EntityMention, b: EntityMention, **kwargs: Any) -> ResolutionDecision:
        return ResolutionDecision(
            decision="keep_distinct", strategy="person_different_handles"
        )

    def canonical_name(self, mentions: list[EntityMention]) -> str:
        handles = [
            m.attributes.get("github_handle")
            for m in mentions
            if m.attributes.get("github_handle")
        ]
        if handles:
            return Counter(handles).most_common(1)[0][0]
        return super().canonical_name(mentions)


class ParameterResolver(BaseEntityResolver):
    entity_type = EntityType.PARAMETER

    def make_key(self, mention: EntityMention) -> Any:
        return mention.surface_form.strip().upper().replace(" ", "_")

    def should_merge(self, a: EntityMention, b: EntityMention, **kwargs: Any) -> ResolutionDecision:
        return ResolutionDecision(
            decision="keep_distinct", strategy="parameter_different_names"
        )

    def canonical_attrs(self, mentions: list[EntityMention]) -> dict[str, Any]:
        distinct_values = list(
            {
                m.attributes.get("value")
                for m in mentions
                if m.attributes.get("value") is not None
            }
        )
        attrs: dict[str, Any] = {}
        for m in mentions:
            attrs.update(m.attributes)
        if len(distinct_values) == 0:
            attrs["value"] = None
        elif len(distinct_values) == 1:
            attrs["value"] = distinct_values[0]
        else:
            attrs["value"] = sorted(distinct_values, key=str)
        return attrs


class FirmwareVersionResolver(BaseEntityResolver):
    entity_type = EntityType.FIRMWARE_VERSION

    @staticmethod
    def _strip_prefix(s: str) -> str:
        return re.sub(r"^[A-Za-z]+-", "", s).strip()

    def make_key(self, mention: EntityMention) -> Any:
        version = self._strip_prefix(
            mention.attributes.get("version_string", mention.surface_form)
        )
        vehicle = mention.attributes.get("vehicle_type")
        return (version, vehicle)

    def should_merge(self, a: EntityMention, b: EntityMention, **kwargs: Any) -> ResolutionDecision:
        va = self._strip_prefix(a.attributes.get("version_string", a.surface_form))
        vb = self._strip_prefix(b.attributes.get("version_string", b.surface_form))
        if va != vb:
            return ResolutionDecision(
                decision="keep_distinct", strategy="firmware_different_versions"
            )
        vta = a.attributes.get("vehicle_type")
        vtb = b.attributes.get("vehicle_type")
        if vta is None or vtb is None:
            return ResolutionDecision(
                decision="merge", strategy="firmware_null_vehicle_merge"
            )
        return ResolutionDecision(
            decision="keep_distinct", strategy="firmware_different_vehicle_types"
        )

    def canonical_name(self, mentions: list[EntityMention]) -> str:
        with_vehicle = [m for m in mentions if m.attributes.get("vehicle_type")]
        pool = with_vehicle if with_vehicle else mentions
        return super().canonical_name(pool)


class SemanticEntityResolver(BaseEntityResolver):
    """Shared logic for COMPONENT and HARDWARE_PLATFORM.

    Five-step structural decision tree (no fuzzy matching):
    1. Numeric variant guard → keep_distinct
    2. Exact normalized match → merge
    3. Structural key match (suffix/prefix stripped) → merge
    4. Co-document Levenshtein-1 → llm_candidate (or distinct if no shared doc)
    4b. Shared prefix (>=4) co-document → llm_candidate
    5. Default → keep_distinct
    """

    GENERIC_SUFFIXES = {
        "module", "subsystem", "system", "handler", "controller", "driver",
        "library", "support", "interface", "backend", "service", "handling",
        "management",
    }

    NAMESPACE_PREFIXES = ("ap_", "ardu")

    @staticmethod
    def _is_numeric_variant(shorter: str, longer: str) -> bool:
        suffix = longer[len(shorter):]
        return len(suffix) > 0 and suffix.isdigit()

    @classmethod
    def _strip_generic_suffix(cls, form: str) -> str:
        """Strip a trailing ' <suffix>' or '_<suffix>' if it's a generic suffix."""
        for sep in (" ", "_"):
            idx = form.rfind(sep)
            if idx > 0 and form[idx + 1:] in cls.GENERIC_SUFFIXES:
                return form[:idx]
        return form

    @classmethod
    def _strip_namespace_prefix(cls, form: str) -> str:
        """Strip a leading namespace prefix (ap_, ardu)."""
        for prefix in cls.NAMESPACE_PREFIXES:
            if form.startswith(prefix):
                return form[len(prefix):]
        return form

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        """Simple O(mn) Levenshtein distance for short entity names."""
        m, n = len(a), len(b)
        if m == 0:
            return n
        if n == 0:
            return m
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
            prev, curr = curr, prev
        return prev[n]

    @classmethod
    def _structural_key(cls, nf: str) -> str:
        """Compute structural key: strip namespace prefix, then generic suffix."""
        return cls._strip_namespace_prefix(cls._strip_generic_suffix(nf))

    def make_key(self, mention: EntityMention) -> Any:
        return mention.normalized_form

    def should_merge(
        self,
        a: EntityMention,
        b: EntityMention,
        *,
        group_doc_ids_a: set[str] | None = None,
        group_doc_ids_b: set[str] | None = None,
    ) -> ResolutionDecision:
        nf_a, nf_b = a.normalized_form, b.normalized_form
        shorter, longer = sorted([nf_a, nf_b], key=len)

        # Step 1: Numeric variant guard
        if self._is_numeric_variant(shorter, longer):
            return ResolutionDecision(
                decision="keep_distinct",
                strategy="numeric_variant_distinct",
            )

        # Step 2: Exact normalized match
        if nf_a == nf_b:
            return ResolutionDecision(
                decision="merge",
                strategy="exact_normalized_match",
            )

        # Step 3: Structural key match
        key_a = self._structural_key(nf_a)
        key_b = self._structural_key(nf_b)
        if key_a == key_b and len(key_a) >= 3:
            return ResolutionDecision(
                decision="merge",
                strategy="structural_key_match",
            )

        # Step 4: Co-document Levenshtein-1
        if self._edit_distance(nf_a, nf_b) == 1:
            if group_doc_ids_a and group_doc_ids_b and group_doc_ids_a & group_doc_ids_b:
                return ResolutionDecision(
                    decision="llm_candidate",
                    strategy="levenshtein_1_codoc",
                    similarity_score=0.85,
                )
            return ResolutionDecision(
                decision="keep_distinct",
                strategy="no_structural_match",
            )

        # Step 4b: Shared prefix (>=4) co-document
        if group_doc_ids_a and group_doc_ids_b and group_doc_ids_a & group_doc_ids_b:
            prefix_len = 0
            for ca, cb in zip(key_a, key_b):
                if ca == cb:
                    prefix_len += 1
                else:
                    break
            if prefix_len >= 4:
                return ResolutionDecision(
                    decision="llm_candidate",
                    strategy="shared_prefix_codoc",
                    similarity_score=0.75,
                )

        # Step 5: Default
        return ResolutionDecision(
            decision="keep_distinct",
            strategy="no_structural_match",
        )


class ComponentResolver(SemanticEntityResolver):
    entity_type = EntityType.COMPONENT


class HardwarePlatformResolver(SemanticEntityResolver):
    entity_type = EntityType.HARDWARE_PLATFORM


class EntityResolver:
    def __init__(self) -> None:
        self._resolvers: dict[EntityType, BaseEntityResolver] = {
            EntityType.PERSON: PersonResolver(),
            EntityType.PARAMETER: ParameterResolver(),
            EntityType.FIRMWARE_VERSION: FirmwareVersionResolver(),
            EntityType.COMPONENT: ComponentResolver(),
            EntityType.HARDWARE_PLATFORM: HardwarePlatformResolver(),
        }
        self._llm_resolver = MergeResolver()

    def resolve(
        self, mentions: list[EntityMention]
    ) -> tuple[list[CanonicalEntity], list[MergeRecord], list[MergeDecision]]:
        all_entities: list[CanonicalEntity] = []
        all_records: list[MergeRecord] = []
        all_decisions: list[MergeDecision] = []

        for entity_type, resolver in self._resolvers.items():
            type_mentions = [m for m in mentions if m.entity_type == entity_type]
            if not type_mentions:
                continue
            entities, records, decisions = resolver.resolve_group(
                type_mentions, llm_resolver=self._llm_resolver
            )
            all_entities.extend(entities)
            all_records.extend(records)
            all_decisions.extend(decisions)

        return all_entities, all_records, all_decisions


def resolve_aliases(
    entities: list[CanonicalEntity],
    mention_to_canonical: dict[str, str],
    alias_claims: list,  # list[ClaimMention] — import avoided for circular dep
) -> tuple[list[CanonicalEntity], dict[str, str], list[MergeRecord]]:
    """Merge canonical entities based on ALIAS_OF claims.

    For each ALIAS_OF claim:
      - subject_mention_id → canonical entity A (the canonical name)
      - object_freetext → lookup → canonical entity B (the alias)
      - If A != B: absorb B into A

    Returns updated (entities, mention_to_canonical, merge_records).
    """
    # Build lookup: normalized name/alias → canonical entity
    name_to_entity: dict[str, CanonicalEntity] = {}
    for e in entities:
        name_to_entity[e.canonical_name.lower().strip()] = e
        for alias in e.aliases:
            name_to_entity[alias.lower().strip()] = e

    # Build canonical_id → entity index for fast removal
    id_to_entity: dict[str, CanonicalEntity] = {e.canonical_id: e for e in entities}

    records: list[MergeRecord] = []
    absorbed: set[str] = set()  # canonical_ids that have been absorbed

    for claim in alias_claims:
        # Resolve subject to canonical entity A
        canon_id_a = mention_to_canonical.get(claim.subject_mention_id)
        if canon_id_a is None:
            continue
        entity_a = id_to_entity.get(canon_id_a)
        if entity_a is None or entity_a.canonical_id in absorbed:
            continue

        # Resolve object_freetext to canonical entity B
        alias_text = (claim.object_freetext or "").lower().strip()
        if not alias_text:
            continue
        entity_b = name_to_entity.get(alias_text)
        if entity_b is None or entity_b.canonical_id == entity_a.canonical_id:
            continue
        if entity_b.canonical_id in absorbed:
            continue

        # Absorb B into A
        entity_a.mention_ids.extend(entity_b.mention_ids)
        entity_a.aliases = sorted(set(entity_a.aliases) | set(entity_b.aliases) | {entity_b.canonical_name} - {entity_a.canonical_name})
        entity_a.first_seen = min(entity_a.first_seen, entity_b.first_seen)
        entity_a.last_seen = max(entity_a.last_seen, entity_b.last_seen)

        # Update mention_to_canonical for all of B's mentions
        for mid in entity_b.mention_ids:
            mention_to_canonical[mid] = entity_a.canonical_id

        # Update lookups
        name_to_entity[entity_b.canonical_name.lower().strip()] = entity_a
        for alias in entity_b.aliases:
            name_to_entity[alias.lower().strip()] = entity_a
        id_to_entity[entity_b.canonical_id] = entity_a  # redirect

        absorbed.add(entity_b.canonical_id)

        records.append(MergeRecord(
            canonical_id=entity_a.canonical_id,
            canonical_name=entity_a.canonical_name,
            merged_mention_ids=entity_b.mention_ids,
            strategy="corpus_alias_evidence",
            similarity_score=None,
            decided_at=datetime.now(),
        ))

    # Remove absorbed entities
    surviving = [e for e in entities if e.canonical_id not in absorbed]

    return surviving, mention_to_canonical, records
