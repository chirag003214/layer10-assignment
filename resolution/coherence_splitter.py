from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from extraction.schema import Predicate
from resolution.claim_resolver import CanonicalClaim
from resolution.entity_resolver import CanonicalEntity
from resolution.freetext_deduper import Embedder

logger = logging.getLogger(__name__)

# Minimum number of claims for an entity to be considered for splitting
_MIN_CLAIMS_FOR_SPLIT = 5

# Cosine similarity threshold for connecting claims in the graph
_SIMILARITY_THRESHOLD = 0.3


@dataclass
class SplitDecision:
    canonical_id: str
    canonical_name: str
    proposed_group_a: list[str]  # claim IDs
    proposed_group_b: list[str]  # claim IDs
    group_a_sample: str
    group_b_sample: str
    confidence: float


class CoherenceSplitter:
    """Detects entities whose claims form disconnected semantic clusters.

    For each entity with >= _MIN_CLAIMS_FOR_SPLIT claims, embeds claim
    descriptions, builds a similarity graph, and checks for disconnected
    components. Entities with 2+ components are flagged as split candidates.
    """

    def __init__(self) -> None:
        self.embedder = Embedder()

    def find_splits(
        self,
        entities: list[CanonicalEntity],
        claims: list[CanonicalClaim],
    ) -> list[SplitDecision]:
        # Build entity_id → claims mapping
        entity_claims: dict[str, list[CanonicalClaim]] = {}
        for c in claims:
            entity_claims.setdefault(c.subject_canonical_id, []).append(c)

        entity_dict = {e.canonical_id: e for e in entities}
        decisions: list[SplitDecision] = []

        for eid, eclaims in entity_claims.items():
            if len(eclaims) < _MIN_CLAIMS_FOR_SPLIT:
                continue

            entity = entity_dict.get(eid)
            if entity is None:
                continue

            split = self._check_entity(entity, eclaims)
            if split is not None:
                decisions.append(split)

        return decisions

    def _claim_text(self, claim: CanonicalClaim) -> str:
        """Extract representative text from a claim for embedding."""
        parts = [claim.predicate.value]
        if claim.object_freetext:
            parts.append(claim.object_freetext)
        if claim.object_canonical_id:
            parts.append(claim.object_canonical_id)
        return " ".join(parts)

    def _check_entity(
        self,
        entity: CanonicalEntity,
        claims: list[CanonicalClaim],
    ) -> SplitDecision | None:
        texts = [self._claim_text(c) for c in claims]
        embeddings = self.embedder.embed(texts)
        sim_matrix = embeddings @ embeddings.T

        n = len(claims)

        # Build adjacency and find connected components via BFS
        adj: dict[int, list[int]] = {i: [] for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                if float(sim_matrix[i, j]) >= _SIMILARITY_THRESHOLD:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * n
        components: list[list[int]] = []

        for start in range(n):
            if visited[start]:
                continue
            component: list[int] = []
            queue = [start]
            visited[start] = True
            while queue:
                node = queue.pop(0)
                component.append(node)
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            components.append(component)

        if len(components) < 2:
            return None

        # Sort components by size descending, take the two largest
        components.sort(key=len, reverse=True)
        group_a = components[0]
        group_b = components[1]

        # Compute confidence based on how cleanly separated the groups are
        confidence = min(len(group_a), len(group_b)) / n

        return SplitDecision(
            canonical_id=entity.canonical_id,
            canonical_name=entity.canonical_name,
            proposed_group_a=[claims[i].canonical_claim_id for i in group_a],
            proposed_group_b=[claims[i].canonical_claim_id for i in group_b],
            group_a_sample=texts[group_a[0]],
            group_b_sample=texts[group_b[0]],
            confidence=confidence,
        )
