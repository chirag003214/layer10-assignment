from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np

import config
from extraction.schema import Predicate
from resolution.claim_resolver import CanonicalClaim
from resolution.entity_resolver import CanonicalEntity

logger = logging.getLogger(__name__)

# Similarity thresholds
_AUTO_MERGE_THRESHOLD = 0.92
_LLM_LOWER_THRESHOLD = 0.75

# Predicates subject to freetext deduplication
_FREETEXT_DEDUP_PREDICATES = {Predicate.CAUSES, Predicate.REGRESSED_IN}


class FreetextMergeRecord:
    """Record of a freetext deduplication decision."""

    def __init__(
        self,
        *,
        canonical_claim_id: str,
        merged_claim_ids: list[str],
        subject_canonical_id: str,
        subject_name: str,
        predicate: str,
        similarity_score: float | None,
        strategy: str,
        llm_decision: str | None,
        llm_confidence: float | None,
        llm_reasoning: str | None,
        decided_at: datetime,
    ) -> None:
        self.canonical_claim_id = canonical_claim_id
        self.merged_claim_ids = merged_claim_ids
        self.subject_canonical_id = subject_canonical_id
        self.subject_name = subject_name
        self.predicate = predicate
        self.similarity_score = similarity_score
        self.strategy = strategy
        self.llm_decision = llm_decision
        self.llm_confidence = llm_confidence
        self.llm_reasoning = llm_reasoning
        self.decided_at = decided_at

    def to_dict(self) -> dict:
        return {
            "canonical_claim_id": self.canonical_claim_id,
            "merged_claim_ids": self.merged_claim_ids,
            "subject_canonical_id": self.subject_canonical_id,
            "subject_name": self.subject_name,
            "predicate": self.predicate,
            "similarity_score": self.similarity_score,
            "strategy": self.strategy,
            "llm_decision": self.llm_decision,
            "llm_confidence": self.llm_confidence,
            "llm_reasoning": self.llm_reasoning,
            "decided_at": self.decided_at.isoformat(),
        }


@dataclass
class _ClaimMergeResult:
    decision: Literal["merge", "keep_distinct"]
    confidence: float
    reasoning: str
    strategy: str


_CLAIM_MERGE_SYSTEM_PROMPT = """\
You are a knowledge graph deduplication engine for ArduPilot GitHub issues.
Your task: decide if two claims about the same component describe the same root cause or symptom.

MERGE if the claims are near-synonyms, describe the same underlying problem from different angles,
or use different wording for the same observable failure mode.

KEEP DISTINCT if the claims describe genuinely different symptoms, different failure modes,
or different conditions, even if they share some vocabulary.

Be conservative — when in doubt, keep distinct.\
"""


def _get_claim_merge_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "record_claim_merge_decision",
            "description": "Record whether these two claims should be merged.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["merge", "keep_distinct"],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence explaining the decision.",
                    },
                },
                "required": ["decision", "confidence", "reasoning"],
            },
        },
    }


class ClaimMergeResolver:
    def should_merge(
        self,
        claim_a: CanonicalClaim,
        claim_b: CanonicalClaim,
        subject_name: str,
    ) -> _ClaimMergeResult:
        import litellm

        user_prompt = (
            f"Component: {subject_name}\n\n"
            f'Claim A ({claim_a.predicate.value}): "{claim_a.object_freetext}"\n'
            f'Claim B ({claim_b.predicate.value}): "{claim_b.object_freetext}"\n\n'
            "Should these two claims be merged into one?"
        )

        try:
            response = litellm.completion(
                messages=[
                    {"role": "system", "content": _CLAIM_MERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                tools=[_get_claim_merge_tool()],
                tool_choice="required",
                **config.LLM_KWARGS,
            )

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("No tool calls in LLM response")

            args = json.loads(tool_calls[0].function.arguments)
            decision = args["decision"]
            confidence = float(args["confidence"])
            reasoning = args["reasoning"]

            if decision not in ("merge", "keep_distinct"):
                raise ValueError(f"Unexpected decision: {decision!r}")

            strategy = "llm_claim_merge" if decision == "merge" else "llm_claim_distinct"
            return _ClaimMergeResult(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                strategy=strategy,
            )

        except Exception as exc:
            logger.error(
                "ClaimMergeResolver LLM call failed for %s / %s: %s",
                claim_a.canonical_claim_id,
                claim_b.canonical_claim_id,
                exc,
            )
            return _ClaimMergeResult(
                decision="keep_distinct",
                confidence=0.0,
                reasoning="llm_error",
                strategy="llm_error",
            )


class Embedder:
    def __init__(self) -> None:
        self._local_model = None

    def embed(self, texts: list[str]) -> np.ndarray:
        if config.USE_LOCAL_EMBEDDINGS:
            return self._embed_local(texts)
        try:
            return self._embed_litellm(texts)
        except Exception as exc:
            logger.warning("LiteLLM embedding failed (%s), falling back to local model", exc)
            return self._embed_local(texts)

    def _embed_litellm(self, texts: list[str]) -> np.ndarray:
        import litellm

        # Strip 'model' from LLM_KWARGS since embedding uses its own model
        kwargs = {k: v for k, v in config.LLM_KWARGS.items() if k != "model"}
        response = litellm.embedding(
            model=config.EMBEDDING_MODEL,
            input=texts,
            **kwargs,
        )
        vectors = np.array([item["embedding"] for item in response.data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms

    def _embed_local(self, texts: list[str]) -> np.ndarray:
        if self._local_model is None:
            from sentence_transformers import SentenceTransformer

            self._local_model = SentenceTransformer(config.EMBEDDING_LOCAL_MODEL)

        vectors = self._local_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.array(vectors, dtype=np.float32)


class FreetextClaimDeduper:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.merge_resolver = ClaimMergeResolver()

    def deduplicate(
        self,
        claims: list[CanonicalClaim],
        canonical_entities: dict[str, CanonicalEntity],
    ) -> tuple[list[CanonicalClaim], list[FreetextMergeRecord]]:
        # Invariant check: warn if any CAUSES/REGRESSED_IN claim has >1 freetext_variants
        for c in claims:
            if c.predicate in _FREETEXT_DEDUP_PREDICATES and len(c.object_freetext_variants) > 1:
                logger.warning(
                    "Claim %s has %d object_freetext_variants — possible old grouping logic",
                    c.canonical_claim_id,
                    len(c.object_freetext_variants),
                )

        candidates = [
            c for c in claims
            if c.predicate in _FREETEXT_DEDUP_PREDICATES and c.object_freetext is not None
        ]
        passthrough = [c for c in claims if c not in candidates]

        records: list[FreetextMergeRecord] = []
        surviving: list[CanonicalClaim] = list(passthrough)

        # Group by subject_canonical_id
        by_subject: dict[str, list[CanonicalClaim]] = {}
        for c in candidates:
            by_subject.setdefault(c.subject_canonical_id, []).append(c)

        for subj_id, group in by_subject.items():
            if len(group) == 1:
                surviving.append(group[0])
                continue

            subject_name = ""
            entity = canonical_entities.get(subj_id)
            if entity is not None:
                subject_name = entity.canonical_name

            merged_group, group_records = self._deduplicate_group(
                group, subj_id, subject_name
            )
            surviving.extend(merged_group)
            records.extend(group_records)

        return surviving, records

    def _embed_group(self, group: list[CanonicalClaim], subject_name: str) -> np.ndarray:
        texts = [
            f"{subject_name} {c.predicate.value} {c.object_freetext}"
            for c in group
        ]
        return self.embedder.embed(texts)

    def _deduplicate_group(
        self,
        group: list[CanonicalClaim],
        subj_id: str,
        subject_name: str,
    ) -> tuple[list[CanonicalClaim], list[FreetextMergeRecord]]:
        embeddings = self._embed_group(group, subject_name)
        sim_matrix = embeddings @ embeddings.T

        n = len(group)
        records: list[FreetextMergeRecord] = []

        # Union-find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            for j in range(i + 1, n):
                score = float(sim_matrix[i, j])

                if score >= _AUTO_MERGE_THRESHOLD:
                    union(i, j)
                    records.append(
                        FreetextMergeRecord(
                            canonical_claim_id=group[i].canonical_claim_id,
                            merged_claim_ids=[group[i].canonical_claim_id, group[j].canonical_claim_id],
                            subject_canonical_id=subj_id,
                            subject_name=subject_name,
                            predicate=group[i].predicate.value,
                            similarity_score=score,
                            strategy="embedding_auto_merge",
                            llm_decision=None,
                            llm_confidence=None,
                            llm_reasoning=None,
                            decided_at=datetime.now(tz=timezone.utc),
                        )
                    )

                elif score >= _LLM_LOWER_THRESHOLD:
                    result = self.merge_resolver.should_merge(group[i], group[j], subject_name)
                    if result.decision == "merge":
                        union(i, j)
                    records.append(
                        FreetextMergeRecord(
                            canonical_claim_id=group[i].canonical_claim_id,
                            merged_claim_ids=[group[i].canonical_claim_id, group[j].canonical_claim_id],
                            subject_canonical_id=subj_id,
                            subject_name=subject_name,
                            predicate=group[i].predicate.value,
                            similarity_score=score,
                            strategy=result.strategy,
                            llm_decision=result.decision,
                            llm_confidence=result.confidence,
                            llm_reasoning=result.reasoning,
                            decided_at=datetime.now(tz=timezone.utc),
                        )
                    )

                else:
                    records.append(
                        FreetextMergeRecord(
                            canonical_claim_id=group[i].canonical_claim_id,
                            merged_claim_ids=[group[i].canonical_claim_id, group[j].canonical_claim_id],
                            subject_canonical_id=subj_id,
                            subject_name=subject_name,
                            predicate=group[i].predicate.value,
                            similarity_score=score,
                            strategy="embedding_below_threshold",
                            llm_decision=None,
                            llm_confidence=None,
                            llm_reasoning=None,
                            decided_at=datetime.now(tz=timezone.utc),
                        )
                    )

        # Build connected components
        components: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            components.setdefault(root, []).append(i)

        merged: list[CanonicalClaim] = []
        for indices in components.values():
            if len(indices) == 1:
                merged.append(group[indices[0]])
                continue

            members = [group[i] for i in indices]
            # Base = claim with most evidence_ids
            base = max(members, key=lambda c: len(c.evidence_ids))

            # Union evidence_ids (preserving order)
            seen_ev: set[str] = set()
            union_ev: list[str] = []
            for c in [base] + [m for m in members if m is not base]:
                for eid in c.evidence_ids:
                    if eid not in seen_ev:
                        union_ev.append(eid)
                        seen_ev.add(eid)

            # Union freetext_variants
            seen_ft: set[str] = set()
            union_ft: list[str] = []
            for c in members:
                for v in c.object_freetext_variants:
                    if v not in seen_ft:
                        union_ft.append(v)
                        seen_ft.add(v)

            # Longest freetext as canonical
            canonical_freetext = max(
                (c.object_freetext for c in members if c.object_freetext),
                key=len,
                default=base.object_freetext,
            )

            # Max confidence
            max_conf = max(c.confidence for c in members)

            # Sum mention counts, union provenance
            all_prov = []
            seen_prov: set[str] = set()
            for c in members:
                for p in c.provenance:
                    key = f"{p.doc_id}:{p.source_hash}"
                    if key not in seen_prov:
                        all_prov.append(p)
                        seen_prov.add(key)

            import dataclasses

            new_claim = dataclasses.replace(
                base,
                evidence_ids=union_ev,
                object_freetext=canonical_freetext,
                object_freetext_variants=union_ft,
                confidence=max_conf,
                provenance=all_prov,
            )
            merged.append(new_claim)

        return merged, records
