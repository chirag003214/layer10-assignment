from __future__ import annotations

import dataclasses
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from extraction.schema import ClaimMention, EntityMention, EntityType, EvidenceRecord, Predicate
from monitoring.quality import generate_coherence_report
from resolution.claim_resolver import ClaimResolver
from resolution.entity_resolver import CanonicalEntity, EntityResolver, MergeRecord, resolve_aliases
from resolution.freetext_deduper import FreetextClaimDeduper
from resolution.merge_resolver import MergeDecision
from resolution.coherence_splitter import CoherenceSplitter
from resolution.temporal_resolver import SupersedingResolver, TemporalResolver


def _json_default(obj: object) -> object:
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, EntityType):
        return obj.value
    if isinstance(obj, Predicate):
        return obj.value
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _to_dict(dc: object) -> dict:
    return json.loads(json.dumps(dataclasses.asdict(dc), default=_json_default))


def run() -> None:
    mentions_path = Path("data/mentions/entity_mentions.jsonl")
    mentions: list[EntityMention] = []
    with mentions_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                mentions.append(EntityMention.model_validate_json(line))

    print(f"Loaded {len(mentions)} entity mentions")

    # ── Phase 1: Entity Resolution ───────────────────────────────────────────
    resolver = EntityResolver()
    entities, records, decisions = resolver.resolve(mentions)

    out_dir = Path("data/canonical")
    out_dir.mkdir(parents=True, exist_ok=True)

    # canonical_entities.json
    (out_dir / "canonical_entities.json").write_text(
        json.dumps([_to_dict(e) for e in entities], indent=2, default=_json_default)
    )

    # mention_to_canonical.json
    mention_to_canonical: dict[str, str] = {}
    for entity in entities:
        for mid in entity.mention_ids:
            mention_to_canonical[mid] = entity.canonical_id
    (out_dir / "mention_to_canonical.json").write_text(
        json.dumps(mention_to_canonical, indent=2)
    )

    # merge_log.jsonl
    with (out_dir / "merge_log.jsonl").open("w") as f:
        for record in records:
            f.write(json.dumps(_to_dict(record), default=_json_default) + "\n")

    # merge_decisions.jsonl — LLM decisions for the 75-89 similarity band
    with (out_dir / "merge_decisions.jsonl").open("w") as f:
        for decision in decisions:
            f.write(json.dumps(dataclasses.asdict(decision)) + "\n")

    # Summary
    type_counts = Counter(e.entity_type.value for e in entities)
    print("\n=== Summary ===")
    print(f"Canonical entities: {len(entities)}")
    for etype, count in sorted(type_counts.items()):
        print(f"  {etype}: {count}")
    print(f"Merge log entries:  {len(records)}")
    print(f"LLM merge decisions: {len(decisions)}")
    print(f"Original mentions:  {len(mentions)}")
    if mentions:
        print(f"Reduction ratio:    {1 - len(entities) / len(mentions):.1%}")

    # Verify all mention_ids appear in mention_to_canonical exactly once
    all_mid_set = {m.mention_id for m in mentions}
    mapped_mid_set = set(mention_to_canonical.keys())
    if all_mid_set != mapped_mid_set:
        missing = all_mid_set - mapped_mid_set
        extra = mapped_mid_set - all_mid_set
        raise AssertionError(
            f"mention_to_canonical mismatch! Missing: {missing}, Extra: {extra}"
        )
    print(f"\nVerification: all {len(all_mid_set)} mention_ids mapped ✓")

    # ── Phase 1b: Alias Resolution ───────────────────────────────────────────

    claims_path = Path("data/mentions/claim_mentions.jsonl")
    evidence_path = Path("data/mentions/evidence_records.jsonl")

    claim_mentions: list[ClaimMention] = []
    if claims_path.exists():
        with claims_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    claim_mentions.append(ClaimMention.model_validate_json(line))

    evidence_records: list[EvidenceRecord] = []
    if evidence_path.exists():
        with evidence_path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    evidence_records.append(EvidenceRecord.model_validate_json(line))

    print(f"\nLoaded {len(claim_mentions)} claim mentions, {len(evidence_records)} evidence records")

    # Alias resolution: merge entities linked by ALIAS_OF claims
    alias_claims = [c for c in claim_mentions if c.predicate == Predicate.ALIAS_OF]
    if alias_claims:
        entities, mention_to_canonical, alias_records = resolve_aliases(
            entities, mention_to_canonical, alias_claims
        )
        records.extend(alias_records)
        # Re-write canonical_entities.json and mention_to_canonical.json
        (out_dir / "canonical_entities.json").write_text(
            json.dumps([_to_dict(e) for e in entities], indent=2, default=_json_default)
        )
        (out_dir / "mention_to_canonical.json").write_text(
            json.dumps(mention_to_canonical, indent=2)
        )
        print(f"\n=== Phase 1b: Alias Resolution ===")
        print(f"ALIAS_OF claims:      {len(alias_claims)}")
        print(f"Alias merges:         {len(alias_records)}")
        print(f"Entities after alias: {len(entities)}")

    # Filter ALIAS_OF claims out before passing to ClaimResolver
    claim_mentions = [c for c in claim_mentions if c.predicate != Predicate.ALIAS_OF]

    # ── Phase 2: Claim Resolution ────────────────────────────────────────────

    # Build entity dict for temporal resolver
    entity_dict = {e.canonical_id: e for e in entities}

    # Claim resolution
    claim_resolver = ClaimResolver(claim_mentions, mention_to_canonical, evidence_records)
    canonical_claims, claim_skip_count = claim_resolver.resolve()

    # ── Phase 3: Freetext Claim Deduplication ─────────────────────────────────
    deduper = FreetextClaimDeduper()
    canonical_claims, freetext_merge_records = deduper.deduplicate(canonical_claims, entity_dict)

    # Write freetext_merge_log.jsonl
    with (out_dir / "freetext_merge_log.jsonl").open("w") as f:
        for rec in freetext_merge_records:
            f.write(json.dumps(rec.to_dict()) + "\n")

    merged_count = sum(1 for r in freetext_merge_records if r.strategy == "embedding_auto_merge")
    llm_merge = sum(1 for r in freetext_merge_records if r.strategy == "llm_claim_merge")
    llm_distinct = sum(1 for r in freetext_merge_records if r.strategy == "llm_claim_distinct")
    below_thresh = sum(1 for r in freetext_merge_records if r.strategy == "embedding_below_threshold")

    print("\n=== Phase 3: Freetext Claim Deduplication ===")
    print(f"Auto-merged pairs:      {merged_count}")
    print(f"LLM-merged pairs:       {llm_merge}")
    print(f"LLM kept distinct:      {llm_distinct}")
    print(f"Below threshold pairs:  {below_thresh}")
    print(f"Remaining claims:       {len(canonical_claims)}")

    # ── Phase 4: Temporal Resolution ─────────────────────────────────────────
    canonical_claims = TemporalResolver().resolve(canonical_claims, entity_dict)

    # ── Phase 5: Superseding Resolution ──────────────────────────────────────
    canonical_claims = SupersedingResolver().resolve(canonical_claims)

    # ── Phase 5b: Coherence Splitting ─────────────────────────────────────────
    splitter = CoherenceSplitter()
    split_decisions = splitter.find_splits(entities, canonical_claims)
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "split_candidates.json").write_text(
        json.dumps(
            [dataclasses.asdict(d) for d in split_decisions],
            indent=2,
            default=_json_default,
        )
    )
    print(f"\n=== Phase 5b: Coherence Splitting ===")
    print(f"Split candidates found: {len(split_decisions)}")

    # ── Write outputs ─────────────────────────────────────────────────────────

    # Write canonical_claims.json
    (out_dir / "canonical_claims.json").write_text(
        json.dumps(
            [_to_dict(c) for c in canonical_claims],
            indent=2,
            default=_json_default,
        )
    )

    # Write claim_merge_log.jsonl
    with (out_dir / "claim_merge_log.jsonl").open("w") as f:
        for cc in canonical_claims:
            if len(cc.provenance) > 1:
                entry = {
                    "canonical_claim_id": cc.canonical_claim_id,
                    "predicate": cc.predicate.value,
                    "merged_doc_ids": [p.doc_id for p in cc.provenance],
                    "evidence_count": len(cc.evidence_ids),
                }
                f.write(json.dumps(entry) + "\n")

    # Phase summary
    pred_counts = Counter(c.predicate.value for c in canonical_claims)
    historical = sum(1 for c in canonical_claims if not c.is_current)
    with_valid_from = sum(1 for c in canonical_claims if c.valid_from is not None)
    superseded = sum(1 for c in canonical_claims if c.superseded_by is not None)

    print("\n=== Phase 4-5: Claim Resolution Summary ===")
    print(f"Canonical claims:         {len(canonical_claims)}")
    print(f"Skipped claims:           {claim_skip_count}")
    for pred, count in sorted(pred_counts.items()):
        print(f"  {pred}: {count}")
    print(f"Historical (not current): {historical}")
    print(f"With valid_from:          {with_valid_from}")
    print(f"Superseded:               {superseded}")
    if claim_mentions:
        print(f"Claim reduction:          {1 - len(canonical_claims) / len(claim_mentions):.1%}")

    # ── Phase 6: Coherence Report ─────────────────────────────────────────────
    reports_dir = Path("data/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    coherence_path = reports_dir / "coherence_report.json"

    report = generate_coherence_report(
        str(out_dir / "canonical_entities.json"),
        str(out_dir / "canonical_claims.json"),
        str(evidence_path) if evidence_path.exists() else str(out_dir / "evidence_records.jsonl"),
        str(coherence_path),
    )

    warnings = [
        (eid, f["flag"])
        for eid, er in report["entities"].items()
        for f in er["flags"]
        if f["severity"] == "warning"
    ]
    print(f"\n=== Phase 6: Coherence Report ===")
    print(f"Report written to: {coherence_path}")
    print(f"Warning flags:     {len(warnings)}")
    for eid, flag in warnings[:10]:
        print(f"  [{flag}] {eid}")
    if len(warnings) > 10:
        print(f"  ... and {len(warnings) - 10} more")


if __name__ == "__main__":
    run()
