from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)


def generate_coherence_report(
    canonical_entities_path: str,
    canonical_claims_path: str,
    evidence_records_path: str,
    output_path: str,
) -> dict:
    """Generate a coherence report over resolved canonical data.

    Loads entities (JSON array), claims (JSON array), evidence (JSONL),
    computes summary statistics and per-entity flags, writes JSON to output_path.
    """
    # Load data
    entities: list[dict] = json.loads(Path(canonical_entities_path).read_text())
    claims: list[dict] = json.loads(Path(canonical_claims_path).read_text())

    evidence_records: list[dict] = []
    ep = Path(evidence_records_path)
    if ep.exists():
        for line in ep.read_text().splitlines():
            line = line.strip()
            if line:
                evidence_records.append(json.loads(line))

    # Build per-entity claim lookups
    as_subject: dict[str, list[dict]] = defaultdict(list)
    as_object: dict[str, list[dict]] = defaultdict(list)
    for c in claims:
        subj = c.get("subject_canonical_id")
        if subj:
            as_subject[subj].append(c)
        obj = c.get("object_canonical_id")
        if obj:
            as_object[obj].append(c)

    # Summary
    pred_dist = Counter(c.get("predicate", "") for c in claims)
    etype_dist = Counter(e.get("entity_type", "") for e in entities)
    conf_values = [c.get("confidence", 0.0) for c in claims if c.get("confidence") is not None]
    conf_distribution = {
        "mean": sum(conf_values) / len(conf_values) if conf_values else 0.0,
        "min": min(conf_values) if conf_values else 0.0,
        "max": max(conf_values) if conf_values else 0.0,
    }
    grounding_dist = Counter(c.get("grounding_status", "fully_grounded") for c in claims)

    # Per-entity reports
    entity_reports: dict[str, dict] = {}
    for e in entities:
        eid = e["canonical_id"]
        etype = e.get("entity_type", "")
        aliases = e.get("aliases", [])
        mention_ids = e.get("mention_ids", [])
        mention_count = len(mention_ids)

        subj_claims = as_subject.get(eid, [])
        obj_claims = as_object.get(eid, [])
        claim_count = len(subj_claims) + len(obj_claims)

        # mention_source_distribution — doc_ids from claim provenance
        doc_ids: list[str] = []
        for c in subj_claims:
            for prov in c.get("provenance", []):
                doc_ids.append(prov.get("doc_id", ""))
        mention_source_dist = dict(Counter(doc_ids))

        # Confidence of subject claims
        subj_confs = [c.get("confidence", 0.0) for c in subj_claims]
        mean_conf = sum(subj_confs) / len(subj_confs) if subj_confs else None

        # Generate flags
        flags: list[dict] = []

        if mention_count == 1:
            flags.append({"flag": "singleton_entity", "severity": "info", "detail": f"mention_count={mention_count}"})

        if etype == "component" and claim_count == 0:
            flags.append({"flag": "orphan_entity", "severity": "warning", "detail": "component with no claims"})

        if len(aliases) >= 4:
            flags.append({"flag": "high_alias_count", "severity": "info", "detail": f"aliases={len(aliases)}"})

        # mixed_predicate_component: COMPONENT with both CAUSES (is_current=True) and FIXED_IN claims
        if etype == "component":
            current_causes = [c for c in subj_claims if c.get("predicate") == "causes" and c.get("is_current", True)]
            fixed_in_claims = [c for c in subj_claims if c.get("predicate") == "fixed_in"]
            if current_causes and fixed_in_claims:
                flags.append({
                    "flag": "mixed_predicate_component",
                    "severity": "warning",
                    "detail": f"{len(current_causes)} current CAUSES + {len(fixed_in_claims)} FIXED_IN",
                })

        # low_confidence_claims
        if mean_conf is not None and mean_conf < 0.65:
            flags.append({
                "flag": "low_confidence_claims",
                "severity": "info",
                "detail": f"mean_confidence={mean_conf:.3f}",
            })

        entity_reports[eid] = {
            "canonical_id": eid,
            "canonical_name": e.get("canonical_name", ""),
            "entity_type": etype,
            "mention_count": mention_count,
            "claim_count": claim_count,
            "as_subject": len(subj_claims),
            "as_object": len(obj_claims),
            "mention_source_distribution": mention_source_dist,
            "claims": [c.get("canonical_claim_id") for c in subj_claims],
            "flags": flags,
        }

    report = {
        "summary": {
            "total_entities": len(entities),
            "total_claims": len(claims),
            "total_evidence": len(evidence_records),
            "predicate_distribution": dict(pred_dist),
            "entity_type_distribution": dict(etype_dist),
            "confidence_distribution": conf_distribution,
            "grounding_status_distribution": dict(grounding_dist),
        },
        "entities": entity_reports,
    }

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    return report
