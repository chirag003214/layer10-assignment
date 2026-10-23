"""Export the memory graph as a self-contained JSON artifact.

Produces data/outputs/graph_export.json with entities, claims (evidence
embedded), and summary statistics.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANONICAL_DIR = DATA_DIR / "canonical"
MENTIONS_DIR = DATA_DIR / "mentions"
OUTPUT_DIR = DATA_DIR / "outputs"


def export_graph(output_path: Path | None = None) -> Path:
    """Build and write graph_export.json."""
    output_path = output_path or (OUTPUT_DIR / "graph_export.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # load data
    with open(CANONICAL_DIR / "canonical_entities.json") as f:
        entities = json.load(f)
    with open(CANONICAL_DIR / "canonical_claims.json") as f:
        claims = json.load(f)

    evidence_by_id: dict[str, dict] = {}
    ev_path = MENTIONS_DIR / "evidence_records.jsonl"
    if ev_path.exists():
        with open(ev_path) as f:
            for line in f:
                rec = json.loads(line)
                evidence_by_id[rec["evidence_id"]] = rec

    entities_by_id = {e["canonical_id"]: e for e in entities}

    # embed evidence into claims
    claims_with_evidence = []
    for c in claims:
        claim_out = dict(c)
        embedded_evidence = []
        for ev_id in c.get("evidence_ids", []):
            ev = evidence_by_id.get(ev_id)
            if ev:
                embedded_evidence.append(ev)
        claim_out["evidence"] = embedded_evidence

        # resolve entity names
        subj = entities_by_id.get(c.get("subject_canonical_id"), {})
        claim_out["subject_name"] = subj.get("canonical_name")
        claim_out["subject_type"] = subj.get("entity_type")
        if c.get("object_canonical_id"):
            obj = entities_by_id.get(c["object_canonical_id"], {})
            claim_out["object_name"] = obj.get("canonical_name")
            claim_out["object_type"] = obj.get("entity_type")

        claims_with_evidence.append(claim_out)

    # compute stats
    type_counts = Counter(e["entity_type"] for e in entities)
    pred_counts = Counter(c["predicate"] for c in claims)
    n_current = sum(1 for c in claims if c.get("is_current"))
    n_historical = len(claims) - n_current
    avg_confidence = (
        sum(c.get("confidence", 0) or 0 for c in claims) / len(claims)
        if claims
        else 0
    )

    export = {
        "entities": entities,
        "claims": claims_with_evidence,
        "stats": {
            "total_entities": len(entities),
            "total_claims": len(claims),
            "total_evidence": len(evidence_by_id),
            "entities_by_type": dict(type_counts),
            "claims_by_predicate": dict(pred_counts),
            "current_claims": n_current,
            "historical_claims": n_historical,
            "avg_confidence": round(avg_confidence, 3),
        },
    }

    with open(output_path, "w") as f:
        json.dump(export, f, indent=2, default=str)

    print(f"Exported graph to {output_path}")
    print(f"  Entities: {len(entities)}")
    print(f"  Claims: {len(claims)} ({n_current} current, {n_historical} historical)")
    print(f"  Evidence: {len(evidence_by_id)}")
    return output_path


if __name__ == "__main__":
    export_graph()
