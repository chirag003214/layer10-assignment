from __future__ import annotations

from pathlib import Path

from extraction.schema import ClaimMention, EvidenceRecord


def get_stale_claims(
    claims_path: Path,
    current_schema_version: str,
) -> list[str]:
    """
    Returns deduplicated list of doc_ids where a claim's
    extraction_metadata.schema_version != current_schema_version.
    """
    stale: set[str] = set()
    with claims_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            claim = ClaimMention.model_validate_json(line)
            if claim.extraction_metadata.schema_version != current_schema_version:
                stale.add(claim.doc_id)
    return list(stale)


def get_ungrounded_claims(
    claims_path: Path,
    evidence_path: Path,
) -> list[dict]:
    """
    Returns list of dicts {doc_id, predicate, evidence_excerpt}
    where any referenced EvidenceRecord.source_status is "edited" or "deleted".
    """
    evidence_map: dict[str, EvidenceRecord] = {}
    with evidence_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ev = EvidenceRecord.model_validate_json(line)
            evidence_map[ev.evidence_id] = ev

    ungrounded: list[dict] = []
    with claims_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            claim = ClaimMention.model_validate_json(line)
            for eid in claim.evidence_ids:
                ev = evidence_map.get(eid)
                if ev and ev.source_status in ("edited", "deleted"):
                    ungrounded.append({
                        "doc_id": claim.doc_id,
                        "predicate": claim.predicate.value,
                        "evidence_excerpt": ev.excerpt,
                    })
    return ungrounded
