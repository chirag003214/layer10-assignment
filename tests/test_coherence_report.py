from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from monitoring.quality import generate_coherence_report


def _entity(
    canonical_id: str,
    entity_type: str,
    name: str,
    mention_count: int = 1,
    aliases: list[str] | None = None,
) -> dict:
    return {
        "canonical_id": canonical_id,
        "entity_type": entity_type,
        "canonical_name": name,
        "aliases": aliases or [],
        "attributes": {},
        "mention_ids": [f"m_{i}" for i in range(mention_count)],
        "first_seen": "2026-01-01T00:00:00+00:00",
        "last_seen": "2026-01-01T00:00:00+00:00",
    }


def _claim(
    claim_id: str,
    predicate: str,
    subject_id: str,
    object_id: str | None = None,
    object_freetext: str | None = None,
    is_current: bool = True,
    confidence: float = 0.9,
    provenance_doc_ids: list[str] | None = None,
) -> dict:
    provenance = [
        {
            "doc_id": doc_id,
            "source_hash": "abc",
            "predicate": predicate,
            "extraction_schema_version": "2.0.0",
            "extraction_prompt_version": "v1",
            "extracted_at": "2026-01-01T00:00:00+00:00",
        }
        for doc_id in (provenance_doc_ids or ["doc1"])
    ]
    return {
        "canonical_claim_id": claim_id,
        "predicate": predicate,
        "subject_canonical_id": subject_id,
        "object_canonical_id": object_id,
        "object_freetext": object_freetext,
        "object_freetext_variants": [],
        "evidence_ids": ["ev1"],
        "provenance": provenance,
        "confidence": confidence,
        "is_current": is_current,
        "valid_from": None,
        "valid_until": None,
        "assertion_time": None,
        "event_time": None,
        "superseded_by": None,
    }


def _evidence_line(evidence_id: str, doc_id: str = "doc1") -> str:
    return json.dumps({
        "evidence_id": evidence_id,
        "doc_id": doc_id,
        "source_hash": "abc",
        "excerpt": "test",
        "char_offset_start": 0,
        "char_offset_end": 10,
        "timestamp": "2026-01-01T00:00:00+00:00",
        "author": "user",
        "source_status": "active",
    })


def _write_files(tmp: Path, entities: list, claims: list, evidence_lines: list[str] | None = None) -> tuple[Path, Path, Path]:
    ent_path = tmp / "entities.json"
    claims_path = tmp / "claims.json"
    ev_path = tmp / "evidence.jsonl"
    ent_path.write_text(json.dumps(entities))
    claims_path.write_text(json.dumps(claims))
    ev_path.write_text("\n".join(evidence_lines or []))
    return ent_path, claims_path, ev_path


# ── Test 1: basic report structure and counts ──

def test_basic_report_structure(tmp_path):
    entities = [_entity("ce1", "component", "EKF")]
    claims = [_claim("cc1", "affects", "ce1")]
    ent_path, claims_path, ev_path = _write_files(tmp_path, entities, claims)
    output_path = tmp_path / "report.json"

    report = generate_coherence_report(
        str(ent_path), str(claims_path), str(ev_path), str(output_path)
    )

    assert "summary" in report
    assert "entities" in report
    assert report["summary"]["total_entities"] == 1
    assert report["summary"]["total_claims"] == 1
    assert output_path.exists()
    saved = json.loads(output_path.read_text())
    assert saved["summary"]["total_entities"] == 1


# ── Test 2: singleton_entity flag (info) for mention_count == 1 ──

def test_singleton_entity_flag(tmp_path):
    entities = [_entity("ce1", "component", "EKF", mention_count=1)]
    claims = [_claim("cc1", "causes", "ce1", object_freetext="crash")]
    ent_path, claims_path, ev_path = _write_files(tmp_path, entities, claims)

    report = generate_coherence_report(
        str(ent_path), str(claims_path), str(ev_path), str(tmp_path / "r.json")
    )

    entity_report = report["entities"]["ce1"]
    flag_types = [f["flag"] for f in entity_report["flags"]]
    assert "singleton_entity" in flag_types


# ── Test 3: orphan_entity warning for COMPONENT with zero claims ──

def test_orphan_entity_flag(tmp_path):
    entities = [
        _entity("ce1", "component", "GPS"),
        _entity("ce2", "component", "EKF"),
    ]
    claims = [_claim("cc1", "affects", "ce1")]  # ce2 has no claims
    ent_path, claims_path, ev_path = _write_files(tmp_path, entities, claims)

    report = generate_coherence_report(
        str(ent_path), str(claims_path), str(ev_path), str(tmp_path / "r.json")
    )

    # ce2 should have orphan_entity warning
    ce2_flags = [f["flag"] for f in report["entities"]["ce2"]["flags"]]
    assert "orphan_entity" in ce2_flags

    # ce1 should not have orphan_entity (has a claim)
    ce1_flags = [f["flag"] for f in report["entities"]["ce1"]["flags"]]
    assert "orphan_entity" not in ce1_flags


# ── Test 4: high_alias_count flag (info) for >= 4 aliases ──

def test_high_alias_count_flag(tmp_path):
    entities = [
        _entity("ce1", "component", "EKF", aliases=["EKF3", "EKF filter", "estimator", "nav filter"]),
    ]
    claims = []
    ent_path, claims_path, ev_path = _write_files(tmp_path, entities, claims)

    report = generate_coherence_report(
        str(ent_path), str(claims_path), str(ev_path), str(tmp_path / "r.json")
    )

    flags = [f["flag"] for f in report["entities"]["ce1"]["flags"]]
    assert "high_alias_count" in flags


# ── Test 5: predicate_distribution in summary ──

def test_predicate_distribution(tmp_path):
    entities = [
        _entity("ce1", "component", "EKF"),
        _entity("ce2", "firmware_version", "4.5.1"),
    ]
    claims = [
        _claim("cc1", "affects", "ce1"),
        _claim("cc2", "affects", "ce1"),
        _claim("cc3", "fixed_in", "ce1", object_id="ce2"),
    ]
    ent_path, claims_path, ev_path = _write_files(tmp_path, entities, claims)

    report = generate_coherence_report(
        str(ent_path), str(claims_path), str(ev_path), str(tmp_path / "r.json")
    )

    dist = report["summary"]["predicate_distribution"]
    assert dist.get("affects", 0) == 2
    assert dist.get("fixed_in", 0) == 1
