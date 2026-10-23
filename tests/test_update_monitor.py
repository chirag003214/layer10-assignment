from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from monitoring.update_monitor import StatusResult, UpdateMonitor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def _make_ev(evidence_id: str, doc_id: str, excerpt: str, body: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "doc_id": doc_id,
        "source_hash": _sha1(body),
        "excerpt": excerpt,
        "source_status": "active",
    }


def _make_monitor(tmp_path: Path, raw_root: Path | None = None) -> UpdateMonitor:
    return UpdateMonitor(
        raw_root=raw_root or tmp_path / "raw",
        evidence_path=tmp_path / "extractions.jsonl",
        claims_path=tmp_path / "canonical_claims.json",
        scan_cache_path=tmp_path / "cache.json",
        report_path=tmp_path / "report.json",
    )


def _write_issue(raw_root: Path, issue_num: int, body: str) -> None:
    issues_dir = raw_root / "issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    (issues_dir / f"{issue_num}.json").write_text(json.dumps({"body": body}))


# ---------------------------------------------------------------------------
# Test 1: Active document — hash matches → "active"
# ---------------------------------------------------------------------------

def test_active_document(tmp_path: Path) -> None:
    body = "This is the issue body."
    raw_root = tmp_path / "raw"
    _write_issue(raw_root, 42, body)

    ev = _make_ev("ev_001", "ardupilot/ardupilot#42", "issue body", body)
    monitor = _make_monitor(tmp_path, raw_root)
    results = monitor.scan_evidence([ev])

    assert results["ev_001"].new_status == "active"
    assert not results["ev_001"].changed


# ---------------------------------------------------------------------------
# Test 2: Edited document, excerpt still present → "edited_excerpt_intact"
# ---------------------------------------------------------------------------

def test_edited_excerpt_intact(tmp_path: Path) -> None:
    original_body = "Original content here."
    new_body = "Original content here. Extra sentence added."

    raw_root = tmp_path / "raw"
    _write_issue(raw_root, 10, new_body)

    ev = _make_ev("ev_002", "ardupilot/ardupilot#10", "Original content here.", original_body)
    monitor = _make_monitor(tmp_path, raw_root)
    results = monitor.scan_evidence([ev])

    assert results["ev_002"].new_status == "edited_excerpt_intact"
    assert results["ev_002"].changed


# ---------------------------------------------------------------------------
# Test 3: Edited document, excerpt removed → "edited_excerpt_removed"
# ---------------------------------------------------------------------------

def test_edited_excerpt_removed(tmp_path: Path) -> None:
    original_body = "Buggy behavior observed."
    new_body = "Completely different content now."

    raw_root = tmp_path / "raw"
    _write_issue(raw_root, 7, new_body)

    ev = _make_ev("ev_003", "ardupilot/ardupilot#7", "Buggy behavior observed.", original_body)
    monitor = _make_monitor(tmp_path, raw_root)
    results = monitor.scan_evidence([ev])

    assert results["ev_003"].new_status == "edited_excerpt_removed"


# ---------------------------------------------------------------------------
# Test 4: Missing file → "deleted"
# ---------------------------------------------------------------------------

def test_missing_file_deleted(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "issues").mkdir(parents=True, exist_ok=True)
    # Do NOT write issue 99

    ev = _make_ev("ev_004", "ardupilot/ardupilot#99", "some excerpt", "some body")
    monitor = _make_monitor(tmp_path, raw_root)
    results = monitor.scan_evidence([ev])

    assert results["ev_004"].new_status == "deleted"


# ---------------------------------------------------------------------------
# Test 5: All active evidence → claim "fully_grounded"
# ---------------------------------------------------------------------------

def test_all_active_fully_grounded(tmp_path: Path) -> None:
    status_results = {
        "ev_a": StatusResult("ev_a", "active", "active", "doc#1", False),
        "ev_b": StatusResult("ev_b", "active", "active", "doc#1", False),
    }
    claims = [{"canonical_claim_id": "cc_001", "evidence_ids": ["ev_a", "ev_b"]}]
    monitor = _make_monitor(tmp_path)
    updated = monitor.propagate_to_claims(status_results, claims)
    assert updated[0]["grounding_status"] == "fully_grounded"


# ---------------------------------------------------------------------------
# Test 6: Mixed active/deleted → "partially_grounded"
# ---------------------------------------------------------------------------

def test_mixed_active_deleted_partially_grounded(tmp_path: Path) -> None:
    status_results = {
        "ev_a": StatusResult("ev_a", "active", "active", "doc#1", False),
        "ev_b": StatusResult("ev_b", "active", "deleted", "doc#2", True),
    }
    claims = [{"canonical_claim_id": "cc_002", "evidence_ids": ["ev_a", "ev_b"]}]
    monitor = _make_monitor(tmp_path)
    updated = monitor.propagate_to_claims(status_results, claims)
    assert updated[0]["grounding_status"] == "partially_grounded"


# ---------------------------------------------------------------------------
# Test 7: All deleted → "ungrounded"
# ---------------------------------------------------------------------------

def test_all_deleted_ungrounded(tmp_path: Path) -> None:
    status_results = {
        "ev_a": StatusResult("ev_a", "active", "deleted", "doc#1", True),
        "ev_b": StatusResult("ev_b", "active", "deleted", "doc#2", True),
    }
    claims = [{"canonical_claim_id": "cc_003", "evidence_ids": ["ev_a", "ev_b"]}]
    monitor = _make_monitor(tmp_path)
    updated = monitor.propagate_to_claims(status_results, claims)
    assert updated[0]["grounding_status"] == "ungrounded"


# ---------------------------------------------------------------------------
# Test 8: All edited_excerpt_removed → "unverifiable"
# ---------------------------------------------------------------------------

def test_all_removed_unverifiable(tmp_path: Path) -> None:
    status_results = {
        "ev_a": StatusResult("ev_a", "active", "edited_excerpt_removed", "doc#1", True),
    }
    claims = [{"canonical_claim_id": "cc_004", "evidence_ids": ["ev_a"]}]
    monitor = _make_monitor(tmp_path)
    updated = monitor.propagate_to_claims(status_results, claims)
    assert updated[0]["grounding_status"] == "unverifiable"


# ---------------------------------------------------------------------------
# Test 9: Cache skip — second run does not re-hash unchanged docs
# ---------------------------------------------------------------------------

def test_cache_skip(tmp_path: Path) -> None:
    body = "Stable content."
    raw_root = tmp_path / "raw"
    _write_issue(raw_root, 5, body)

    ev = _make_ev("ev_009", "ardupilot/ardupilot#5", "Stable content.", body)
    monitor = _make_monitor(tmp_path, raw_root)

    # First run: no cache → builds cache
    results1 = monitor.scan_evidence([ev])
    assert results1["ev_009"].new_status == "active"

    # Verify cache was written with the correct hash
    cache = json.loads((tmp_path / "cache.json").read_text())
    assert "ardupilot/ardupilot#5" in cache
    assert cache["ardupilot/ardupilot#5"] == _sha1(body)

    # Second run: cache hit — result should still be active
    results2 = monitor.scan_evidence([ev])
    assert results2["ev_009"].new_status == "active"


# ---------------------------------------------------------------------------
# Test 10: Idempotency — two consecutive runs produce identical results
# ---------------------------------------------------------------------------

def test_idempotency(tmp_path: Path) -> None:
    body = "Some issue body text."
    raw_root = tmp_path / "raw"
    _write_issue(raw_root, 20, body)

    ev = _make_ev("ev_010", "ardupilot/ardupilot#20", "issue body", body)
    monitor = _make_monitor(tmp_path, raw_root)

    results1 = monitor.scan_evidence([ev])
    results2 = monitor.scan_evidence([ev])

    assert results1["ev_010"].new_status == results2["ev_010"].new_status
    assert results1["ev_010"].changed == results2["ev_010"].changed


# ---------------------------------------------------------------------------
# Test 11: Report JSON has all required fields
# ---------------------------------------------------------------------------

def test_report_has_required_fields(tmp_path: Path) -> None:
    status_results = {
        "ev_a": StatusResult("ev_a", "active", "active", "doc#1", False),
    }
    claims = [{"canonical_claim_id": "cc_x", "evidence_ids": ["ev_a"], "grounding_status": "fully_grounded"}]
    monitor = _make_monitor(tmp_path)
    report = monitor.generate_report(status_results, claims)

    summary = report["summary"]
    assert "total_evidence_scanned" in summary
    assert "changed_count" in summary
    assert "status_distribution" in summary
    assert "grounding_status_distribution" in summary
    assert "deleted_doc_count" in summary
    assert "edited_doc_count" in summary
    assert "deleted_docs" in report
    assert "edited_docs" in report

    # Report file written
    assert (tmp_path / "report.json").exists()
    written = json.loads((tmp_path / "report.json").read_text())
    assert written["summary"]["total_evidence_scanned"] == 1


# ---------------------------------------------------------------------------
# Test 12: Claims file written with grounding_status on all claims
# ---------------------------------------------------------------------------

def test_claims_file_written_with_grounding_status(tmp_path: Path) -> None:
    body = "Firmware crash on boot."
    raw_root = tmp_path / "raw"
    _write_issue(raw_root, 3, body)

    ev = _make_ev("ev_012", "ardupilot/ardupilot#3", "Firmware crash", body)

    claims = [
        {"canonical_claim_id": "cc_aa", "evidence_ids": ["ev_012"]},
        {"canonical_claim_id": "cc_bb", "evidence_ids": []},
    ]
    claims_path = tmp_path / "canonical_claims.json"
    claims_path.write_text(json.dumps(claims))

    monitor = UpdateMonitor(
        raw_root=raw_root,
        evidence_path=tmp_path / "extractions.jsonl",
        claims_path=claims_path,
        scan_cache_path=tmp_path / "cache.json",
        report_path=tmp_path / "report.json",
    )

    evidence_records = [ev]
    status_results = monitor.scan_evidence(evidence_records)
    updated_claims = monitor.propagate_to_claims(status_results, claims)

    claims_path.write_text(json.dumps(updated_claims))

    written = json.loads(claims_path.read_text())
    assert all("grounding_status" in c for c in written), "All claims must have grounding_status"
    assert written[0]["grounding_status"] == "fully_grounded"
    assert written[1]["grounding_status"] == "fully_grounded"
