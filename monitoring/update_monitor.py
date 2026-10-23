from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_RAW_ROOT = Path("data/raw")
_DEFAULT_EVIDENCE_PATH = Path("data/extractions.jsonl")
_DEFAULT_CLAIMS_PATH = Path("data/canonical/canonical_claims.json")
_DEFAULT_SCAN_CACHE = Path("data/reports/update_monitor_cache.json")
_DEFAULT_REPORT_PATH = Path("data/reports/update_monitor_report.json")


@dataclass
class StatusResult:
    evidence_id: str
    old_status: str
    new_status: str
    doc_id: str
    changed: bool


class UpdateMonitor:
    def __init__(
        self,
        raw_root: Path = _DEFAULT_RAW_ROOT,
        evidence_path: Path = _DEFAULT_EVIDENCE_PATH,
        claims_path: Path = _DEFAULT_CLAIMS_PATH,
        scan_cache_path: Path = _DEFAULT_SCAN_CACHE,
        report_path: Path = _DEFAULT_REPORT_PATH,
    ) -> None:
        self.raw_root = Path(raw_root)
        self.evidence_path = Path(evidence_path)
        self.claims_path = Path(claims_path)
        self.scan_cache_path = Path(scan_cache_path)
        self.report_path = Path(report_path)

    def _resolve_doc_body(self, doc_id: str) -> str | None:
        """Resolve a doc_id to its current body text, or None if the file is missing."""
        # doc_id formats:
        #   ardupilot/ardupilot#{n}           → issues/{n}.json → raw["body"]
        #   ardupilot/ardupilot#{n}/c/{cid}   → comments/{n}.json list → item["id"] == cid
        #   ardupilot/ardupilot#{n}/t/{eid}   → timeline/{n}.json list → item["id"] == eid

        if "/c/" in doc_id:
            base, cid_str = doc_id.rsplit("/c/", 1)
            issue_num = base.split("#")[1]
            path = self.raw_root / "comments" / f"{issue_num}.json"
            if not path.exists():
                return None
            items: list[dict] = json.loads(path.read_text())
            target_id = int(cid_str)
            for item in items:
                if item.get("id") == target_id:
                    return item.get("body") or ""
            return None

        elif "/t/" in doc_id:
            base, eid_str = doc_id.rsplit("/t/", 1)
            issue_num = base.split("#")[1]
            path = self.raw_root / "timeline" / f"{issue_num}.json"
            if not path.exists():
                return None
            items = json.loads(path.read_text())
            target_id = int(eid_str)
            for item in items:
                if item.get("id") == target_id:
                    return item.get("body") or ""
            return None

        else:
            # plain issue
            issue_num = doc_id.split("#")[1]
            path = self.raw_root / "issues" / f"{issue_num}.json"
            if not path.exists():
                return None
            raw: dict = json.loads(path.read_text())
            return raw.get("body") or ""

    def scan_evidence(self, evidence_records: list[dict]) -> dict[str, StatusResult]:
        """Scan all evidence records and return status results keyed by evidence_id."""
        # Load scan cache
        cache: dict[str, str] = {}
        if self.scan_cache_path.exists():
            cache = json.loads(self.scan_cache_path.read_text())

        # Group evidence records by doc_id
        by_doc: dict[str, list[dict]] = {}
        for ev in evidence_records:
            doc_id = ev["doc_id"]
            by_doc.setdefault(doc_id, []).append(ev)

        results: dict[str, StatusResult] = {}

        for doc_id, records in by_doc.items():
            body = self._resolve_doc_body(doc_id)

            if body is None:
                # File missing → all records deleted
                for ev in records:
                    results[ev["evidence_id"]] = StatusResult(
                        evidence_id=ev["evidence_id"],
                        old_status=ev.get("source_status", "active"),
                        new_status="deleted",
                        doc_id=doc_id,
                        changed=ev.get("source_status", "active") != "deleted",
                    )
                cache.pop(doc_id, None)
                continue

            current_hash = hashlib.sha1(body.encode()).hexdigest()

            # Cache hit: hash unchanged → all records active
            if doc_id in cache and cache[doc_id] == current_hash:
                for ev in records:
                    old = ev.get("source_status", "active")
                    results[ev["evidence_id"]] = StatusResult(
                        evidence_id=ev["evidence_id"],
                        old_status=old,
                        new_status="active",
                        doc_id=doc_id,
                        changed=old != "active",
                    )
            else:
                # Hash changed (or not in cache) → check each record
                for ev in records:
                    old = ev.get("source_status", "active")
                    if current_hash == ev.get("source_hash", ""):
                        new_status = "active"
                    elif ev.get("excerpt", "") and ev["excerpt"] in body:
                        new_status = "edited_excerpt_intact"
                    else:
                        new_status = "edited_excerpt_removed"
                    results[ev["evidence_id"]] = StatusResult(
                        evidence_id=ev["evidence_id"],
                        old_status=old,
                        new_status=new_status,
                        doc_id=doc_id,
                        changed=old != new_status,
                    )
                cache[doc_id] = current_hash

        # Write updated cache
        self.scan_cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.scan_cache_path.write_text(json.dumps(cache, indent=2))

        return results

    def propagate_to_claims(
        self,
        status_results: dict[str, StatusResult],
        claims: list[dict],
    ) -> list[dict]:
        """Assign grounding_status to each claim based on its evidence statuses."""
        for claim in claims:
            evidence_ids: list[str] = claim.get("evidence_ids", [])
            statuses = [status_results[eid].new_status for eid in evidence_ids if eid in status_results]

            if not statuses:
                claim["grounding_status"] = "fully_grounded"
                continue

            active = [s for s in statuses if s == "active"]
            intact = [s for s in statuses if s == "edited_excerpt_intact"]
            removed = [s for s in statuses if s == "edited_excerpt_removed"]
            deleted = [s for s in statuses if s == "deleted"]

            if active:
                if len(active) == len(statuses):
                    claim["grounding_status"] = "fully_grounded"
                else:
                    claim["grounding_status"] = "partially_grounded"
            elif intact:
                claim["grounding_status"] = "edited_excerpt_intact"
            elif removed:
                claim["grounding_status"] = "unverifiable"
            elif deleted:
                claim["grounding_status"] = "ungrounded"
            else:
                claim["grounding_status"] = "fully_grounded"

        return claims

    def generate_report(
        self,
        status_results: dict[str, StatusResult],
        claims: list[dict],
    ) -> dict:
        """Build the update monitor report dict and write to report_path."""
        from collections import Counter

        status_counts: Counter[str] = Counter(r.new_status for r in status_results.values())
        changed_count = sum(1 for r in status_results.values() if r.changed)
        grounding_counts: Counter[str] = Counter(c.get("grounding_status", "fully_grounded") for c in claims)

        deleted_docs: list[str] = list({
            r.doc_id for r in status_results.values() if r.new_status == "deleted"
        })
        edited_docs: list[str] = list({
            r.doc_id
            for r in status_results.values()
            if r.new_status in ("edited_excerpt_intact", "edited_excerpt_removed")
        })

        report = {
            "summary": {
                "total_evidence_scanned": len(status_results),
                "changed_count": changed_count,
                "status_distribution": dict(status_counts),
                "grounding_status_distribution": dict(grounding_counts),
                "deleted_doc_count": len(deleted_docs),
                "edited_doc_count": len(edited_docs),
            },
            "deleted_docs": deleted_docs,
            "edited_docs": edited_docs,
        }

        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(report, indent=2))
        return report

    def run(self) -> None:
        """Full scan: load evidence, scan, propagate to claims, write report."""
        # Load evidence records
        evidence_records: list[dict] = []
        if self.evidence_path.exists():
            for line in self.evidence_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                # extractions.jsonl contains extraction records; evidence is nested
                evs = rec.get("evidence_records", [])
                if evs:
                    evidence_records.extend(evs)
                # Also handle flat evidence records (if stored directly)
                elif "evidence_id" in rec:
                    evidence_records.append(rec)

        logger.info("Loaded %d evidence records", len(evidence_records))

        status_results = self.scan_evidence(evidence_records)

        # Load canonical claims
        claims: list[dict] = []
        if self.claims_path.exists():
            claims = json.loads(self.claims_path.read_text())

        claims = self.propagate_to_claims(status_results, claims)

        # Write updated claims
        self.claims_path.parent.mkdir(parents=True, exist_ok=True)
        self.claims_path.write_text(json.dumps(claims, indent=2))

        report = self.generate_report(status_results, claims)

        # Print summary
        summary = report["summary"]
        print(f"\n=== UpdateMonitor Report ===")
        print(f"Evidence scanned : {summary['total_evidence_scanned']}")
        print(f"Changed          : {summary['changed_count']}")
        print(f"\nEvidence status distribution:")
        for status, count in sorted(summary["status_distribution"].items()):
            print(f"  {status:<30} {count}")
        print(f"\nClaim grounding_status distribution:")
        for status, count in sorted(summary["grounding_status_distribution"].items()):
            print(f"  {status:<30} {count}")
        print(f"\nDeleted docs: {summary['deleted_doc_count']}")
        print(f"Edited docs : {summary['edited_doc_count']}")
        print(f"\nReport written to: {self.report_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = UpdateMonitor()
    monitor.run()
