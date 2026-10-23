"""
Quick debug script: prints structured extraction output for the first N docs (two-pass flow).
Usage: python debug_extraction.py [--n 5]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from extraction.extractor import Extractor
from ingestion.preprocessor import RawDocument

INPUT_PATH = Path("data/processed/documents.jsonl")


def main(n: int, doc_id: str | None = None) -> None:
    extractor = Extractor()
    docs_seen = 0

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            doc = RawDocument.model_validate_json(line)

            if doc_id is not None:
                if doc.doc_id != doc_id:
                    continue
            elif docs_seen >= n:
                break

            result = extractor.extract(doc)

            print(f"\n{'='*10} DOC: {result.doc_id} {'='*10}")
            print(f"BODY PREVIEW: {(doc.body or '')[:300]!r}")

            print("\n--- PASS 1: EntityMentions ---")
            if result.entity_mentions:
                for em in result.entity_mentions:
                    print(
                        f"{em.mention_id} | {em.entity_type.name} | "
                        f"surface: '{em.surface_form}' | attrs: {em.attributes}"
                    )
            else:
                print("(none)")

            print("\n--- PASS 2: EvidenceRecords ---")
            if result.evidence_records:
                for er in result.evidence_records:
                    print(f"{er.evidence_id} | significance: '{er.significance}' | excerpt: '{er.excerpt[:80]}...'")
            else:
                print("(none)")

            print("\n--- PASS 3: ClaimMentions ---")
            if result.claim_mentions:
                for cm in result.claim_mentions:
                    print(
                        f"{cm.claim_mention_id} | {cm.predicate.name} | "
                        f"subject: {cm.subject_mention_id} | "
                        f"object_id: {cm.object_mention_id} | "
                        f"object_text: '{cm.object_freetext}' | "
                        f"evidence: {cm.evidence_ids} | "
                        f"conf: {cm.confidence}"
                    )
            else:
                print("(none)")

            docs_seen += 1

    if doc_id is not None and docs_seen == 0:
        print(f"ERROR: doc_id {doc_id!r} not found in {INPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--doc-id", type=str, default=None,
                        help="Run extraction on a single specific document by ID")
    args = parser.parse_args()
    main(args.n, doc_id=args.doc_id)
