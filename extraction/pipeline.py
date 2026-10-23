from __future__ import annotations

import json
import logging
from pathlib import Path

from extraction.extractor import Extractor, SCHEMA_VERSION, PROMPT_VERSION
from extraction.schema import MentionExtractionResult
from extraction.structured_extractor import StructuredExtractor
from ingestion.preprocessor import RawDocument

logger = logging.getLogger(__name__)

INPUT_PATH = Path("data/processed/documents.jsonl")
MENTIONS_DIR = Path("data/mentions")
ENTITY_PATH = MENTIONS_DIR / "entity_mentions.jsonl"
CLAIM_PATH = MENTIONS_DIR / "claim_mentions.jsonl"
EVIDENCE_PATH = MENTIONS_DIR / "evidence_records.jsonl"


def _load_processed_ids(entity_path: Path) -> set[str]:
    processed: set[str] = set()
    if not entity_path.exists():
        return processed
    with entity_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                processed.add(json.loads(line)["doc_id"])
            except Exception:
                pass
    return processed


def run_extraction(
    input_path: Path = INPUT_PATH,
    mentions_dir: Path = MENTIONS_DIR,
    verbose: bool = True,
    limit: int | None = None,
) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    mentions_dir.mkdir(parents=True, exist_ok=True)
    entity_path = mentions_dir / "entity_mentions.jsonl"
    claim_path = mentions_dir / "claim_mentions.jsonl"
    evidence_path = mentions_dir / "evidence_records.jsonl"

    processed_ids = _load_processed_ids(entity_path)

    if processed_ids:
        print(f"Resuming — {len(processed_ids)} docs already processed")

    extractor = Extractor()
    structured_extractor = StructuredExtractor(schema_version=SCHEMA_VERSION, prompt_version=PROMPT_VERSION)
    new_count = 0
    skipped_count = 0
    error_count = 0
    total_entities = 0
    total_claims = 0
    total_evidence = 0
    llm_count = 0
    structured_count = 0

    with (
        entity_path.open("a", encoding="utf-8") as ef,
        claim_path.open("a", encoding="utf-8") as cf,
        evidence_path.open("a", encoding="utf-8") as evf,
        input_path.open("r", encoding="utf-8") as in_f,
    ):
        for line in in_f:
            line = line.strip()
            if not line:
                continue

            try:
                doc = RawDocument.model_validate_json(line)
            except Exception as exc:
                logger.error("Failed to parse document line: %s", exc)
                continue

            doc_id = doc.doc_id

            if doc_id in processed_ids:
                skipped_count += 1
                print(f"[{doc_id}] SKIPPED: already processed")
                continue

            if limit is not None and new_count >= limit:
                break

            try:
                if doc.doc_type == "timeline_event":
                    result: MentionExtractionResult = structured_extractor.extract(doc)
                    extractor_label = "structured"
                    structured_count += 1
                else:
                    result = extractor.extract(doc)
                    extractor_label = "llm"
                    llm_count += 1

                for mention in result.entity_mentions:
                    ef.write(json.dumps(mention.model_dump(mode="json")) + "\n")
                for record in result.evidence_records:
                    evf.write(json.dumps(record.model_dump(mode="json")) + "\n")
                for claim in result.claim_mentions:
                    cf.write(json.dumps(claim.model_dump(mode="json")) + "\n")

                ef.flush()
                cf.flush()
                evf.flush()

                n_entities = len(result.entity_mentions)
                n_evidence = len(result.evidence_records)
                n_claims = len(result.claim_mentions)

                total_entities += n_entities
                total_evidence += n_evidence
                total_claims += n_claims
                new_count += 1

                print(f"[{doc_id}] [{extractor_label}] entities: {n_entities} | evidence: {n_evidence} | claims: {n_claims}")

            except Exception as exc:
                print(f"[{doc_id}] ERROR: {exc}")
                error_count += 1

    print(f"Total docs processed: {new_count}")
    print(f"Total entity mentions: {total_entities}")
    print(f"Total evidence records: {total_evidence}")
    print(f"Total claim mentions: {total_claims}")
    print(f"Docs skipped (already processed): {skipped_count}")
    print(f"Docs with errors: {error_count}")
    print(f"Docs processed by LLMExtractor: {llm_count}")
    print(f"Docs processed by StructuredExtractor: {structured_count}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max number of docs to process")
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--mentions-dir", type=Path, default=MENTIONS_DIR)
    args = parser.parse_args()

    run_extraction(input_path=args.input, mentions_dir=args.mentions_dir, limit=args.limit)
