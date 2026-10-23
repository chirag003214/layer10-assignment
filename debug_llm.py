"""
Debug script: shows raw LLM prompts, raw tool responses, filtering decisions,
and final parsed output for each pass of the extraction pipeline.

Usage:
  python debug_llm.py               # first doc only
  python debug_llm.py --n 3         # first 3 docs
  python debug_llm.py --doc-id ardupilot/ardupilot#32351
  python debug_llm.py --doc-id ardupilot/ardupilot#32351 --pass claims
"""
from __future__ import annotations

import argparse
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import litellm
from pydantic import ValidationError

import config
from extraction.extractor import Extractor
from extraction.prompts import (
    CLAIM_SYSTEM_PROMPT,
    ENTITY_SYSTEM_PROMPT,
    EVIDENCE_SYSTEM_PROMPT,
    build_claim_user_prompt,
    build_entity_user_prompt,
    build_evidence_user_prompt,
)
from extraction.schema import (
    ClaimMention,
    EntityMention,
    EntityType,
    EvidenceRecord,
    ExtractionMetadata,
    IDGenerator,
    MentionExtractionResult,
    Predicate,
    SourceStatus,
)
from extraction.tools import get_claim_tool, get_entity_tool, get_evidence_tool
from ingestion.preprocessor import RawDocument

INPUT_PATH = Path("data/processed/documents.jsonl")
SEP = "─" * 80
THIN = "·" * 80


def hdr(title: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def section(title: str) -> None:
    print(f"\n  ── {title} ──")


def indent(text: str, prefix: str = "    ") -> str:
    return textwrap.indent(str(text), prefix)


def print_prompt(system: str, user: str) -> None:
    section("SYSTEM PROMPT")
    print(indent(system))
    section("USER PROMPT")
    print(indent(user))


def print_raw_blocks(blocks: list[dict]) -> None:
    section("RAW LLM TOOL RESPONSE")
    if not blocks:
        print("    (no tool calls returned)")
        return
    for b in blocks:
        print(f"    tool: {b['name']}")
        print(indent(json.dumps(b["input"], indent=2)))


class DebugExtractor(Extractor):
    """Extractor subclass that prints every LLM call and filtering decision."""

    def __init__(self, only_pass: str | None = None) -> None:
        super().__init__()
        self.only_pass = only_pass  # "entities" | "evidence" | "claims" | None

    # ------------------------------------------------------------------ #
    # Intercept _call_llm to capture prompts + raw responses              #
    # ------------------------------------------------------------------ #

    def _call_llm_verbose(
        self,
        pass_name: str,
        user_prompt: str,
        tools: list[dict],
        system_prompt: str,
    ) -> list[dict]:
        if self.only_pass and self.only_pass not in pass_name:
            # Still call but suppress printing
            return super()._call_llm(user_prompt, tools, system_prompt)

        hdr(f"PASS: {pass_name.upper()}")
        print_prompt(system_prompt, user_prompt)

        blocks = super()._call_llm(user_prompt, tools, system_prompt)

        print_raw_blocks(blocks)
        return blocks

    # ------------------------------------------------------------------ #
    # Override each pass to use verbose LLM call                          #
    # ------------------------------------------------------------------ #

    def _run_pass1_entities_verbose(
        self,
        doc: RawDocument,
        id_gen: IDGenerator,
        metadata: ExtractionMetadata,
    ) -> tuple[list[EntityMention], dict[str, str]]:
        user_prompt = build_entity_user_prompt(doc)
        blocks = self._call_llm_verbose(
            "entities", user_prompt, [self.entity_tool], self.entity_system_prompt
        )

        entity_mentions, local_id_map = self._parse_entity_blocks(blocks, doc, id_gen, metadata)

        # Show what survived filtering
        section("PARSED ENTITIES (after filter)")
        if entity_mentions:
            for em in entity_mentions:
                lid = next((k for k, v in local_id_map.items() if v == em.mention_id), "?")
                print(f"    {em.mention_id} [{lid}] {em.entity_type.name} '{em.surface_form}' attrs={em.attributes}")
        else:
            print("    (none)")

        # Show raw entities from LLM to reveal what was dropped
        raw_entities: list[dict] = []
        for b in blocks:
            if b.get("name") == "extract_entities":
                raw_entities = b.get("input", {}).get("entities", [])
                break

        dropped = [
            e for e in raw_entities
            if e.get("local_id") not in local_id_map
        ]
        if dropped:
            section("FILTERED OUT (entities)")
            for e in dropped:
                print(f"    local_id={e.get('local_id')} type={e.get('entity_type')} surface={e.get('surface_form')!r}")

        return entity_mentions, local_id_map

    def _run_pass2_evidence_verbose(
        self,
        doc: RawDocument,
        id_gen: IDGenerator,
        metadata: ExtractionMetadata,
    ) -> list[EvidenceRecord]:
        import hashlib

        source_hash = hashlib.sha1((doc.body or "").encode()).hexdigest()
        body = doc.body or ""
        try:
            timestamp = datetime.fromisoformat(doc.created_at)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)

        user_prompt = build_evidence_user_prompt(doc)
        blocks = self._call_llm_verbose(
            "evidence", user_prompt, [self.evidence_tool], self.evidence_system_prompt
        )

        raw_evidence: list[dict] = []
        for b in blocks:
            if b.get("name") == "extract_evidence":
                raw_evidence = b.get("input", {}).get("evidence", [])
                break

        evidence_records: list[EvidenceRecord] = []
        seen_excerpts: set[str] = set()

        section("EVIDENCE PARSE DETAIL")
        for item in raw_evidence:
            excerpt = item.get("excerpt", "")
            if not excerpt:
                print(f"    SKIP (empty excerpt)")
                continue
            if excerpt in seen_excerpts:
                print(f"    SKIP (duplicate): {excerpt[:60]!r}")
                continue
            seen_excerpts.add(excerpt)

            start = body.find(excerpt)
            if start == -1:
                print(f"    WARN (not found in body): {excerpt[:60]!r}")
                end = -1
            else:
                end = start + len(excerpt)

            try:
                ev = EvidenceRecord(
                    evidence_id=id_gen.evidence_id(),
                    doc_id=doc.doc_id,
                    source_hash=source_hash,
                    significance=item.get("significance"),
                    excerpt=excerpt,
                    char_offset_start=start,
                    char_offset_end=end,
                    timestamp=timestamp,
                    author=doc.author,
                    source_status=SourceStatus.ACTIVE,
                )
                evidence_records.append(ev)
                print(f"    OK  {ev.evidence_id} [{start}:{end}] {excerpt[:60]!r}")
            except ValidationError as exc:
                print(f"    VALIDATION FAIL: {exc}")

        return evidence_records

    def _run_pass3_claims_verbose(
        self,
        doc: RawDocument,
        entity_mentions: list[EntityMention],
        evidence_records: list[EvidenceRecord],
        local_id_map: dict[str, str],
        id_gen: IDGenerator,
        metadata: ExtractionMetadata,
    ) -> list[ClaimMention]:
        evidence_ids_set = {ev.evidence_id for ev in evidence_records}
        mention_id_to_entity = {e.mention_id: e for e in entity_mentions}

        local_id_pairs = [
            (lid, mention_id_to_entity[mid])
            for lid, mid in local_id_map.items()
            if mid in mention_id_to_entity
        ]
        entity_list_str = self._format_entity_list(local_id_pairs)
        evidence_list_str = self._format_evidence_list(evidence_records)

        user_prompt = build_claim_user_prompt(doc, entity_list_str, evidence_list_str)
        blocks = self._call_llm_verbose(
            "claims", user_prompt, [self.claim_tool], self.claim_system_prompt
        )

        raw_claims: list[dict] = []
        for b in blocks:
            if b.get("name") == "extract_claims":
                raw_claims = b.get("input", {}).get("claims", [])
                break

        claim_mentions: list[ClaimMention] = []

        section("CLAIM PARSE DETAIL")
        for raw in raw_claims:
            subject_local_id = raw.get("subject_local_id")
            subject_mention_id = local_id_map.get(subject_local_id or "")
            if subject_mention_id is None:
                print(f"    SKIP (unknown subject local_id={subject_local_id!r}): {raw}")
                continue

            subject_entity = mention_id_to_entity.get(subject_mention_id)
            subject_label = f"{subject_entity.surface_form!r}" if subject_entity else subject_mention_id

            object_local_id = raw.get("object_local_id")
            object_mention_id = local_id_map.get(object_local_id) if object_local_id else None
            if object_local_id and object_mention_id is None:
                print(f"    WARN (unknown object local_id={object_local_id!r}), setting None")

            object_freetext = raw.get("object_freetext") or None
            confidence = float(raw.get("confidence", 0.0))

            try:
                predicate = Predicate[raw["predicate"]]
            except KeyError:
                print(f"    SKIP (unknown predicate={raw.get('predicate')!r}): {raw}")
                continue

            raw_ev_ids = raw.get("evidence_ids") or []
            valid_ev_ids = [eid for eid in raw_ev_ids if eid in evidence_ids_set]
            invalid_ev_ids = [eid for eid in raw_ev_ids if eid not in evidence_ids_set]

            if not valid_ev_ids:
                print(f"    SKIP (no valid evidence IDs, got {raw_ev_ids!r}): predicate={predicate.name}")
                continue

            threshold_flag = " *** BELOW 0.5 THRESHOLD ***" if confidence < 0.5 else ""
            print(
                f"    conf={confidence:.2f}{threshold_flag} | {predicate.name} | "
                f"subj={subject_label} | "
                f"obj_id={object_mention_id} obj_text={object_freetext!r} | "
                f"evidence={valid_ev_ids}"
                + (f" | INVALID_EV={invalid_ev_ids}" if invalid_ev_ids else "")
            )

            try:
                cm = ClaimMention(
                    claim_mention_id=id_gen.claim_mention_id(),
                    doc_id=doc.doc_id,
                    predicate=predicate,
                    subject_mention_id=subject_mention_id,
                    object_mention_id=object_mention_id,
                    object_freetext=object_freetext,
                    evidence_ids=valid_ev_ids,
                    confidence=confidence,
                    extraction_metadata=metadata,
                )
                claim_mentions.append(cm)
            except ValidationError as exc:
                print(f"      VALIDATION FAIL: {exc}")

        return claim_mentions

    # ------------------------------------------------------------------ #
    # Main entry point                                                     #
    # ------------------------------------------------------------------ #

    def extract_verbose(self, doc: RawDocument) -> MentionExtractionResult:
        print(f"\n{'='*80}")
        print(f"  DOC: {doc.doc_id}")
        print(f"  Title: {doc.title or '(none)'}")
        print(f"  Author: {doc.author}  |  Type: {doc.doc_type}  |  State: {doc.state or '?'}")
        print(f"  Body ({len(doc.body or '')} chars):")
        print(indent(textwrap.shorten(doc.body or "(empty)", width=300, placeholder="…")))
        print(f"{'='*80}")

        id_gen = IDGenerator(doc.doc_id)
        metadata = ExtractionMetadata(
            schema_version="2.0.0",
            model=config.LLM_KWARGS["model"],
            prompt_version="v3",
            extracted_at=datetime.now(timezone.utc),
        )

        # Pass 1
        entity_mentions, local_id_map = self._run_pass1_entities_verbose(doc, id_gen, metadata)

        if not entity_mentions:
            print("\n  No entities extracted — skipping evidence + claims.")
            return MentionExtractionResult(
                doc_id=doc.doc_id,
                entity_mentions=[],
                claim_mentions=[],
                evidence_records=[],
                extraction_metadata=metadata,
            )

        # Pass 2
        evidence_records = self._run_pass2_evidence_verbose(doc, id_gen, metadata)

        if not evidence_records:
            print("\n  No evidence extracted — skipping claims.")
            return MentionExtractionResult(
                doc_id=doc.doc_id,
                entity_mentions=entity_mentions,
                claim_mentions=[],
                evidence_records=[],
                extraction_metadata=metadata,
            )

        # Pass 3
        claim_mentions = self._run_pass3_claims_verbose(
            doc, entity_mentions, evidence_records, local_id_map, id_gen, metadata
        )

        section("FINAL SUMMARY")
        print(f"    entities:  {len(entity_mentions)}")
        print(f"    evidence:  {len(evidence_records)}")
        print(f"    claims:    {len(claim_mentions)}  ({sum(1 for c in claim_mentions if c.confidence < 0.5)} below 0.5)")

        return MentionExtractionResult(
            doc_id=doc.doc_id,
            entity_mentions=entity_mentions,
            claim_mentions=claim_mentions,
            evidence_records=evidence_records,
            extraction_metadata=metadata,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug LLM extraction pass-by-pass")
    parser.add_argument("--n", type=int, default=1, help="Number of docs to process (default: 1)")
    parser.add_argument("--doc-id", type=str, default=None, help="Process a specific doc_id only")
    parser.add_argument(
        "--pass",
        dest="only_pass",
        choices=["entities", "evidence", "claims"],
        default=None,
        help="Only show verbose output for this pass",
    )
    args = parser.parse_args()

    extractor = DebugExtractor(only_pass=args.only_pass)
    docs_seen = 0

    with INPUT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            doc = RawDocument.model_validate_json(line)

            if args.doc_id and doc.doc_id != args.doc_id:
                continue

            extractor.extract_verbose(doc)
            docs_seen += 1

            if args.doc_id:
                break
            if docs_seen >= args.n:
                break

    if docs_seen == 0:
        print(f"No documents found matching doc_id={args.doc_id!r}")


if __name__ == "__main__":
    main()
