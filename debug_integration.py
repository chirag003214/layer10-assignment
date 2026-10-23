"""Debug script to trace LLM output at every extraction pass for the two flaky fixtures."""
from __future__ import annotations

import json
import logging
import os
import importlib
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv(override=True)

# Suppress litellm noise
for name in ("LiteLLM", "litellm", "httpx", "httpcore", "openai", "asyncio"):
    logging.getLogger(name).setLevel(logging.WARNING)

# Enable extraction module debug logging
logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
logging.getLogger("extraction").setLevel(logging.DEBUG)

import config
importlib.reload(config)

from extraction.extractor import Extractor, SCHEMA_VERSION, PROMPT_VERSION
from extraction.schema import IDGenerator, ExtractionMetadata
from extraction.prompts import (
    build_entity_user_prompt,
    build_evidence_user_prompt,
    build_claim_user_prompt,
)
from tests.integration.fixtures import fix_confirmation_doc, duplicate_issue_doc


def debug_doc(label, doc):
    ext = Extractor()

    print(f"\n{'='*80}")
    print(f"FIXTURE: {label}")
    print(f"doc_id={doc.doc_id}  title={doc.title}")
    print(f"body:\n{doc.body}")
    print(f"{'='*80}")

    id_gen = IDGenerator(doc.doc_id)
    metadata = ExtractionMetadata(
        schema_version=SCHEMA_VERSION,
        model=config.LLM_KWARGS["model"],
        prompt_version=PROMPT_VERSION,
        extracted_at=datetime.now(timezone.utc),
    )

    # --- PASS 1: Entities ---
    entity_prompt = build_entity_user_prompt(doc)
    print(f"\n--- PASS 1: ENTITY USER PROMPT ---")
    print(entity_prompt)

    blocks1 = ext._call_llm(entity_prompt, [ext.entity_tool], ext.entity_system_prompt)
    print(f"\n--- PASS 1: RAW LLM ENTITY BLOCKS ---")
    print(json.dumps(blocks1, indent=2, default=str))

    entities, local_id_map = ext._parse_entity_blocks(blocks1, doc, id_gen, metadata)
    print(f"\n--- PASS 1: PARSED + FILTERED ENTITIES ({len(entities)}) ---")
    for e in entities:
        print(f"  {e.mention_id}: {e.entity_type.name} \"{e.surface_form}\" attrs={e.attributes}")
    print(f"  local_id_map: {local_id_map}")

    if not entities:
        print("\n  ** NO ENTITIES AFTER FILTERING - pipeline returns empty **")
        return

    # --- PASS 2: Evidence ---
    evidence = ext._run_pass2_evidence(doc, id_gen, metadata)
    print(f"\n--- PASS 2: EVIDENCE ({len(evidence)}) ---")
    for ev in evidence:
        found = doc.body[ev.char_offset_start:ev.char_offset_end] if ev.char_offset_start >= 0 else "(not found)"
        match = "MATCH" if found == ev.excerpt else "MISMATCH"
        print(f"  {ev.evidence_id}: \"{ev.excerpt[:100]}\"")
        print(f"    offset=[{ev.char_offset_start}:{ev.char_offset_end}] {match}")

    if not evidence:
        print("\n  ** NO EVIDENCE - pipeline skips claims **")
        return

    # --- PASS 3: Claims ---
    claims = ext._run_pass3_claims(doc, entities, evidence, local_id_map, id_gen, metadata)
    print(f"\n--- PASS 3: CLAIMS ({len(claims)}) ---")
    for cm in claims:
        print(f"  {cm.claim_mention_id}: {cm.predicate.name}")
        print(f"    subject={cm.subject_mention_id} object={cm.object_mention_id}")
        print(f"    freetext={cm.object_freetext} confidence={cm.confidence}")
        print(f"    evidence_ids={cm.evidence_ids}")

    preds = {cm.predicate for cm in claims}
    print(f"\n  ALL PREDICATES: {preds}")


if __name__ == "__main__":
    debug_doc("FIX_CONFIRMATION (expects FIXED_IN)", fix_confirmation_doc())
    print("\n" * 3)
    debug_doc("DUPLICATE_ISSUE (expects DUPLICATE_OF)", duplicate_issue_doc())
