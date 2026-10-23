from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone

import litellm
from pydantic import ValidationError

import config
from extraction.prompts import (
    ENTITY_SYSTEM_PROMPT,
    EVIDENCE_SYSTEM_PROMPT,
    CLAIM_SYSTEM_PROMPT,
    build_entity_user_prompt,
    build_evidence_user_prompt,
    build_claim_user_prompt,
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
from extraction.tools import get_entity_tool, get_evidence_tool, get_claim_tool
from ingestion.preprocessor import RawDocument

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0.0"
PROMPT_VERSION = "v3"

# ---------------------------------------------------------------------------
# Per-predicate allowed subject entity types
# ---------------------------------------------------------------------------

_CAUSAL_SUBJECT_TYPES = frozenset({
    EntityType.COMPONENT,
    EntityType.HARDWARE_PLATFORM,
    EntityType.FIRMWARE_VERSION,
})
_STRUCTURAL_SUBJECT_TYPES = frozenset({
    EntityType.COMPONENT,
    EntityType.HARDWARE_PLATFORM,
})
_PREDICATE_SUBJECT_TYPES: dict[Predicate, frozenset[EntityType]] = {
    Predicate.CAUSES:       _CAUSAL_SUBJECT_TYPES,
    Predicate.AFFECTS:      _CAUSAL_SUBJECT_TYPES,
    Predicate.REGRESSED_IN: _CAUSAL_SUBJECT_TYPES,
    Predicate.FIXED_IN:     _STRUCTURAL_SUBJECT_TYPES,
    Predicate.HAS_PARAMETER: _STRUCTURAL_SUBJECT_TYPES,
    Predicate.DUPLICATE_OF: frozenset({EntityType.PERSON}),
    Predicate.ALIAS_OF:     _STRUCTURAL_SUBJECT_TYPES,
}


class Extractor:
    def __init__(self) -> None:
        self.entity_tool = get_entity_tool()
        self.evidence_tool = get_evidence_tool()
        self.claim_tool = get_claim_tool()
        self.entity_system_prompt = ENTITY_SYSTEM_PROMPT
        self.evidence_system_prompt = EVIDENCE_SYSTEM_PROMPT
        self.claim_system_prompt = CLAIM_SYSTEM_PROMPT
        self.llm_kwargs = config.LLM_KWARGS

    def extract(self, doc: RawDocument) -> MentionExtractionResult:
        return self._with_retry(doc)

    def _call_llm(
        self, user_prompt: str, tools: list[dict], system_prompt: str
    ) -> list[dict]:
        response = litellm.completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=tools,
            tool_choice="required",
            **self.llm_kwargs,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or []
        blocks = []
        for tc in tool_calls:
            blocks.append({
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments),
            })
        return blocks

    def _parse_entity_blocks(
        self,
        blocks: list[dict],
        doc: RawDocument,
        id_gen: IDGenerator,
        metadata: ExtractionMetadata,
    ) -> tuple[list[EntityMention], dict[str, str]]:
        raw_entities: list[dict] = []
        for block in blocks:
            if block.get("name") == "extract_entities":
                raw_entities = block.get("input", {}).get("entities", [])
                break

        entity_mentions: list[EntityMention] = []
        local_id_map: dict[str, str] = {}  # local_id -> mention_id

        for raw in raw_entities:
            try:
                entity_type = EntityType[raw["entity_type"]]
            except KeyError:
                logger.warning("Unknown entity_type %r, skipping", raw.get("entity_type"))
                continue

            surface_form = raw.get("surface_form", "")
            if not surface_form:
                logger.warning("Entity missing surface_form, skipping")
                continue

            local_id = raw.get("local_id", "")
            attributes = raw.get("attributes") or {}
            context_sentence = raw.get("context_sentence") or None

            mention_id = id_gen.mention_id()
            try:
                em = EntityMention(
                    mention_id=mention_id,
                    doc_id=doc.doc_id,
                    entity_type=entity_type,
                    surface_form=surface_form,
                    normalized_form=surface_form.lower().strip(),
                    attributes=attributes,
                    extraction_metadata=metadata,
                    context_sentence=context_sentence,
                )
            except ValidationError as exc:
                logger.warning("EntityMention validation failed: %s", exc)
                continue

            entity_mentions.append(em)
            if local_id:
                local_id_map[local_id] = mention_id

        entity_mentions, local_id_map = self._filter_entities(entity_mentions, local_id_map)
        return entity_mentions, local_id_map

    def _mark_document_author(
        self, entity_mentions: list[EntityMention], doc: RawDocument
    ) -> None:
        """Set is_document_author=True on the PERSON entity matching doc.author."""
        author_lower = (doc.author or "").lower()
        if not author_lower:
            return
        for em in entity_mentions:
            if em.entity_type != EntityType.PERSON:
                continue
            handle = (em.attributes.get("github_handle") or "").lower()
            if handle == author_lower or em.surface_form.lower() == author_lower:
                em.attributes["is_document_author"] = True
                return

    def _filter_entities(
        self,
        entities: list[EntityMention],
        local_id_map: dict[str, str],
    ) -> tuple[list[EntityMention], dict[str, str]]:
        kept: list[EntityMention] = []
        dropped_mention_ids: set[str] = set()
        dropped_info: list[tuple[str, str]] = []

        for e in entities:
            if e.entity_type == EntityType.FIRMWARE_VERSION:
                version_string = e.attributes.get("version_string", "")
                if re.fullmatch(r'[0-9a-f]{40}', version_string.strip()):
                    dropped_mention_ids.add(e.mention_id)
                    dropped_info.append((version_string, "git-sha firmware_version"))
                    continue
            elif e.entity_type == EntityType.PARAMETER:
                if "." in e.surface_form:
                    dropped_mention_ids.add(e.mention_id)
                    dropped_info.append((e.surface_form, "dot in parameter name"))
                    continue
                if not re.fullmatch(r'[A-Z][A-Z0-9_]*', e.surface_form):
                    dropped_mention_ids.add(e.mention_id)
                    dropped_info.append((e.surface_form, "parameter name not ALL_CAPS"))
                    continue
            kept.append(e)

        if dropped_info:
            logger.debug(
                "Filtered %d entit%s: %s",
                len(dropped_info),
                "y" if len(dropped_info) == 1 else "ies",
                "; ".join(f"'{n}' ({r})" for n, r in dropped_info),
            )

        updated_map = {
            lid: mid for lid, mid in local_id_map.items()
            if mid not in dropped_mention_ids
        }
        return kept, updated_map

    def _format_entity_list(self, pairs: list[tuple[str, EntityMention]]) -> str:
        lines = []
        for local_id, em in pairs:
            lines.append(f"- {local_id}: {em.entity_type.name} '{em.surface_form}'")
        return "\n".join(lines) if lines else "(none)"

    def _format_evidence_list(self, evidence_records: list[EvidenceRecord]) -> str:
        lines = [f"- {ev.evidence_id}: '{ev.excerpt}'" for ev in evidence_records]
        return "\n".join(lines) if lines else "(none)"

    def _run_pass2_evidence(
        self,
        doc: RawDocument,
        id_gen: IDGenerator,
        metadata: ExtractionMetadata,
    ) -> list[EvidenceRecord]:
        source_hash = hashlib.sha1((doc.body or "").encode()).hexdigest()
        body = doc.body or ""

        try:
            timestamp = datetime.fromisoformat(doc.created_at)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)

        evidence_user_prompt = build_evidence_user_prompt(doc)
        max_retries = 2
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                blocks = self._call_llm(
                    evidence_user_prompt,
                    tools=[self.evidence_tool],
                    system_prompt=self.evidence_system_prompt,
                )
                raw_evidence: list[dict] = []
                for block in blocks:
                    if block.get("name") == "extract_evidence":
                        raw_evidence = block.get("input", {}).get("evidence", [])
                        break

                evidence_records: list[EvidenceRecord] = []
                seen_excerpts: set[str] = set()

                for item in raw_evidence:
                    excerpt = item.get("excerpt", "")
                    if not excerpt or excerpt in seen_excerpts:
                        continue
                    seen_excerpts.add(excerpt)

                    significance = item.get("significance")

                    start = body.find(excerpt)
                    if start == -1:
                        logger.warning(
                            "[%s] Excerpt not found in body: %r", doc.doc_id, excerpt[:60]
                        )
                        end = -1
                    else:
                        end = start + len(excerpt)

                    try:
                        ev = EvidenceRecord(
                            evidence_id=id_gen.evidence_id(),
                            doc_id=doc.doc_id,
                            source_hash=source_hash,
                            significance=significance,
                            excerpt=excerpt,
                            char_offset_start=start,
                            char_offset_end=end,
                            timestamp=timestamp,
                            author=doc.author,
                            source_status=SourceStatus.ACTIVE,
                        )
                    except ValidationError as exc:
                        logger.warning("EvidenceRecord validation failed: %s", exc)
                        continue

                    evidence_records.append(ev)

                last_exc = None
                return evidence_records

            except (ValidationError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Pass 2 attempt %d/%d failed for %s: %s",
                    attempt + 1, max_retries + 1, doc.doc_id, exc,
                )

        logger.error("Pass 2 all retries exhausted for %s: %s", doc.doc_id, last_exc)
        return []

    def _run_pass3_claims(
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

        claim_user_prompt = build_claim_user_prompt(doc, entity_list_str, evidence_list_str)
        max_retries = 2
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                blocks = self._call_llm(
                    claim_user_prompt,
                    tools=[self.claim_tool],
                    system_prompt=self.claim_system_prompt,
                )
                raw_claims: list[dict] = []
                for block in blocks:
                    if block.get("name") == "extract_claims":
                        raw_claims = block.get("input", {}).get("claims", [])
                        break

                claim_mentions: list[ClaimMention] = []

                for raw in raw_claims:
                    subject_local_id = raw.get("subject_local_id")
                    subject_mention_id = local_id_map.get(subject_local_id or "")
                    if subject_mention_id is None:
                        logger.warning(
                            "[%s] Claim references unknown subject local_id %r, skipping",
                            doc.doc_id, subject_local_id,
                        )
                        continue

                    object_local_id = raw.get("object_local_id")
                    object_mention_id = (
                        local_id_map.get(object_local_id)
                        if object_local_id
                        else None
                    )
                    if object_local_id and object_mention_id is None:
                        logger.warning(
                            "[%s] Claim references unknown object local_id %r, setting to None",
                            doc.doc_id, object_local_id,
                        )
                    object_freetext = raw.get("object_freetext") or None

                    try:
                        predicate = Predicate[raw["predicate"]]
                    except KeyError:
                        logger.warning(
                            "[%s] Unknown predicate %r, skipping", doc.doc_id, raw.get("predicate")
                        )
                        continue

                    subject_entity = mention_id_to_entity.get(subject_mention_id)
                    if subject_entity is not None:
                        # Check 1: document author may only be subject of DUPLICATE_OF
                        if (
                            subject_entity.entity_type == EntityType.PERSON
                            and subject_entity.attributes.get("is_document_author")
                            and predicate != Predicate.DUPLICATE_OF
                        ):
                            logger.warning(
                                "[%s] Dropping claim: document author %r used as subject of %s",
                                doc.doc_id, subject_entity.surface_form, predicate.value,
                            )
                            continue
                        # Check 2: per-predicate subject type constraint
                        allowed_types = _PREDICATE_SUBJECT_TYPES.get(predicate)
                        if allowed_types and subject_entity.entity_type not in allowed_types:
                            logger.warning(
                                "[%s] Dropping claim: %s subject must be %s, got %s",
                                doc.doc_id, predicate.value,
                                {t.value for t in allowed_types},
                                subject_entity.entity_type.value,
                            )
                            continue

                    raw_evidence_ids = raw.get("evidence_ids") or []
                    valid_ev_ids = [eid for eid in raw_evidence_ids if eid in evidence_ids_set]
                    if not valid_ev_ids:
                        logger.warning(
                            "[%s] Claim has no valid evidence IDs (got %r), skipping",
                            doc.doc_id, raw_evidence_ids,
                        )
                        continue

                    try:
                        cm = ClaimMention(
                            claim_mention_id=id_gen.claim_mention_id(),
                            doc_id=doc.doc_id,
                            predicate=predicate,
                            subject_mention_id=subject_mention_id,
                            object_mention_id=object_mention_id,
                            object_freetext=object_freetext,
                            evidence_ids=valid_ev_ids,
                            confidence=float(raw.get("confidence", 0.0)),
                            extraction_metadata=metadata,
                        )
                    except ValidationError as exc:
                        logger.warning("[%s] ClaimMention validation failed: %s", doc.doc_id, exc)
                        continue

                    if cm.confidence < 0.5:
                        logger.debug(
                            "[%s] Dropping claim %s: confidence %.2f < 0.5",
                            doc.doc_id, cm.claim_mention_id, cm.confidence,
                        )
                        continue
                    claim_mentions.append(cm)

                last_exc = None
                return claim_mentions

            except (ValidationError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Pass 3 attempt %d/%d failed for %s: %s",
                    attempt + 1, max_retries + 1, doc.doc_id, exc,
                )

        logger.error("Pass 3 all retries exhausted for %s: %s", doc.doc_id, last_exc)
        return []

    def _with_retry(self, doc: RawDocument, max_retries: int = 2) -> MentionExtractionResult:
        id_gen = IDGenerator(doc.doc_id)
        metadata = ExtractionMetadata(
            schema_version=SCHEMA_VERSION,
            model=config.LLM_KWARGS["model"],
            prompt_version=PROMPT_VERSION,
            extracted_at=datetime.now(timezone.utc),
        )

        entity_user_prompt = build_entity_user_prompt(doc)

        # --- Pass 1: entity extraction ---
        entity_mentions: list[EntityMention] = []
        local_id_map: dict[str, str] = {}
        last_exc: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                blocks = self._call_llm(
                    entity_user_prompt,
                    tools=[self.entity_tool],
                    system_prompt=self.entity_system_prompt,
                )
                entity_mentions, local_id_map = self._parse_entity_blocks(
                    blocks, doc, id_gen, metadata
                )
                self._mark_document_author(entity_mentions, doc)
                last_exc = None
                break
            except (ValidationError, ValueError) as exc:
                last_exc = exc
                logger.warning(
                    "Pass 1 attempt %d/%d failed for %s: %s",
                    attempt + 1, max_retries + 1, doc.doc_id, exc,
                )

        if last_exc is not None:
            logger.error("Pass 1 all retries exhausted for %s: %s", doc.doc_id, last_exc)
            return MentionExtractionResult(
                doc_id=doc.doc_id,
                entity_mentions=[],
                claim_mentions=[],
                evidence_records=[],
                extraction_metadata=metadata,
            )

        if not entity_mentions:
            return MentionExtractionResult(
                doc_id=doc.doc_id,
                entity_mentions=[],
                claim_mentions=[],
                evidence_records=[],
                extraction_metadata=metadata,
            )

        # --- Pass 2: evidence extraction ---
        evidence_records = self._run_pass2_evidence(doc, id_gen, metadata)

        if not evidence_records:
            return MentionExtractionResult(
                doc_id=doc.doc_id,
                entity_mentions=entity_mentions,
                claim_mentions=[],
                evidence_records=[],
                extraction_metadata=metadata,
            )

        # --- Pass 3: claim extraction ---
        claim_mentions = self._run_pass3_claims(
            doc, entity_mentions, evidence_records, local_id_map, id_gen, metadata
        )

        return MentionExtractionResult(
            doc_id=doc.doc_id,
            entity_mentions=entity_mentions,
            claim_mentions=claim_mentions,
            evidence_records=evidence_records,
            extraction_metadata=metadata,
        )
