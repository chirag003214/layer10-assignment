from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from extraction.schema import (
    EntityMention,
    EntityType,
    ExtractionMetadata,
    IDGenerator,
    MentionExtractionResult,
    MentionSource,
)
from ingestion.preprocessor import RawDocument

logger = logging.getLogger(__name__)

KNOWN_COMPONENT_LABELS = {
    "copter", "plane", "rover", "sub", "tracker", "blimp",
    "gps", "ekf", "compass", "baro", "motors", "dshot",
    "mavlink", "sitl", "scripting", "logging", "camera",
    "gimbal", "rangefinder", "airspeed", "battery", "servo",
}

VERSION_RE = re.compile(r'(\d+\.\d+(?:\.\d+)?(?:-\w+)?)')

VEHICLE_PREFIXES = ["Copter", "Plane", "Rover"]


class StructuredExtractor:
    def __init__(self, schema_version: str, prompt_version: str) -> None:
        self.schema_version = schema_version
        self.prompt_version = prompt_version

    def _make_metadata(self) -> ExtractionMetadata:
        return ExtractionMetadata(
            schema_version=self.schema_version,
            model="structured_rules_v1",
            prompt_version=self.prompt_version,
            extracted_at=datetime.now(timezone.utc),
        )

    def _empty(self, doc: RawDocument) -> MentionExtractionResult:
        return MentionExtractionResult(
            doc_id=doc.doc_id,
            entity_mentions=[],
            claim_mentions=[],
            evidence_records=[],
            extraction_metadata=self._make_metadata(),
        )

    def extract(self, doc: RawDocument) -> MentionExtractionResult:
        event_type = doc.event_type or ""
        detail = doc.event_detail or {}

        if event_type in ("assigned", "unassigned"):
            return self._handle_assigned(doc, detail)
        elif event_type in ("labeled", "unlabeled"):
            return self._handle_labeled(doc, detail)
        elif event_type == "milestoned":
            return self._handle_milestoned(doc, detail)
        else:
            return self._empty(doc)

    def _handle_assigned(self, doc: RawDocument, detail: dict) -> MentionExtractionResult:
        assignee = detail.get("assignee")
        if not assignee:
            return self._empty(doc)

        metadata = self._make_metadata()
        id_gen = IDGenerator(doc.doc_id)
        surface_form = assignee
        em = EntityMention(
            mention_id=id_gen.mention_id(),
            doc_id=doc.doc_id,
            entity_type=EntityType.PERSON,
            surface_form=surface_form,
            normalized_form=surface_form.lower().strip(),
            attributes={"github_handle": assignee, "display_name": None},
            extraction_metadata=metadata,
            mention_source=MentionSource.STRUCTURED_RULE,
        )
        return MentionExtractionResult(
            doc_id=doc.doc_id,
            entity_mentions=[em],
            claim_mentions=[],
            evidence_records=[],
            extraction_metadata=metadata,
        )

    def _handle_labeled(self, doc: RawDocument, detail: dict) -> MentionExtractionResult:
        label = detail.get("label")
        if not label:
            return self._empty(doc)

        if label.lower() == "duplicate":
            return self._empty(doc)

        if label.lower() in KNOWN_COMPONENT_LABELS:
            metadata = self._make_metadata()
            id_gen = IDGenerator(doc.doc_id)
            surface_form = label
            em = EntityMention(
                mention_id=id_gen.mention_id(),
                doc_id=doc.doc_id,
                entity_type=EntityType.COMPONENT,
                surface_form=surface_form,
                normalized_form=surface_form.lower().strip(),
                attributes={},
                extraction_metadata=metadata,
                mention_source=MentionSource.STRUCTURED_RULE,
            )
            return MentionExtractionResult(
                doc_id=doc.doc_id,
                entity_mentions=[em],
                claim_mentions=[],
                evidence_records=[],
                extraction_metadata=metadata,
            )

        return self._empty(doc)

    def _handle_milestoned(self, doc: RawDocument, detail: dict) -> MentionExtractionResult:
        milestone = detail.get("milestone")
        if not milestone:
            return self._empty(doc)

        metadata = self._make_metadata()
        id_gen = IDGenerator(doc.doc_id)

        m = VERSION_RE.search(milestone)
        version_string = m.group(1) if m else milestone

        vehicle_type = None
        for prefix in VEHICLE_PREFIXES:
            if milestone.startswith(prefix):
                vehicle_type = prefix
                break

        surface_form = milestone
        em = EntityMention(
            mention_id=id_gen.mention_id(),
            doc_id=doc.doc_id,
            entity_type=EntityType.FIRMWARE_VERSION,
            surface_form=surface_form,
            normalized_form=surface_form.lower().strip(),
            attributes={"version_string": version_string, "vehicle_type": vehicle_type},
            extraction_metadata=metadata,
            mention_source=MentionSource.STRUCTURED_RULE,
        )
        return MentionExtractionResult(
            doc_id=doc.doc_id,
            entity_mentions=[em],
            claim_mentions=[],
            evidence_records=[],
            extraction_metadata=metadata,
        )
