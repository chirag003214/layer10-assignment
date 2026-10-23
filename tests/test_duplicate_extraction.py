from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from extraction.extractor import Extractor
from extraction.schema import (
    EntityType,
    ExtractionMetadata,
    Predicate,
)
from ingestion.preprocessor import RawDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta() -> ExtractionMetadata:
    return ExtractionMetadata(
        schema_version="2.0.0",
        model="test-model",
        prompt_version="v3",
        extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _raw_doc(**kwargs) -> RawDocument:
    defaults = {
        "doc_id": "ardupilot/ardupilot#30000",
        "doc_type": "comment",
        "issue_number": 30000,
        "source_url": "https://github.com/ardupilot/ardupilot/issues/30000#comment-1",
        "author": "peterbarker",
        "created_at": "2024-06-15T10:00:00Z",
        "title": "GPS glitch on CubeOrange",
        "body": (
            "This is a duplicate of #29500. Closing this issue."
        ),
        "state": "closed",
        "labels": ["Bug"],
    }
    return RawDocument(**(defaults | kwargs))


# ---------------------------------------------------------------------------
# Mock LLM responses for the 3-pass extraction
# ---------------------------------------------------------------------------

PASS1_ENTITY_RESPONSE = [
    {
        "name": "extract_entities",
        "input": {
            "entities": [
                {
                    "entity_type": "PERSON",
                    "surface_form": "peterbarker",
                    "attributes": {"github_handle": "peterbarker"},
                    "local_id": "e1",
                },
            ]
        },
    }
]

PASS2_EVIDENCE_RESPONSE = [
    {
        "name": "extract_evidence",
        "input": {
            "evidence": [
                {
                    "excerpt": "This is a duplicate of #29500. Closing this issue.",
                    "significance": "Explicit duplicate declaration linking this issue to #29500.",
                },
            ]
        },
    }
]


def _make_pass3_response(evidence_id: str) -> list[dict]:
    """Build pass 3 response referencing the actual evidence ID generated."""
    return [
        {
            "name": "extract_claims",
            "input": {
                "claims": [
                    {
                        "predicate": "DUPLICATE_OF",
                        "subject_local_id": "e1",
                        "object_local_id": None,
                        "object_freetext": "29500",
                        "evidence_ids": [evidence_id],
                        "confidence": 0.95,
                    },
                ]
            },
        }
    ]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

class TestDuplicateOfExtraction:
    def test_duplicate_claim_extracted(self):
        """Full 3-pass extraction produces a DUPLICATE_OF claim with correct fields."""
        doc = _raw_doc()
        extractor = Extractor()

        call_count = 0

        def mock_call_llm(user_prompt, tools, system_prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return PASS1_ENTITY_RESPONSE
            elif call_count == 2:
                return PASS2_EVIDENCE_RESPONSE
            else:
                # Pass 3 needs a real evidence ID — extract it from the prompt
                # The evidence list in the prompt looks like: "- ev_XXXX_0: '...'"
                for line in user_prompt.splitlines():
                    if line.strip().startswith("- ev_"):
                        ev_id = line.strip().split(":")[0].lstrip("- ")
                        return _make_pass3_response(ev_id)
                # Fallback: use a placeholder that will fail validation
                return _make_pass3_response("ev_unknown_0")

        with patch.object(extractor, "_call_llm", side_effect=mock_call_llm):
            result = extractor.extract(doc)

        # Verify entity: PERSON extracted
        assert len(result.entity_mentions) == 1
        person = result.entity_mentions[0]
        assert person.entity_type == EntityType.PERSON
        assert person.surface_form == "peterbarker"

        # Verify evidence: duplicate statement extracted
        assert len(result.evidence_records) == 1
        ev = result.evidence_records[0]
        assert "duplicate" in ev.excerpt.lower()

        # Verify claim: DUPLICATE_OF with correct fields
        assert len(result.claim_mentions) == 1
        claim = result.claim_mentions[0]
        assert claim.predicate == Predicate.DUPLICATE_OF
        assert claim.object_freetext == "29500"
        assert claim.object_mention_id is None
        assert len(claim.evidence_ids) >= 1
        assert claim.evidence_ids[0] == ev.evidence_id
        assert claim.confidence >= 0.7

    def test_duplicate_claim_subject_is_person(self):
        """The DUPLICATE_OF claim's subject should reference the PERSON entity."""
        doc = _raw_doc()
        extractor = Extractor()

        call_count = 0

        def mock_call_llm(user_prompt, tools, system_prompt):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return PASS1_ENTITY_RESPONSE
            elif call_count == 2:
                return PASS2_EVIDENCE_RESPONSE
            else:
                for line in user_prompt.splitlines():
                    if line.strip().startswith("- ev_"):
                        ev_id = line.strip().split(":")[0].lstrip("- ")
                        return _make_pass3_response(ev_id)
                return _make_pass3_response("ev_unknown_0")

        with patch.object(extractor, "_call_llm", side_effect=mock_call_llm):
            result = extractor.extract(doc)

        claim = result.claim_mentions[0]
        person = result.entity_mentions[0]
        assert claim.subject_mention_id == person.mention_id
