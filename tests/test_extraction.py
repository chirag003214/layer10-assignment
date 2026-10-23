from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

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
from extraction.tools import get_tools
from extraction.prompts import build_entity_user_prompt
from extraction.extractor import Extractor
from extraction.pipeline import _load_processed_ids, run_extraction
from ingestion.preprocessor import RawDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _meta(schema_version: str = "2.0.0") -> ExtractionMetadata:
    return ExtractionMetadata(
        schema_version=schema_version,
        model="test-model",
        prompt_version="v3",
        extracted_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
    )


def _raw_doc(**kwargs) -> RawDocument:
    defaults = {
        "doc_id": "ardupilot/ardupilot#1",
        "doc_type": "issue",
        "issue_number": 1,
        "source_url": "https://github.com/ardupilot/ardupilot/issues/1",
        "author": "tridge",
        "created_at": "2024-01-01T00:00:00Z",
        "title": "GPS glitch on Copter 4.3",
        "body": "EKF lost GPS lock after upgrading to 4.3.2",
        "state": "closed",
        "labels": ["Bug", "GPS"],
    }
    return RawDocument(**(defaults | kwargs))


def _entity_mention(
    mention_id: str = "m_test_0",
    entity_type: EntityType = EntityType.COMPONENT,
    surface_form: str = "EKF",
    attributes: dict | None = None,
    doc_id: str = "ardupilot/ardupilot#1",
) -> EntityMention:
    return EntityMention(
        mention_id=mention_id,
        doc_id=doc_id,
        entity_type=entity_type,
        surface_form=surface_form,
        normalized_form=surface_form.lower().strip(),
        attributes=attributes or {},
        extraction_metadata=_meta(),
    )


def _evidence_record(
    evidence_id: str = "ev_test_0",
    doc_id: str = "ardupilot/ardupilot#1",
    source_status: SourceStatus = SourceStatus.ACTIVE,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        doc_id=doc_id,
        source_hash="abc123",
        significance="test significance",
        excerpt="EKF lost GPS lock",
        char_offset_start=0,
        char_offset_end=18,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        author="tridge",
        source_status=source_status,
    )


def _claim_mention(
    claim_mention_id: str = "cm_test_0",
    predicate: Predicate = Predicate.AFFECTS,
    subject_mention_id: str = "m_test_0",
    object_mention_id: str | None = None,
    object_freetext: str | None = None,
    evidence_ids: list[str] | None = None,
    confidence: float = 0.9,
    doc_id: str = "ardupilot/ardupilot#1",
) -> ClaimMention:
    return ClaimMention(
        claim_mention_id=claim_mention_id,
        doc_id=doc_id,
        predicate=predicate,
        subject_mention_id=subject_mention_id,
        object_mention_id=object_mention_id,
        object_freetext=object_freetext,
        evidence_ids=evidence_ids or ["ev_test_0"],
        confidence=confidence,
        extraction_metadata=_meta(),
    )


# ---------------------------------------------------------------------------
# 1. Schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entity_type,surface,attrs", [
    (EntityType.PERSON, "tridge", {"github_handle": "tridge"}),
    (EntityType.COMPONENT, "EKF", {}),
    (EntityType.PARAMETER, "EK3_ENABLE", {"value": "1"}),
    (EntityType.FIRMWARE_VERSION, "4.3.2", {"version_string": "4.3.2"}),
    (EntityType.HARDWARE_PLATFORM, "Pixhawk 4", {}),
])
def test_entity_mention_roundtrip(entity_type, surface, attrs):
    em = _entity_mention(entity_type=entity_type, surface_form=surface, attributes=attrs)
    json_str = em.model_dump_json()
    back = EntityMention.model_validate_json(json_str)
    assert back.entity_type == entity_type
    assert back.surface_form == surface


@pytest.mark.parametrize("predicate,obj_mid,obj_ft", [
    (Predicate.CAUSES, None, "GPS loss"),
    (Predicate.AFFECTS, None, None),
    (Predicate.FIXED_IN, "m_test_1", None),
    (Predicate.REGRESSED_IN, "m_test_1", None),
    (Predicate.HAS_PARAMETER, "m_test_1", None),
    (Predicate.DUPLICATE_OF, None, "42"),
])
def test_claim_mention_roundtrip(predicate, obj_mid, obj_ft):
    cm = _claim_mention(
        predicate=predicate,
        object_mention_id=obj_mid,
        object_freetext=obj_ft,
    )
    json_str = cm.model_dump_json()
    back = ClaimMention.model_validate_json(json_str)
    assert back.predicate == predicate


def test_mention_extraction_result_roundtrip():
    em = _entity_mention()
    ev = _evidence_record()
    cm = _claim_mention()
    result = MentionExtractionResult(
        doc_id="ardupilot/ardupilot#1",
        entity_mentions=[em],
        claim_mentions=[cm],
        evidence_records=[ev],
        extraction_metadata=_meta(),
    )
    json_str = result.model_dump_json()
    back = MentionExtractionResult.model_validate_json(json_str)
    assert back.doc_id == result.doc_id
    assert len(back.entity_mentions) == 1
    assert len(back.claim_mentions) == 1
    assert len(back.evidence_records) == 1


def test_invalid_entity_type_raises():
    with pytest.raises(ValidationError):
        EntityMention(
            mention_id="m1",
            doc_id="doc1",
            entity_type="robot",
            surface_form="x",
            normalized_form="x",
            attributes={},
            extraction_metadata=_meta(),
        )


def test_claim_mention_rejects_empty_evidence():
    with pytest.raises(ValidationError):
        ClaimMention(
            claim_mention_id="cm1",
            doc_id="doc1",
            predicate=Predicate.AFFECTS,
            subject_mention_id="m1",
            evidence_ids=[],
            confidence=0.9,
            extraction_metadata=_meta(),
        )


# ---------------------------------------------------------------------------
# 2. Tools
# ---------------------------------------------------------------------------

def test_get_tools_returns_three_tools():
    tools = get_tools()
    assert len(tools) == 3
    names = {t["function"]["name"] for t in tools}
    assert names == {"extract_entities", "extract_evidence", "extract_claims"}


def test_tool_schemas_have_defs():
    tools = get_tools()
    for tool in tools:
        assert "$defs" in tool["function"]["parameters"], (
            f"{tool['function']['name']} schema missing '$defs'"
        )


# ---------------------------------------------------------------------------
# 3. Prompts
# ---------------------------------------------------------------------------

def test_entity_prompt_contains_doc_fields():
    doc = _raw_doc()
    prompt = build_entity_user_prompt(doc)
    assert doc.doc_id in prompt
    assert doc.author in prompt
    assert doc.body in prompt


def test_entity_prompt_no_body():
    doc = _raw_doc(body=None)
    prompt = build_entity_user_prompt(doc)
    assert "(no body)" in prompt


def test_entity_prompt_no_labels():
    doc = _raw_doc(labels=[])
    prompt = build_entity_user_prompt(doc)
    assert "(none)" in prompt


# ---------------------------------------------------------------------------
# 4. Extractor (all with mocked LLM)
# ---------------------------------------------------------------------------

def test_parse_entity_blocks_valid():
    extractor = Extractor()
    doc = _raw_doc()
    id_gen = IDGenerator(doc.doc_id)
    metadata = _meta()

    entity_data = {
        "entity_type": "COMPONENT",
        "surface_form": "EKF",
        "attributes": {},
        "local_id": "e1",
    }
    blocks = [
        {"name": "extract_entities", "input": {"entities": [entity_data]}},
    ]

    mentions, local_id_map = extractor._parse_entity_blocks(blocks, doc, id_gen, metadata)
    assert len(mentions) == 1
    assert mentions[0].entity_type == EntityType.COMPONENT
    assert mentions[0].surface_form == "EKF"
    assert "e1" in local_id_map


def test_filter_entities_drops_git_sha_version():
    extractor = Extractor()
    sha = "a" * 40
    em = _entity_mention(
        entity_type=EntityType.FIRMWARE_VERSION,
        surface_form=sha,
        attributes={"version_string": sha},
    )
    local_map = {"e1": em.mention_id}
    kept, updated_map = extractor._filter_entities([em], local_map)
    assert len(kept) == 0
    assert "e1" not in updated_map


def test_filter_entities_drops_dotted_parameter():
    extractor = Extractor()
    em = _entity_mention(
        entity_type=EntityType.PARAMETER,
        surface_form="rpmdata.last_update_us",
    )
    kept, _ = extractor._filter_entities([em], {})
    assert len(kept) == 0


def test_with_retry_fallback_on_llm_error():
    extractor = Extractor()
    doc = _raw_doc()
    with patch.object(extractor, "_call_llm", side_effect=ValueError("LLM unavailable")):
        result = extractor._with_retry(doc, max_retries=0)
    assert result.doc_id == doc.doc_id
    assert result.entity_mentions == []
    assert result.claim_mentions == []


# ---------------------------------------------------------------------------
# 5. Pipeline
# ---------------------------------------------------------------------------

def test_load_processed_ids_no_file(tmp_path):
    ids = _load_processed_ids(tmp_path / "nonexistent.jsonl")
    assert ids == set()


def test_load_processed_ids_from_jsonl(tmp_path):
    jsonl_path = tmp_path / "entity_mentions.jsonl"
    em1 = _entity_mention(mention_id="m1", doc_id="ardupilot/ardupilot#1")
    em2 = _entity_mention(mention_id="m2", doc_id="ardupilot/ardupilot#2")
    jsonl_path.write_text(
        em1.model_dump_json() + "\n" + em2.model_dump_json() + "\n"
    )
    ids = _load_processed_ids(jsonl_path)
    assert ids == {"ardupilot/ardupilot#1", "ardupilot/ardupilot#2"}


# ---------------------------------------------------------------------------
# 6. Author flagging and subject-type validation
# ---------------------------------------------------------------------------

def _make_person_mention(
    mention_id: str = "m_person_0",
    surface_form: str = "tridge",
    github_handle: str = "tridge",
    doc_id: str = "ardupilot/ardupilot#1",
) -> EntityMention:
    return EntityMention(
        mention_id=mention_id,
        doc_id=doc_id,
        entity_type=EntityType.PERSON,
        surface_form=surface_form,
        normalized_form=surface_form.lower().strip(),
        attributes={"github_handle": github_handle},
        extraction_metadata=_meta(),
    )


def _run_pass3_with_one_claim(
    extractor: Extractor,
    doc: RawDocument,
    subject_entity: EntityMention,
    predicate_key: str,  # Enum name, e.g. "CAUSES"
    mark_author: bool = False,
) -> list[ClaimMention]:
    """Call _run_pass3_claims with a single raw claim for the given subject + predicate."""
    if mark_author:
        subject_entity.attributes["is_document_author"] = True

    ev = _evidence_record(doc_id=doc.doc_id)
    local_id_map = {"e1": subject_entity.mention_id}
    id_gen = IDGenerator(doc.doc_id)
    metadata = _meta()

    if predicate_key == "DUPLICATE_OF":
        raw_claim = {
            "predicate": "DUPLICATE_OF",
            "subject_local_id": "e1",
            "object_local_id": None,
            "object_freetext": "12345",
            "evidence_ids": [ev.evidence_id],
            "confidence": 0.9,
        }
    elif predicate_key == "AFFECTS":
        raw_claim = {
            "predicate": "AFFECTS",
            "subject_local_id": "e1",
            "object_local_id": None,
            "object_freetext": None,
            "evidence_ids": [ev.evidence_id],
            "confidence": 0.9,
        }
    else:
        raw_claim = {
            "predicate": predicate_key,
            "subject_local_id": "e1",
            "object_local_id": None,
            "object_freetext": "observable symptom",
            "evidence_ids": [ev.evidence_id],
            "confidence": 0.9,
        }

    blocks = [{"name": "extract_claims", "input": {"claims": [raw_claim]}}]
    with patch.object(extractor, "_call_llm", return_value=blocks):
        return extractor._run_pass3_claims(
            doc, [subject_entity], [ev], local_id_map, id_gen, metadata
        )


def test_mark_document_author_sets_flag_on_github_handle_match():
    extractor = Extractor()
    doc = _raw_doc(author="tridge")
    em = _make_person_mention(surface_form="tridge", github_handle="tridge")
    extractor._mark_document_author([em], doc)
    assert em.attributes.get("is_document_author") is True


def test_mark_document_author_no_match_leaves_flag_unset():
    extractor = Extractor()
    doc = _raw_doc(author="tridge")
    em = _make_person_mention(surface_form="peterbarker", github_handle="peterbarker")
    extractor._mark_document_author([em], doc)
    assert "is_document_author" not in em.attributes


def test_author_person_dropped_for_causes():
    extractor = Extractor()
    doc = _raw_doc(author="tridge")
    person = _make_person_mention(surface_form="tridge", github_handle="tridge")
    claims = _run_pass3_with_one_claim(extractor, doc, person, "CAUSES", mark_author=True)
    assert claims == []


def test_author_person_kept_for_duplicate_of():
    extractor = Extractor()
    doc = _raw_doc(author="tridge")
    person = _make_person_mention(surface_form="tridge", github_handle="tridge")
    claims = _run_pass3_with_one_claim(extractor, doc, person, "DUPLICATE_OF", mark_author=True)
    assert len(claims) == 1
    assert claims[0].predicate == Predicate.DUPLICATE_OF


def test_non_author_person_dropped_for_causes():
    extractor = Extractor()
    doc = _raw_doc(author="tridge")
    # Different person — not the document author
    person = _make_person_mention(surface_form="peterbarker", github_handle="peterbarker")
    claims = _run_pass3_with_one_claim(extractor, doc, person, "CAUSES", mark_author=False)
    assert claims == []


def test_component_kept_for_causes():
    extractor = Extractor()
    doc = _raw_doc(author="tridge")
    component = _entity_mention(
        mention_id="m_comp_0",
        entity_type=EntityType.COMPONENT,
        surface_form="EKF",
        doc_id=doc.doc_id,
    )
    claims = _run_pass3_with_one_claim(extractor, doc, component, "CAUSES")
    assert len(claims) == 1
    assert claims[0].predicate == Predicate.CAUSES


def test_component_dropped_for_duplicate_of():
    extractor = Extractor()
    doc = _raw_doc(author="tridge")
    component = _entity_mention(
        mention_id="m_comp_0",
        entity_type=EntityType.COMPONENT,
        surface_form="EKF",
        doc_id=doc.doc_id,
    )
    claims = _run_pass3_with_one_claim(extractor, doc, component, "DUPLICATE_OF")
    assert claims == []


def test_run_extraction_skips_already_processed(tmp_path):
    doc1 = _raw_doc(doc_id="ardupilot/ardupilot#1", issue_number=1)
    doc2 = _raw_doc(doc_id="ardupilot/ardupilot#2", issue_number=2)

    input_path = tmp_path / "documents.jsonl"
    input_path.write_text(doc1.model_dump_json() + "\n" + doc2.model_dump_json() + "\n")

    mentions_dir = tmp_path / "mentions"
    mentions_dir.mkdir()

    # Pre-populate entity_mentions.jsonl so doc1 is considered processed
    em1 = _entity_mention(mention_id="m1", doc_id="ardupilot/ardupilot#1")
    (mentions_dir / "entity_mentions.jsonl").write_text(em1.model_dump_json() + "\n")
    (mentions_dir / "claim_mentions.jsonl").write_text("")
    (mentions_dir / "evidence_records.jsonl").write_text("")

    # Mock Extractor.extract to return a result for doc2
    mock_result = MentionExtractionResult(
        doc_id="ardupilot/ardupilot#2",
        entity_mentions=[_entity_mention(mention_id="m2", doc_id="ardupilot/ardupilot#2")],
        claim_mentions=[],
        evidence_records=[],
        extraction_metadata=_meta(),
    )

    with patch("extraction.pipeline.Extractor") as MockExtractor:
        MockExtractor.return_value.extract.return_value = mock_result
        run_extraction(input_path=input_path, mentions_dir=mentions_dir, verbose=False)

    # entity_mentions.jsonl should now have 2 lines (doc1 + doc2)
    lines = [
        l for l in (mentions_dir / "entity_mentions.jsonl").read_text().strip().split("\n") if l
    ]
    assert len(lines) == 2
    doc_ids = {json.loads(l)["doc_id"] for l in lines}
    assert doc_ids == {"ardupilot/ardupilot#1", "ardupilot/ardupilot#2"}
