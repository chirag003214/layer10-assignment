from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import kuzu
import pytest

from graph.schema import NODE_TABLES, REL_TABLES
from graph.loader import GraphLoader


@pytest.fixture()
def tmp_db(tmp_path):
    """Create a temporary Kuzu database."""
    db_path = tmp_path / "test.kuzu"
    return str(db_path)


@pytest.fixture()
def sample_data_dir(tmp_path):
    """Create minimal sample data files for testing."""
    # documents.jsonl
    docs_dir = tmp_path / "processed"
    docs_dir.mkdir()
    docs = [
        {
            "doc_id": "repo/proj#1",
            "doc_type": "issue",
            "issue_number": 1,
            "title": "Test issue",
            "author": "alice",
            "created_at": "2026-01-01T00:00:00Z",
            "state": "open",
            "source_url": "https://github.com/repo/proj/issues/1",
        },
        {
            "doc_id": "repo/proj#1/c/100",
            "doc_type": "comment",
            "issue_number": 1,
            "title": None,
            "author": "bob",
            "created_at": "2026-01-02T00:00:00Z",
            "state": None,
            "source_url": "https://github.com/repo/proj/issues/1#issuecomment-100",
        },
    ]
    with open(docs_dir / "documents.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")

    # evidence_records.jsonl
    mentions_dir = tmp_path / "mentions"
    mentions_dir.mkdir()
    evidence = [
        {
            "evidence_id": "ev_abc123_0",
            "doc_id": "repo/proj#1",
            "source_hash": "hash1",
            "significance": "Shows the bug symptom",
            "excerpt": "The widget crashes on startup",
            "char_offset_start": 0,
            "char_offset_end": 30,
            "timestamp": "2026-01-01T00:00:00Z",
            "author": "alice",
            "source_status": "active",
        },
        {
            "evidence_id": "ev_abc123_1",
            "doc_id": "repo/proj#1/c/100",
            "source_hash": "hash2",
            "significance": "Confirms the fix",
            "excerpt": "Fixed in v2.0",
            "char_offset_start": 0,
            "char_offset_end": 13,
            "timestamp": "2026-01-02T00:00:00Z",
            "author": "bob",
            "source_status": "active",
        },
    ]
    with open(mentions_dir / "evidence_records.jsonl", "w") as f:
        for e in evidence:
            f.write(json.dumps(e) + "\n")

    # entity_mentions.jsonl
    mentions = [
        {
            "mention_id": "m_aaa_0",
            "doc_id": "repo/proj#1",
            "entity_type": "person",
            "surface_form": "alice",
            "normalized_form": "alice",
            "attributes": {"github_handle": "alice"},
            "mention_source": "llm_extracted",
        },
        {
            "mention_id": "m_aaa_1",
            "doc_id": "repo/proj#1",
            "entity_type": "component",
            "surface_form": "Widget",
            "normalized_form": "widget",
            "attributes": {},
            "mention_source": "llm_extracted",
        },
        {
            "mention_id": "m_bbb_0",
            "doc_id": "repo/proj#1/c/100",
            "entity_type": "person",
            "surface_form": "bob",
            "normalized_form": "bob",
            "attributes": {"github_handle": "bob"},
            "mention_source": "structured_rule",
        },
        {
            "mention_id": "m_bbb_1",
            "doc_id": "repo/proj#1/c/100",
            "entity_type": "parameter",
            "surface_form": "MAX_SPEED",
            "normalized_form": "max_speed",
            "attributes": {"value": "100"},
            "mention_source": "llm_extracted",
        },
    ]
    with open(mentions_dir / "entity_mentions.jsonl", "w") as f:
        for m in mentions:
            f.write(json.dumps(m) + "\n")

    # canonical_entities.json
    canonical_dir = tmp_path / "canonical"
    canonical_dir.mkdir()
    entities = [
        {
            "canonical_id": "ce_person_aaa",
            "entity_type": "person",
            "canonical_name": "alice",
            "aliases": ["@alice"],
            "attributes": {"github_handle": "alice"},
            "mention_ids": ["m_aaa_0"],
            "first_seen": "2026-01-01T00:00:00Z",
            "last_seen": "2026-01-01T00:00:00Z",
        },
        {
            "canonical_id": "ce_person_bbb",
            "entity_type": "person",
            "canonical_name": "bob",
            "aliases": [],
            "attributes": {"github_handle": "bob"},
            "mention_ids": ["m_bbb_0"],
            "first_seen": "2026-01-02T00:00:00Z",
            "last_seen": "2026-01-02T00:00:00Z",
        },
        {
            "canonical_id": "ce_component_ccc",
            "entity_type": "component",
            "canonical_name": "Widget",
            "aliases": ["widget"],
            "attributes": {},
            "mention_ids": ["m_aaa_1"],
            "first_seen": "2026-01-01T00:00:00Z",
            "last_seen": "2026-01-01T00:00:00Z",
        },
        {
            "canonical_id": "ce_parameter_ddd",
            "entity_type": "parameter",
            "canonical_name": "MAX_SPEED",
            "aliases": [],
            "attributes": {"value": "100"},
            "mention_ids": ["m_bbb_1"],
            "first_seen": "2026-01-02T00:00:00Z",
            "last_seen": "2026-01-02T00:00:00Z",
        },
    ]
    with open(canonical_dir / "canonical_entities.json", "w") as f:
        json.dump(entities, f)

    # canonical_claims.json
    claims = [
        {
            "canonical_claim_id": "cc_claim1",
            "predicate": "affects",
            "subject_canonical_id": "ce_component_ccc",
            "object_canonical_id": None,
            "object_freetext": "startup crash",
            "object_freetext_variants": ["crash on boot"],
            "evidence_ids": ["ev_abc123_0"],
            "provenance": [],
            "confidence": 0.9,
            "is_current": True,
            "valid_from": None,
            "valid_until": None,
            "assertion_time": "2026-01-01T00:00:00Z",
            "event_time": None,
            "superseded_by": None,
        },
        {
            "canonical_claim_id": "cc_claim2",
            "predicate": "fixed_in",
            "subject_canonical_id": "ce_component_ccc",
            "object_canonical_id": None,
            "object_freetext": "version 2.0",
            "object_freetext_variants": [],
            "evidence_ids": ["ev_abc123_1"],
            "provenance": [],
            "confidence": 0.85,
            "is_current": True,
            "valid_from": "2026-01-02",
            "valid_until": None,
            "assertion_time": "2026-01-02T00:00:00Z",
            "event_time": None,
            "superseded_by": None,
        },
        {
            "canonical_claim_id": "cc_claim3",
            "predicate": "has_parameter",
            "subject_canonical_id": "ce_component_ccc",
            "object_canonical_id": "ce_parameter_ddd",
            "object_freetext": None,
            "object_freetext_variants": [],
            "evidence_ids": ["ev_abc123_0", "ev_abc123_1"],
            "provenance": [],
            "confidence": 0.8,
            "is_current": False,
            "valid_from": None,
            "valid_until": None,
            "assertion_time": "2026-01-01T00:00:00Z",
            "event_time": None,
            "superseded_by": "cc_claim2",
        },
    ]
    with open(canonical_dir / "canonical_claims.json", "w") as f:
        json.dump(claims, f)

    # mention_to_canonical.json
    m2c = {
        "m_aaa_0": "ce_person_aaa",
        "m_aaa_1": "ce_component_ccc",
        "m_bbb_0": "ce_person_bbb",
        "m_bbb_1": "ce_parameter_ddd",
    }
    with open(canonical_dir / "mention_to_canonical.json", "w") as f:
        json.dump(m2c, f)

    return tmp_path


# ── schema tests ──────────────────────────────────────────────────

class TestSchema:
    def test_node_tables_is_list_of_strings(self):
        assert isinstance(NODE_TABLES, list)
        assert len(NODE_TABLES) == 8
        for stmt in NODE_TABLES:
            assert "CREATE NODE TABLE" in stmt
            assert "IF NOT EXISTS" in stmt

    def test_rel_tables_is_list_of_strings(self):
        assert isinstance(REL_TABLES, list)
        assert len(REL_TABLES) == 18
        for stmt in REL_TABLES:
            assert "CREATE REL TABLE" in stmt
            assert "IF NOT EXISTS" in stmt

    def test_node_table_names(self):
        expected = {
            "Person", "Component", "Parameter", "FirmwareVersion",
            "HardwarePlatform", "Claim", "Evidence", "Document",
        }
        found = set()
        for stmt in NODE_TABLES:
            # extract table name from "CREATE NODE TABLE IF NOT EXISTS Foo("
            name = stmt.split("IF NOT EXISTS")[1].split("(")[0].strip()
            found.add(name)
        assert found == expected


# ── loader tests ──────────────────────────────────────────────────

class TestLoaderInit:
    def test_init_creates_database(self, tmp_db):
        loader = GraphLoader(tmp_db)
        assert loader.db is not None
        assert loader.conn is not None

    def test_init_schema_runs_without_error(self, tmp_db):
        loader = GraphLoader(tmp_db)
        loader.init_schema()
        # verify a node table exists by querying it
        result = loader.conn.execute("MATCH (d:Document) RETURN count(d)")
        assert result.get_next() == [0]

    def test_init_schema_idempotent(self, tmp_db):
        loader = GraphLoader(tmp_db)
        loader.init_schema()
        loader.init_schema()  # should not raise


class TestLoaderLoad:
    @pytest.fixture(autouse=True)
    def setup_loader(self, tmp_db, sample_data_dir):
        self.loader = GraphLoader(tmp_db, data_dir=str(sample_data_dir))
        self.loader.init_schema()

    def test_load_documents(self):
        self.loader.load_documents()
        result = self.loader.conn.execute("MATCH (d:Document) RETURN count(d)")
        assert result.get_next() == [2]

    def test_load_evidence(self):
        self.loader.load_documents()
        self.loader.load_evidence()
        result = self.loader.conn.execute("MATCH (e:Evidence) RETURN count(e)")
        assert result.get_next() == [2]

    def test_load_entities(self):
        self.loader.load_entities()
        result = self.loader.conn.execute("MATCH (p:Person) RETURN count(p)")
        assert result.get_next() == [2]
        result = self.loader.conn.execute("MATCH (c:Component) RETURN count(c)")
        assert result.get_next() == [1]
        result = self.loader.conn.execute("MATCH (p:Parameter) RETURN count(p)")
        assert result.get_next() == [1]

    def test_load_entities_person_attributes(self):
        self.loader.load_entities()
        result = self.loader.conn.execute(
            "MATCH (p:Person {canonical_id: 'ce_person_aaa'}) "
            "RETURN p.github_handle, p.aliases, p.mention_count"
        )
        row = result.get_next()
        assert row[0] == "alice"
        assert row[1] == ["@alice"]
        assert row[2] == 1

    def test_load_entities_parameter_value(self):
        self.loader.load_entities()
        result = self.loader.conn.execute(
            "MATCH (p:Parameter {canonical_id: 'ce_parameter_ddd'}) "
            "RETURN p.value"
        )
        row = result.get_next()
        assert row[0] == "100"

    def test_load_claims(self):
        self.loader.load_entities()
        self.loader.load_claims()
        result = self.loader.conn.execute("MATCH (c:Claim) RETURN count(c)")
        assert result.get_next() == [3]

    def test_load_claims_support_level(self):
        self.loader.load_entities()
        self.loader.load_claims()
        # claim3 has 2 evidence_ids -> "strong"
        result = self.loader.conn.execute(
            "MATCH (c:Claim {canonical_claim_id: 'cc_claim3'}) "
            "RETURN c.support_level"
        )
        assert result.get_next()[0] == "strong"

    def test_load_subject_relationships(self):
        self.loader.load_entities()
        self.loader.load_claims()
        self.loader.load_subject_relationships()
        result = self.loader.conn.execute(
            "MATCH (c:Component)-[:Component_HAS_SUBJECT]->(cl:Claim) "
            "RETURN count(cl)"
        )
        assert result.get_next() == [3]

    def test_load_object_relationships(self):
        self.loader.load_entities()
        self.loader.load_claims()
        self.loader.load_object_relationships()
        result = self.loader.conn.execute(
            "MATCH (cl:Claim)-[:Claim_HAS_OBJECT_Parameter]->(p:Parameter) "
            "RETURN count(p)"
        )
        assert result.get_next() == [1]

    def test_load_supported_by(self):
        self.loader.load_documents()
        self.loader.load_evidence()
        self.loader.load_entities()
        self.loader.load_claims()
        self.loader.load_supported_by()
        result = self.loader.conn.execute(
            "MATCH (cl:Claim)-[:SUPPORTED_BY]->(e:Evidence) RETURN count(e)"
        )
        # claim1: 1 ev, claim2: 1 ev, claim3: 2 ev = 4 total
        assert result.get_next() == [4]

    def test_load_sourced_from(self):
        self.loader.load_documents()
        self.loader.load_evidence()
        self.loader.load_sourced_from()
        result = self.loader.conn.execute(
            "MATCH (e:Evidence)-[:SOURCED_FROM]->(d:Document) RETURN count(d)"
        )
        assert result.get_next() == [2]

    def test_load_mentioned_in(self):
        self.loader.load_documents()
        self.loader.load_entities()
        self.loader.load_mentioned_in()
        # alice mentioned in doc#1, bob in doc#1/c/100,
        # widget in doc#1, max_speed in doc#1/c/100
        result = self.loader.conn.execute(
            "MATCH (p:Person)-[:Person_MENTIONED_IN]->(d:Document) RETURN count(d)"
        )
        assert result.get_next() == [2]

    def test_load_superseded_by(self):
        self.loader.load_entities()
        self.loader.load_claims()
        self.loader.load_superseded_by()
        result = self.loader.conn.execute(
            "MATCH (c1:Claim)-[:Claim_SUPERSEDED_BY]->(c2:Claim) RETURN count(c2)"
        )
        assert result.get_next() == [1]

    def test_load_all(self):
        """Full integration: load_all should populate everything."""
        self.loader.load_all()
        # spot-check node counts
        result = self.loader.conn.execute("MATCH (d:Document) RETURN count(d)")
        assert result.get_next() == [2]
        result = self.loader.conn.execute("MATCH (e:Evidence) RETURN count(e)")
        assert result.get_next() == [2]
        result = self.loader.conn.execute("MATCH (c:Claim) RETURN count(c)")
        assert result.get_next() == [3]
        # spot-check relationship counts
        result = self.loader.conn.execute(
            "MATCH ()-[r:SUPPORTED_BY]->() RETURN count(r)"
        )
        assert result.get_next() == [4]
