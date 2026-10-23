from __future__ import annotations

"""Load resolved pipeline data into a Kuzu property graph."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import kuzu

from graph.schema import (
    ENTITY_TYPE_TO_TABLE,
    NODE_TABLES,
    REL_TABLES,
)

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "data"


def _parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO timestamp string to a datetime, or return None."""
    if not value:
        return None
    s = value.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def _support_level(n_evidence: int) -> str:
    if n_evidence == 0:
        return "unsupported"
    if n_evidence == 1:
        return "single"
    if n_evidence <= 3:
        return "strong"
    return "overwhelming"


class GraphLoader:
    """Loads all resolved data into a Kuzu graph database."""

    def __init__(self, db_path: str, *, data_dir: str = DEFAULT_DATA_DIR):
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self.data_dir = Path(data_dir)

        # lazily populated caches
        self._mention_source_map: dict[str, str] | None = None
        self._mention_doc_map: dict[str, str] | None = None
        self._entity_type_map: dict[str, str] | None = None
        self._mention_to_canonical: dict[str, str] | None = None

    # ── schema ────────────────────────────────────────────────────

    def init_schema(self) -> None:
        for stmt in NODE_TABLES:
            self.conn.execute(stmt)
        for stmt in REL_TABLES:
            self.conn.execute(stmt)
        logger.info("Schema initialized (%d node + %d rel tables)", len(NODE_TABLES), len(REL_TABLES))

    # ── orchestration ─────────────────────────────────────────────

    def load_all(self) -> None:
        self.load_documents()
        self.load_evidence()
        self.load_entities()
        self.load_claims()
        self.load_subject_relationships()
        self.load_object_relationships()
        self.load_supported_by()
        self.load_sourced_from()
        self.load_mentioned_in()
        self.load_superseded_by()
        logger.info("All data loaded")

    # ── data helpers ──────────────────────────────────────────────

    def _read_jsonl(self, path: Path) -> list[dict]:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    def _read_json(self, path: Path) -> list | dict:
        with open(path) as f:
            return json.load(f)

    def _get_mention_source_map(self) -> dict[str, str]:
        """mention_id → mention_source from entity_mentions.jsonl."""
        if self._mention_source_map is None:
            mentions = self._read_jsonl(self.data_dir / "mentions" / "entity_mentions.jsonl")
            self._mention_source_map = {m["mention_id"]: m.get("mention_source", "") for m in mentions}
        return self._mention_source_map

    def _get_mention_doc_map(self) -> dict[str, str]:
        """mention_id → doc_id from entity_mentions.jsonl."""
        if self._mention_doc_map is None:
            mentions = self._read_jsonl(self.data_dir / "mentions" / "entity_mentions.jsonl")
            self._mention_doc_map = {m["mention_id"]: m["doc_id"] for m in mentions}
        return self._mention_doc_map

    def _get_entity_type_map(self) -> dict[str, str]:
        """canonical_id → entity_type from canonical_entities.json."""
        if self._entity_type_map is None:
            entities = self._read_json(self.data_dir / "canonical" / "canonical_entities.json")
            self._entity_type_map = {e["canonical_id"]: e["entity_type"] for e in entities}
        return self._entity_type_map

    def _get_mention_to_canonical(self) -> dict[str, str]:
        if self._mention_to_canonical is None:
            self._mention_to_canonical = self._read_json(
                self.data_dir / "canonical" / "mention_to_canonical.json"
            )
        return self._mention_to_canonical

    # ── node loaders ──────────────────────────────────────────────

    def load_documents(self) -> None:
        docs = self._read_jsonl(self.data_dir / "processed" / "documents.jsonl")
        for d in docs:
            self.conn.execute(
                "MERGE (n:Document {doc_id: $id}) "
                "SET n.doc_type = $doc_type, n.issue_number = $issue_number, "
                "n.title = $title, n.author = $author, n.created_at = $created_at, "
                "n.state = $state, n.source_url = $source_url",
                parameters={
                    "id": d["doc_id"],
                    "doc_type": d.get("doc_type"),
                    "issue_number": d.get("issue_number"),
                    "title": d.get("title"),
                    "author": d.get("author"),
                    "created_at": _parse_ts(d.get("created_at")),
                    "state": d.get("state"),
                    "source_url": d.get("source_url"),
                },
            )
        logger.info("Loaded %d documents", len(docs))

    def load_evidence(self) -> None:
        records = self._read_jsonl(self.data_dir / "mentions" / "evidence_records.jsonl")
        for ev in records:
            self.conn.execute(
                "MERGE (n:Evidence {evidence_id: $id}) "
                "SET n.excerpt = $excerpt, n.significance = $significance, "
                "n.char_offset_start = $offset_start, n.char_offset_end = $offset_end, "
                "n.timestamp = $ts, n.author = $author, "
                "n.source_hash = $source_hash, n.source_status = $source_status",
                parameters={
                    "id": ev["evidence_id"],
                    "excerpt": ev.get("excerpt"),
                    "significance": ev.get("significance"),
                    "offset_start": ev.get("char_offset_start"),
                    "offset_end": ev.get("char_offset_end"),
                    "ts": _parse_ts(ev.get("timestamp")),
                    "author": ev.get("author"),
                    "source_hash": ev.get("source_hash"),
                    "source_status": ev.get("source_status"),
                },
            )
        logger.info("Loaded %d evidence records", len(records))

    def load_entities(self) -> None:
        entities = self._read_json(self.data_dir / "canonical" / "canonical_entities.json")
        source_map = self._get_mention_source_map()

        for ent in entities:
            etype = ent["entity_type"]
            table = ENTITY_TYPE_TO_TABLE.get(etype)
            if not table:
                logger.warning("Unknown entity type %s, skipping %s", etype, ent["canonical_id"])
                continue

            mention_ids = ent.get("mention_ids", [])
            mention_count = len(mention_ids)
            mention_source = source_map.get(mention_ids[0], "") if mention_ids else ""
            attrs = ent.get("attributes", {})

            if table == "Person":
                self.conn.execute(
                    "MERGE (n:Person {canonical_id: $id}) "
                    "SET n.canonical_name = $name, n.github_handle = $gh, "
                    "n.aliases = $aliases, n.mention_count = $mc, "
                    "n.first_seen = $fs, n.last_seen = $ls, n.mention_source = $ms",
                    parameters={
                        "id": ent["canonical_id"],
                        "name": ent["canonical_name"],
                        "gh": attrs.get("github_handle"),
                        "aliases": ent.get("aliases", []),
                        "mc": mention_count,
                        "fs": _parse_ts(ent.get("first_seen")),
                        "ls": _parse_ts(ent.get("last_seen")),
                        "ms": mention_source,
                    },
                )
            elif table == "Component":
                self.conn.execute(
                    "MERGE (n:Component {canonical_id: $id}) "
                    "SET n.canonical_name = $name, n.aliases = $aliases, "
                    "n.mention_count = $mc, n.first_seen = $fs, n.last_seen = $ls, "
                    "n.mention_source = $ms",
                    parameters={
                        "id": ent["canonical_id"],
                        "name": ent["canonical_name"],
                        "aliases": ent.get("aliases", []),
                        "mc": mention_count,
                        "fs": _parse_ts(ent.get("first_seen")),
                        "ls": _parse_ts(ent.get("last_seen")),
                        "ms": mention_source,
                    },
                )
            elif table == "Parameter":
                self.conn.execute(
                    "MERGE (n:Parameter {canonical_id: $id}) "
                    "SET n.canonical_name = $name, n.value = $val, "
                    "n.aliases = $aliases, n.mention_count = $mc, "
                    "n.first_seen = $fs, n.last_seen = $ls",
                    parameters={
                        "id": ent["canonical_id"],
                        "name": ent["canonical_name"],
                        "val": attrs.get("value"),
                        "aliases": ent.get("aliases", []),
                        "mc": mention_count,
                        "fs": _parse_ts(ent.get("first_seen")),
                        "ls": _parse_ts(ent.get("last_seen")),
                    },
                )
            elif table == "FirmwareVersion":
                self.conn.execute(
                    "MERGE (n:FirmwareVersion {canonical_id: $id}) "
                    "SET n.canonical_name = $name, n.version_string = $vs, "
                    "n.vehicle_type = $vt, n.aliases = $aliases, "
                    "n.mention_count = $mc, n.first_seen = $fs, n.last_seen = $ls",
                    parameters={
                        "id": ent["canonical_id"],
                        "name": ent["canonical_name"],
                        "vs": attrs.get("version_string"),
                        "vt": attrs.get("vehicle_type"),
                        "aliases": ent.get("aliases", []),
                        "mc": mention_count,
                        "fs": _parse_ts(ent.get("first_seen")),
                        "ls": _parse_ts(ent.get("last_seen")),
                    },
                )
            elif table == "HardwarePlatform":
                self.conn.execute(
                    "MERGE (n:HardwarePlatform {canonical_id: $id}) "
                    "SET n.canonical_name = $name, n.aliases = $aliases, "
                    "n.mention_count = $mc, n.first_seen = $fs, n.last_seen = $ls, "
                    "n.mention_source = $ms",
                    parameters={
                        "id": ent["canonical_id"],
                        "name": ent["canonical_name"],
                        "aliases": ent.get("aliases", []),
                        "mc": mention_count,
                        "fs": _parse_ts(ent.get("first_seen")),
                        "ls": _parse_ts(ent.get("last_seen")),
                        "ms": mention_source,
                    },
                )

        logger.info("Loaded %d entities", len(entities))

    def load_claims(self) -> None:
        claims = self._read_json(self.data_dir / "canonical" / "canonical_claims.json")
        for cl in claims:
            n_ev = len(cl.get("evidence_ids", []))
            self.conn.execute(
                "MERGE (n:Claim {canonical_claim_id: $id}) "
                "SET n.predicate = $pred, n.object_freetext = $oft, "
                "n.object_freetext_variants = $oftv, n.confidence = $conf, "
                "n.support_level = $sl, n.is_current = $ic, "
                "n.valid_from = $vf, n.valid_until = $vu, "
                "n.assertion_time = $at, n.event_time = $et, "
                "n.superseded_by = $sb",
                parameters={
                    "id": cl["canonical_claim_id"],
                    "pred": cl.get("predicate"),
                    "oft": cl.get("object_freetext"),
                    "oftv": cl.get("object_freetext_variants", []),
                    "conf": cl.get("confidence"),
                    "sl": _support_level(n_ev),
                    "ic": cl.get("is_current"),
                    "vf": cl.get("valid_from"),
                    "vu": cl.get("valid_until"),
                    "at": _parse_ts(cl.get("assertion_time")),
                    "et": cl.get("event_time"),
                    "sb": cl.get("superseded_by"),
                },
            )
        logger.info("Loaded %d claims", len(claims))

    # ── relationship loaders ──────────────────────────────────────

    def load_subject_relationships(self) -> None:
        claims = self._read_json(self.data_dir / "canonical" / "canonical_claims.json")
        type_map = self._get_entity_type_map()
        count = 0
        for cl in claims:
            subj_id = cl.get("subject_canonical_id")
            if not subj_id:
                continue
            etype = type_map.get(subj_id)
            if not etype:
                logger.warning("Subject %s not found in entity map", subj_id)
                continue
            table = ENTITY_TYPE_TO_TABLE[etype]
            self.conn.execute(
                f"MATCH (e:{table} {{canonical_id: $eid}}), "
                f"(c:Claim {{canonical_claim_id: $cid}}) "
                f"MERGE (e)-[:{table}_HAS_SUBJECT]->(c)",
                parameters={"eid": subj_id, "cid": cl["canonical_claim_id"]},
            )
            count += 1
        logger.info("Created %d HAS_SUBJECT relationships", count)

    def load_object_relationships(self) -> None:
        claims = self._read_json(self.data_dir / "canonical" / "canonical_claims.json")
        type_map = self._get_entity_type_map()
        count = 0
        for cl in claims:
            obj_id = cl.get("object_canonical_id")
            if not obj_id:
                continue
            etype = type_map.get(obj_id)
            if not etype:
                logger.warning("Object %s not found in entity map", obj_id)
                continue
            table = ENTITY_TYPE_TO_TABLE[etype]
            self.conn.execute(
                f"MATCH (c:Claim {{canonical_claim_id: $cid}}), "
                f"(e:{table} {{canonical_id: $eid}}) "
                f"MERGE (c)-[:Claim_HAS_OBJECT_{table}]->(e)",
                parameters={"cid": cl["canonical_claim_id"], "eid": obj_id},
            )
            count += 1
        logger.info("Created %d HAS_OBJECT relationships", count)

    def load_supported_by(self) -> None:
        claims = self._read_json(self.data_dir / "canonical" / "canonical_claims.json")
        count = 0
        for cl in claims:
            for ev_id in cl.get("evidence_ids", []):
                self.conn.execute(
                    "MATCH (c:Claim {canonical_claim_id: $cid}), "
                    "(e:Evidence {evidence_id: $eid}) "
                    "MERGE (c)-[:SUPPORTED_BY]->(e)",
                    parameters={"cid": cl["canonical_claim_id"], "eid": ev_id},
                )
                count += 1
        logger.info("Created %d SUPPORTED_BY relationships", count)

    def load_sourced_from(self) -> None:
        records = self._read_jsonl(self.data_dir / "mentions" / "evidence_records.jsonl")
        count = 0
        for ev in records:
            self.conn.execute(
                "MATCH (e:Evidence {evidence_id: $eid}), "
                "(d:Document {doc_id: $did}) "
                "MERGE (e)-[:SOURCED_FROM]->(d)",
                parameters={"eid": ev["evidence_id"], "did": ev["doc_id"]},
            )
            count += 1
        logger.info("Created %d SOURCED_FROM relationships", count)

    def load_mentioned_in(self) -> None:
        mentions = self._read_jsonl(self.data_dir / "mentions" / "entity_mentions.jsonl")
        m2c = self._get_mention_to_canonical()
        type_map = self._get_entity_type_map()

        # deduplicate (canonical_id, doc_id) pairs
        seen: set[tuple[str, str]] = set()
        count = 0
        for m in mentions:
            canonical_id = m2c.get(m["mention_id"])
            if not canonical_id:
                continue
            doc_id = m["doc_id"]
            pair = (canonical_id, doc_id)
            if pair in seen:
                continue
            seen.add(pair)

            etype = type_map.get(canonical_id)
            if not etype:
                continue
            table = ENTITY_TYPE_TO_TABLE[etype]
            self.conn.execute(
                f"MATCH (e:{table} {{canonical_id: $eid}}), "
                f"(d:Document {{doc_id: $did}}) "
                f"MERGE (e)-[:{table}_MENTIONED_IN]->(d)",
                parameters={"eid": canonical_id, "did": doc_id},
            )
            count += 1
        logger.info("Created %d MENTIONED_IN relationships", count)

    def load_superseded_by(self) -> None:
        claims = self._read_json(self.data_dir / "canonical" / "canonical_claims.json")
        count = 0
        for cl in claims:
            target = cl.get("superseded_by")
            if not target:
                continue
            self.conn.execute(
                "MATCH (c1:Claim {canonical_claim_id: $from_id}), "
                "(c2:Claim {canonical_claim_id: $to_id}) "
                "MERGE (c1)-[:Claim_SUPERSEDED_BY]->(c2)",
                parameters={"from_id": cl["canonical_claim_id"], "to_id": target},
            )
            count += 1
        logger.info("Created %d SUPERSEDED_BY relationships", count)
