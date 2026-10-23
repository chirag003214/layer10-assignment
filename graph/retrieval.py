from __future__ import annotations

"""Retrieval layer: natural language questions → ranked claims with evidence.

Accepts a question, matches entities in the Kuzu graph, selects a query
strategy, and returns a scored ContextPack.
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any

import kuzu
import litellm
from pydantic import BaseModel

import config
from graph.schema import ENTITY_TYPES

logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────

MAINTAINER_HANDLES = {"rmackay9", "bnsgeyer", "IamPete1", "tridge", "peterbarker", "andyp1per"}

JUNK_NAMES = {"<unknown>", "unknown", "", "n/a", "none"}

# ── Pydantic output models ────────────────────────────────────────


class EvidenceSnippet(BaseModel):
    evidence_id: str
    excerpt: str | None = None
    author: str | None = None
    timestamp: str | None = None
    doc_id: str | None = None
    source_url: str | None = None
    source_status: str | None = None


class EnrichedEvidence(BaseModel):
    evidence_id: str
    excerpt: str | None = None
    author: str | None = None
    timestamp: str | None = None
    doc_id: str | None = None
    source_url: str | None = None
    source_status: str | None = None
    issue_title: str | None = None
    issue_number: int | None = None
    issue_state: str | None = None
    doc_type: str | None = None
    author_is_maintainer: bool = False


class RankedClaim(BaseModel):
    canonical_claim_id: str
    predicate: str
    subject_name: str | None = None
    object_name: str | None = None
    object_freetext: str | None = None
    confidence: float
    support_level: str | None = None
    is_current: bool = True
    valid_until: str | None = None
    assertion_time: str | None = None
    score: float
    evidence: list[EnrichedEvidence] = []
    superseded_by: str | None = None


class Conflict(BaseModel):
    older_claim: RankedClaim
    newer_claim: RankedClaim
    conflict_type: str  # "superseded" | "version_disagreement"
    explanation: str


class RelatedEntity(BaseModel):
    canonical_id: str
    canonical_name: str
    entity_type: str
    relationship: str
    direction: str  # "subject" or "object"


class SituationPack(BaseModel):
    entity: dict[str, Any]
    claims_as_subject: list[RankedClaim]
    claims_as_object: list[RankedClaim]
    related_entities: list[RelatedEntity]
    mentioned_in_docs: list[dict[str, Any]]
    timeline: list[RankedClaim]
    conflicts: list[Conflict] = []


class ContextPack(BaseModel):
    question: str
    strategy: str
    matched_entities: list[dict[str, Any]]
    claims: list[RankedClaim]
    generated_at: str
    query_time_ms: float
    synthesized_answer: str | None = None


class QueryIntent(BaseModel):
    strategy: str
    entity_names: list[str] = []
    temporal_filter: str | None = None
    confidence_filter: float | None = None


# ── scoring constants ─────────────────────────────────────────────

SUPPORT_LEVEL_MAP = {
    "overwhelming": 1.0,
    "strong": 0.75,
    "single": 0.4,
    "unsupported": 0.0,
}

# ── LLM tool for entity extraction from questions ─────────────────

_ENTITY_EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "identify_entities",
        "description": (
            "Extract entity names and types mentioned in a user question. "
            "Return the specific technical names the user is asking about."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The entity name as mentioned in the question",
                            },
                            "type": {
                                "type": "string",
                                "enum": ["person", "component", "parameter",
                                         "firmware_version", "hardware_platform"],
                                "description": "The entity type",
                            },
                        },
                        "required": ["name", "type"],
                    },
                },
            },
            "required": ["entities"],
        },
    },
}


_INTENT_CLASSIFY_TOOL = {
    "type": "function",
    "function": {
        "name": "classify_query",
        "description": (
            "Classify a user question about ArduPilot issues into a retrieval strategy "
            "and extract entity names mentioned."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "strategy": {
                    "type": "string",
                    "enum": [
                        "component_deep_dive", "component_bug_scan",
                        "parameter_inventory", "person_ranking",
                        "evidence_chain", "entity_issue_crossref",
                        "temporal_query",
                    ],
                    "description": (
                        "component_deep_dive: all claims about a specific component; "
                        "component_bug_scan: scan for bugs across components; "
                        "parameter_inventory: parameters of a component; "
                        "person_ranking: who are contributors; "
                        "evidence_chain: why/cause questions about a component; "
                        "entity_issue_crossref: which issues mention an entity; "
                        "temporal_query: fixed/regressed/version timeline questions"
                    ),
                },
                "entity_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Entity names mentioned in the question",
                },
                "temporal_filter": {
                    "type": "string",
                    "description": "Version number if the question asks about a specific version",
                },
                "confidence_filter": {
                    "type": "number",
                    "description": "Minimum confidence filter if implied by the question",
                },
            },
            "required": ["strategy", "entity_names"],
        },
    },
}


class MemoryRetriever:
    """Query the Kuzu memory graph with natural language questions."""

    def __init__(self, db_path: str) -> None:
        self.db = kuzu.Database(db_path)
        self.conn = kuzu.Connection(self.db)
        self._doc_cache: dict[str, dict] | None = None

    # ── scoring ────────────────────────────────────────────────────

    def _recency(self, assertion_time: str | None) -> float:
        """Score recency: None → 0.5, recent → higher, old → lower."""
        if assertion_time is None:
            return 0.5
        try:
            ts = assertion_time.replace("Z", "+00:00")
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            days = (datetime.now(timezone.utc) - dt).days
            return max(0.0, min(1.0, 1.0 - (days / 365)))
        except (ValueError, TypeError):
            return 0.5

    def _score_claim(self, claim: dict) -> float:
        """Weighted score: confidence*0.4 + is_current*0.3 + support*0.2 + recency*0.1."""
        conf = float(claim.get("confidence", 0) or 0)
        current = 1.0 if claim.get("is_current") else 0.0
        support = SUPPORT_LEVEL_MAP.get(claim.get("support_level", ""), 0.0)
        recency = self._recency(claim.get("assertion_time"))
        return conf * 0.4 + current * 0.3 + support * 0.2 + recency * 0.1

    # ── entity matching ────────────────────────────────────────────

    def _match_entities(self, question: str) -> list[dict]:
        """Two-stage entity matching: LLM extraction → graph name search."""
        # Stage 1: LLM extracts entity names from the question
        extracted = self._llm_extract_entities(question)
        logger.debug("Entity extraction produced %d candidates", len(extracted))

        # Stage 2: search graph for each extracted name
        seen: dict[str, dict] = {}
        for ent in extracted:
            term = ent["name"]
            matches = self._search_entity_tables(term)
            for m in matches:
                cid = m["canonical_id"]
                if cid not in seen:
                    seen[cid] = m

        # Sort by mention_count desc, return top 5
        entities = sorted(seen.values(), key=lambda e: e.get("mention_count", 0), reverse=True)
        entities = entities[:5]
        logger.debug("Matched %d unique entities (top 5): %s",
                     len(entities), [(e["name"], e["type"], e["mention_count"]) for e in entities])
        return entities

    def _llm_extract_entities(self, question: str) -> list[dict]:
        """Use LLM tool call to extract entity names/types from a question."""
        logger.debug("LLM entity extraction for: %r", question)
        try:
            resp = litellm.completion(
                **config.LLM_KWARGS,
                messages=[
                    {"role": "system", "content": (
                        "You are an entity extractor for ArduPilot issue queries. "
                        "Extract the specific entity names the user is asking about. "
                        "Return them using the identify_entities tool."
                    )},
                    {"role": "user", "content": question},
                ],
                tools=[_ENTITY_EXTRACT_TOOL],
                tool_choice={"type": "function", "function": {"name": "identify_entities"}},
            )
            tool_calls = resp.choices[0].message.tool_calls
            if tool_calls:
                args = json.loads(tool_calls[0].function.arguments)
                entities = args.get("entities", [])
                logger.debug("LLM extracted %d entities: %s", len(entities), entities)
                return entities
            logger.debug("LLM returned no tool calls")
        except Exception:
            logger.warning("LLM entity extraction failed, falling back to empty", exc_info=True)
        return []

    def _search_entity_tables(self, term: str) -> list[dict]:
        """Search all entity tables for names or aliases containing the term."""
        logger.debug("Searching graph for term: %r", term)
        results = []
        seen_ids: set[str] = set()
        term_lower = term.lower()
        for table in ENTITY_TYPES:
            # Search by canonical_name
            try:
                query = (
                    f"MATCH (n:{table}) "
                    f"WHERE lower(n.canonical_name) CONTAINS lower($term) "
                    f"RETURN n.canonical_id, n.canonical_name, '{table}' AS entity_type, "
                    f"n.mention_count"
                )
                result = self.conn.execute(query, parameters={"term": term})
                while result.has_next():
                    row = result.get_next()
                    if row[0] not in seen_ids:
                        seen_ids.add(row[0])
                        results.append({
                            "canonical_id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "mention_count": row[3] or 0,
                        })
            except Exception:
                logger.debug("Name search failed for table %s term %s", table, term, exc_info=True)

            # Search by aliases (separate query to avoid Kuzu parameter bug)
            try:
                alias_query = (
                    f"MATCH (n:{table}) "
                    f"UNWIND n.aliases AS a "
                    f"WITH n, a WHERE lower(a) CONTAINS lower($term) "
                    f"RETURN DISTINCT n.canonical_id, n.canonical_name, '{table}' AS entity_type, "
                    f"n.mention_count"
                )
                result = self.conn.execute(alias_query, parameters={"term": term})
                while result.has_next():
                    row = result.get_next()
                    if row[0] not in seen_ids:
                        seen_ids.add(row[0])
                        results.append({
                            "canonical_id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "mention_count": row[3] or 0,
                        })
            except Exception:
                logger.debug("Alias search failed for table %s term %s", table, term, exc_info=True)
        logger.debug("Found %d matches for %r: %s", len(results), term,
                     [(r["name"], r["type"]) for r in results])
        return results

    def _match_entities_by_names(self, names: list[str]) -> list[dict]:
        """Match pre-extracted entity names against the graph (no LLM call)."""
        names = [n for n in names if n.lower().strip() not in JUNK_NAMES]
        seen: dict[str, dict] = {}
        for term in names:
            matches = self._search_entity_tables(term)
            for m in matches:
                cid = m["canonical_id"]
                if cid not in seen:
                    seen[cid] = m
        entities = sorted(seen.values(), key=lambda e: e.get("mention_count", 0), reverse=True)
        return entities[:5]

    # ── parent entity resolution ────────────────────────────────

    def _resolve_parent_entities(self, entities: list[dict]) -> list[dict]:
        """When only Parameters are matched, walk backwards to find owning Components."""
        has_component = any(
            e.get("type") in ("Component", "HardwarePlatform") for e in entities
        )
        if has_component:
            return entities

        param_ids = [e["canonical_id"] for e in entities if e.get("type") == "Parameter"]
        if not param_ids:
            return entities

        seen_comp_ids: set[str] = {e["canonical_id"] for e in entities}
        resolved = list(entities)
        for pid in param_ids:
            try:
                result = self.conn.execute(
                    "MATCH (c:Claim)-[:Claim_HAS_OBJECT_Parameter]->(p:Parameter {canonical_id: $pid}) "
                    "MATCH (comp:Component)-[:Component_HAS_SUBJECT]->(c) "
                    "WHERE c.predicate = 'has_parameter' "
                    "RETURN comp.canonical_id, comp.canonical_name, 'Component' AS t, comp.mention_count",
                    parameters={"pid": pid},
                )
                while result.has_next():
                    row = result.get_next()
                    if row[0] not in seen_comp_ids:
                        seen_comp_ids.add(row[0])
                        resolved.append({
                            "canonical_id": row[0],
                            "name": row[1],
                            "type": row[2],
                            "mention_count": row[3] or 0,
                        })
            except Exception:
                logger.debug("Parent resolution failed for %s", pid, exc_info=True)
        return resolved

    # ── intent classification ─────────────────────────────────────

    def classify_intent(self, question: str) -> QueryIntent:
        """Classify question into a QueryIntent using LLM with fallback."""
        try:
            resp = litellm.completion(
                **config.LLM_KWARGS,
                messages=[
                    {"role": "system", "content": (
                        "Classify this ArduPilot issue query into a retrieval strategy "
                        "and extract entity names. Use the classify_query tool."
                    )},
                    {"role": "user", "content": question},
                ],
                tools=[_INTENT_CLASSIFY_TOOL],
                tool_choice={"type": "function", "function": {"name": "classify_query"}},
                timeout=10,
            )
            tool_calls = resp.choices[0].message.tool_calls
            if tool_calls:
                args = json.loads(tool_calls[0].function.arguments)
                raw_names = args.get("entity_names", [])
                filtered_names = [n for n in raw_names if n.lower().strip() not in JUNK_NAMES]
                return QueryIntent(
                    strategy=args.get("strategy", "component_bug_scan"),
                    entity_names=filtered_names,
                    temporal_filter=args.get("temporal_filter"),
                    confidence_filter=args.get("confidence_filter"),
                )
        except Exception:
            logger.warning("Intent classification failed, falling back to keyword", exc_info=True)

        # Fallback: keyword-based strategy + empty entity list
        strategy = self._select_strategy(question, [])
        return QueryIntent(strategy=strategy, entity_names=[])

    # ── document cache ────────────────────────────────────────────

    def _ensure_doc_cache(self) -> dict[str, dict]:
        """Lazily load doc_id → {title, issue_number, state} mapping."""
        if self._doc_cache is not None:
            return self._doc_cache
        self._doc_cache = {}
        try:
            result = self.conn.execute(
                "MATCH (d:Document) "
                "RETURN d.doc_id, d.title, d.issue_number, d.state"
            )
            while result.has_next():
                row = result.get_next()
                self._doc_cache[row[0]] = {
                    "title": row[1],
                    "issue_number": row[2],
                    "state": row[3],
                }
        except Exception:
            logger.debug("Doc cache load failed", exc_info=True)
        return self._doc_cache

    @staticmethod
    def _parent_issue_doc_id(doc_id: str) -> str | None:
        """Extract parent issue doc_id from comment/timeline doc_ids."""
        for sep in ("/c/", "/t/"):
            if sep in doc_id:
                return doc_id.split(sep)[0]
        return None

    # ── evidence fetching ──────────────────────────────────────────

    def _fetch_evidence(self, claim_ids: list[str]) -> dict[str, list[EnrichedEvidence]]:
        """Fetch evidence for a list of claim IDs via the graph."""
        evidence_map: dict[str, list[EnrichedEvidence]] = {}
        doc_cache = self._ensure_doc_cache()

        for cid in claim_ids:
            snippets: list[EnrichedEvidence] = []
            try:
                result = self.conn.execute(
                    "MATCH (c:Claim {canonical_claim_id: $cid})"
                    "-[:SUPPORTED_BY]->(e:Evidence)"
                    "-[:SOURCED_FROM]->(d:Document) "
                    "RETURN e.evidence_id, e.excerpt, e.author, "
                    "e.timestamp, d.doc_id, d.source_url, e.source_status, "
                    "d.doc_type, d.issue_number, d.title, d.state",
                    parameters={"cid": cid},
                )
                while result.has_next():
                    row = result.get_next()
                    ts = row[3]
                    if isinstance(ts, datetime):
                        ts = ts.isoformat()
                    author = row[2]
                    doc_id = row[4]
                    doc_type = row[7]
                    issue_number = row[8]
                    issue_title = row[9]
                    issue_state = row[10]

                    # For comment/timeline docs, get parent issue context
                    if doc_type and doc_type != "issue" and doc_id:
                        parent_id = self._parent_issue_doc_id(doc_id)
                        if parent_id and parent_id in doc_cache:
                            parent = doc_cache[parent_id]
                            issue_title = issue_title or parent.get("title")
                            issue_number = issue_number or parent.get("issue_number")
                            issue_state = issue_state or parent.get("state")

                    snippets.append(EnrichedEvidence(
                        evidence_id=row[0],
                        excerpt=row[1],
                        author=author,
                        timestamp=str(ts) if ts else None,
                        doc_id=doc_id,
                        source_url=row[5],
                        source_status=row[6],
                        doc_type=doc_type,
                        issue_number=issue_number,
                        issue_title=issue_title,
                        issue_state=issue_state,
                        author_is_maintainer=(author or "") in MAINTAINER_HANDLES,
                    ))
            except Exception:
                logger.debug("Evidence fetch failed for %s", cid, exc_info=True)
            evidence_map[cid] = snippets
        return evidence_map

    # ── entity situation ──────────────────────────────────────────

    def _find_entity_node(self, canonical_id: str) -> dict | None:
        """Look up an entity node by canonical_id across all entity type tables."""
        for table in ENTITY_TYPES:
            try:
                result = self.conn.execute(
                    f"MATCH (n:{table} {{canonical_id: $cid}}) "
                    f"RETURN n.canonical_id, n.canonical_name, '{table}' AS entity_type, "
                    f"n.aliases, n.mention_count",
                    parameters={"cid": canonical_id},
                )
                if result.has_next():
                    row = result.get_next()
                    return {
                        "canonical_id": row[0],
                        "canonical_name": row[1],
                        "entity_type": row[2],
                        "aliases": row[3] or [],
                        "mention_count": row[4] or 0,
                    }
            except Exception:
                continue
        return None

    def _claims_as_role(
        self, canonical_id: str, table: str, role: str,
    ) -> list[dict]:
        """Fetch claims where entity is subject or object.

        role: "subject" or "object"
        """
        claims: list[dict] = []
        if role == "subject":
            # entity → HAS_SUBJECT → Claim, optionally Claim → HAS_OBJECT → other entity
            try:
                result = self.conn.execute(
                    f"MATCH (e:{table} {{canonical_id: $eid}})"
                    f"-[:{table}_HAS_SUBJECT]->(c:Claim) "
                    f"RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                    f"c.confidence, c.support_level, c.is_current, c.valid_until, "
                    f"c.assertion_time, e.canonical_name, c.superseded_by",
                    parameters={"eid": canonical_id},
                )
                while result.has_next():
                    row = result.get_next()
                    at = row[7]
                    if isinstance(at, datetime):
                        at = at.isoformat()
                    claims.append({
                        "canonical_claim_id": row[0],
                        "predicate": row[1],
                        "object_freetext": row[2],
                        "confidence": row[3] or 0,
                        "support_level": row[4],
                        "is_current": row[5],
                        "valid_until": row[6],
                        "assertion_time": str(at) if at else None,
                        "subject_name": row[8],
                        "object_name": None,
                        "superseded_by": row[9],
                    })
            except Exception:
                logger.debug("Claims as subject failed for %s.%s", table, canonical_id, exc_info=True)

            # Try to resolve object entity names
            for claim in claims:
                cid = claim["canonical_claim_id"]
                for obj_table in ENTITY_TYPES:
                    try:
                        result = self.conn.execute(
                            f"MATCH (c:Claim {{canonical_claim_id: $cid}})"
                            f"-[:Claim_HAS_OBJECT_{obj_table}]->(o:{obj_table}) "
                            f"RETURN o.canonical_name",
                            parameters={"cid": cid},
                        )
                        if result.has_next():
                            claim["object_name"] = result.get_next()[0]
                            break
                    except Exception:
                        continue

        else:  # role == "object"
            for subj_table in ENTITY_TYPES:
                try:
                    result = self.conn.execute(
                        f"MATCH (s:{subj_table})-[:{subj_table}_HAS_SUBJECT]->(c:Claim)"
                        f"-[:Claim_HAS_OBJECT_{table}]->(e:{table} {{canonical_id: $eid}}) "
                        f"RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                        f"c.confidence, c.support_level, c.is_current, c.valid_until, "
                        f"c.assertion_time, s.canonical_name, e.canonical_name, c.superseded_by",
                        parameters={"eid": canonical_id},
                    )
                    while result.has_next():
                        row = result.get_next()
                        at = row[7]
                        if isinstance(at, datetime):
                            at = at.isoformat()
                        claims.append({
                            "canonical_claim_id": row[0],
                            "predicate": row[1],
                            "object_freetext": row[2],
                            "confidence": row[3] or 0,
                            "support_level": row[4],
                            "is_current": row[5],
                            "valid_until": row[6],
                            "assertion_time": str(at) if at else None,
                            "subject_name": row[8],
                            "object_name": row[9],
                            "superseded_by": row[10],
                        })
                except Exception:
                    logger.debug("Claims as object failed for %s→%s.%s",
                                 subj_table, table, canonical_id, exc_info=True)
        return claims

    def get_entity_situation(self, canonical_id: str) -> SituationPack:
        """Build a full situation pack for a given entity."""
        # Find the entity node
        entity = self._find_entity_node(canonical_id)
        if not entity:
            return SituationPack(
                entity={"canonical_id": canonical_id, "canonical_name": "Unknown"},
                claims_as_subject=[], claims_as_object=[],
                related_entities=[], mentioned_in_docs=[],
                timeline=[], conflicts=[],
            )

        table = entity["entity_type"]

        # 1. Claims as subject
        raw_subj = self._claims_as_role(canonical_id, table, "subject")
        for c in raw_subj:
            c["score"] = self._score_claim(c)

        # 2. Claims as object
        raw_obj = self._claims_as_role(canonical_id, table, "object")
        for c in raw_obj:
            c["score"] = self._score_claim(c)

        # 3. Mentioned-in docs
        mentioned_docs: list[dict] = []
        try:
            result = self.conn.execute(
                f"MATCH (e:{table} {{canonical_id: $eid}})"
                f"-[:{table}_MENTIONED_IN]->(d:Document) "
                f"RETURN d.doc_id, d.title, d.issue_number, d.created_at "
                f"ORDER BY d.created_at DESC LIMIT 10",
                parameters={"eid": canonical_id},
            )
            while result.has_next():
                row = result.get_next()
                ca = row[3]
                if isinstance(ca, datetime):
                    ca = ca.isoformat()
                mentioned_docs.append({
                    "doc_id": row[0], "title": row[1],
                    "issue_number": row[2], "created_at": str(ca) if ca else None,
                })
        except Exception:
            logger.debug("Mentioned-in query failed for %s", canonical_id, exc_info=True)

        # 4. Related entities (2-hop via claims)
        related: dict[str, RelatedEntity] = {}
        all_claim_ids = [c["canonical_claim_id"] for c in raw_subj + raw_obj]
        for cid_val in all_claim_ids:
            # For each claim, find the other entity connected
            for other_table in ENTITY_TYPES:
                # Check object side
                try:
                    result = self.conn.execute(
                        f"MATCH (c:Claim {{canonical_claim_id: $cid}})"
                        f"-[:Claim_HAS_OBJECT_{other_table}]->(o:{other_table}) "
                        f"WHERE o.canonical_id <> $eid "
                        f"RETURN o.canonical_id, o.canonical_name, '{other_table}', c.predicate",
                        parameters={"cid": cid_val, "eid": canonical_id},
                    )
                    while result.has_next():
                        row = result.get_next()
                        if row[0] not in related:
                            related[row[0]] = RelatedEntity(
                                canonical_id=row[0], canonical_name=row[1],
                                entity_type=row[2], relationship=row[3],
                                direction="object",
                            )
                except Exception:
                    continue
                # Check subject side
                try:
                    result = self.conn.execute(
                        f"MATCH (s:{other_table})-[:{other_table}_HAS_SUBJECT]->"
                        f"(c:Claim {{canonical_claim_id: $cid}}) "
                        f"WHERE s.canonical_id <> $eid "
                        f"RETURN s.canonical_id, s.canonical_name, '{other_table}', c.predicate",
                        parameters={"cid": cid_val, "eid": canonical_id},
                    )
                    while result.has_next():
                        row = result.get_next()
                        if row[0] not in related:
                            related[row[0]] = RelatedEntity(
                                canonical_id=row[0], canonical_name=row[1],
                                entity_type=row[2], relationship=row[3],
                                direction="subject",
                            )
                except Exception:
                    continue

        # Fetch evidence for all claims
        all_cids = [c["canonical_claim_id"] for c in raw_subj + raw_obj]
        evidence_map = self._fetch_evidence(all_cids) if all_cids else {}

        # Build RankedClaims
        def _to_ranked(raw_list: list[dict]) -> list[RankedClaim]:
            ranked = []
            for c in sorted(raw_list, key=lambda x: x["score"], reverse=True):
                cid = c["canonical_claim_id"]
                ranked.append(RankedClaim(
                    canonical_claim_id=cid,
                    predicate=c["predicate"],
                    subject_name=c.get("subject_name"),
                    object_name=c.get("object_name"),
                    object_freetext=c.get("object_freetext"),
                    confidence=c["confidence"],
                    support_level=c.get("support_level"),
                    is_current=c.get("is_current", True),
                    valid_until=c.get("valid_until"),
                    assertion_time=c.get("assertion_time"),
                    score=c["score"],
                    evidence=evidence_map.get(cid, []),
                ))
            return ranked

        claims_as_subject = _to_ranked(raw_subj)
        claims_as_object = _to_ranked(raw_obj)

        # Build timeline: all claims sorted by assertion_time asc
        all_ranked = claims_as_subject + claims_as_object
        timeline = sorted(
            all_ranked,
            key=lambda c: c.assertion_time or "",
        )

        # Detect conflicts
        conflicts = self.detect_conflicts(claims_as_subject)

        return SituationPack(
            entity=entity,
            claims_as_subject=claims_as_subject,
            claims_as_object=claims_as_object,
            related_entities=list(related.values()),
            mentioned_in_docs=mentioned_docs,
            timeline=timeline,
            conflicts=conflicts,
        )

    # ── synthesis ──────────────────────────────────────────────────

    def synthesize_answer(
        self, question: str, claims: list[RankedClaim],
    ) -> str | None:
        """Use LLM to synthesize a natural language answer from ranked claims."""
        if not claims:
            return None

        # Build structured context from top 5 claims
        context_lines = []
        for i, c in enumerate(claims[:5], 1):
            current = "CURRENT" if c.is_current else "HISTORICAL"
            excerpt = ""
            if c.evidence:
                excerpt = (c.evidence[0].excerpt or "")[:200]
            context_lines.append(
                f"[{i}] claim_id={c.canonical_claim_id}\n"
                f"    {c.subject_name} {c.predicate} {c.object_name or c.object_freetext or ''}\n"
                f"    confidence={c.confidence:.2f} support={c.support_level} {current}\n"
                f"    assertion_time={c.assertion_time or 'unknown'}\n"
                f"    evidence: \"{excerpt}\""
            )

        context = "\n".join(context_lines)

        # Add conflict info if any superseded claims exist
        superseded = [c for c in claims[:5] if c.superseded_by]
        if superseded:
            conflict_lines = []
            for c in superseded:
                conflict_lines.append(
                    f"  - {c.canonical_claim_id} was superseded by {c.superseded_by}"
                )
            context += "\n\nConflicts (updated beliefs):\n" + "\n".join(conflict_lines)

        try:
            resp = litellm.completion(
                **config.LLM_KWARGS,
                messages=[
                    {"role": "system", "content": (
                        "You are a memory assistant for ArduPilot project issues. "
                        "Answer the user's question based ONLY on the provided claims and evidence. "
                        "Cite claims using [claim_id] format. "
                        "If a claim is HISTORICAL (no longer current), note that it has been updated. "
                        "If there are conflicts between claims, acknowledge them. "
                        "Keep your answer to 3-5 sentences. Do not invent information."
                    )},
                    {"role": "user", "content": (
                        f"Question: {question}\n\n"
                        f"Available claims:\n{context}"
                    )},
                ],
            )
            answer = resp.choices[0].message.content
            if answer:
                return answer.strip()
        except Exception:
            logger.warning("Synthesis failed", exc_info=True)
        return None

    # ── conflict detection ─────────────────────────────────────────

    def detect_conflicts(self, claims: list[RankedClaim]) -> list[Conflict]:
        """Detect conflicts among claims (superseded, version disagreements)."""
        conflicts: list[Conflict] = []
        claims_by_id = {c.canonical_claim_id: c for c in claims}

        for claim in claims:
            if not claim.superseded_by:
                continue

            # Check if the superseding claim is already in the list
            newer = claims_by_id.get(claim.superseded_by)
            if not newer:
                # Try to fetch from Kuzu
                try:
                    result = self.conn.execute(
                        "MATCH (c:Claim {canonical_claim_id: $cid}) "
                        "RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                        "c.confidence, c.support_level, c.is_current, c.valid_until, "
                        "c.assertion_time, c.superseded_by",
                        parameters={"cid": claim.superseded_by},
                    )
                    if result.has_next():
                        row = result.get_next()
                        at = row[7]
                        if isinstance(at, datetime):
                            at = at.isoformat()
                        newer = RankedClaim(
                            canonical_claim_id=row[0],
                            predicate=row[1],
                            object_freetext=row[2],
                            confidence=row[3] or 0,
                            support_level=row[4],
                            is_current=row[5] if row[5] is not None else True,
                            valid_until=row[6],
                            assertion_time=str(at) if at else None,
                            score=0,
                            superseded_by=row[8],
                        )
                except Exception:
                    logger.debug("Failed to fetch superseding claim %s", claim.superseded_by, exc_info=True)
                    continue

            if newer:
                conflicts.append(Conflict(
                    older_claim=claim,
                    newer_claim=newer,
                    conflict_type="superseded",
                    explanation=(
                        f"Claim {claim.canonical_claim_id} "
                        f"({claim.predicate}: {claim.object_freetext or '?'}) "
                        f"was superseded by {newer.canonical_claim_id} "
                        f"({newer.predicate}: {newer.object_freetext or '?'})"
                    ),
                ))

        return conflicts

    # ── strategy selection ─────────────────────────────────────────

    def _select_strategy(self, question: str, entities: list[dict]) -> str:
        """Select query strategy based on keyword rules."""
        q = question.lower()
        has_component = any(e.get("type") in ("Component", "HardwarePlatform") for e in entities)

        # Temporal queries
        if any(kw in q for kw in ["fixed", "regressed", "version", "timeline", "when was"]):
            return "temporal_query"

        # Person-oriented queries
        if any(kw in q for kw in ["who ", "contributor", "author", "developer", "person"]):
            return "person_ranking"

        # Parameter-oriented queries
        if any(kw in q for kw in ["parameter", "param ", "setting", "config"]):
            return "parameter_inventory"

        # Issue cross-reference (check before bug keywords since "issue" overlaps)
        if any(kw in q for kw in ["mention", "reference", "document"]):
            return "entity_issue_crossref"

        # Cause/evidence chain queries
        if any(kw in q for kw in ["cause", "why", "root cause", "evidence", "proof"]):
            if has_component:
                return "evidence_chain"
            return "component_bug_scan"

        # Bug/issue-oriented queries
        if any(kw in q for kw in ["bug", "wrong", "issue", "problem", "fail", "crash", "error"]):
            if has_component:
                return "component_deep_dive"
            return "component_bug_scan"

        # Default: if we have component entities, deep dive; otherwise bug scan
        if has_component:
            return "component_deep_dive"
        return "component_bug_scan"

    # ── query strategies ───────────────────────────────────────────

    def _strategy_component_deep_dive(self, entities: list[dict]) -> list[dict]:
        """All claims where a matched component is the subject."""
        claims = []
        for ent in entities:
            if ent.get("type") not in ("Component", "HardwarePlatform"):
                continue
            table = ent["type"]
            try:
                result = self.conn.execute(
                    f"MATCH (e:{table} {{canonical_id: $eid}})"
                    f"-[:{table}_HAS_SUBJECT]->(c:Claim) "
                    f"OPTIONAL MATCH (c)-[:Claim_HAS_OBJECT_Parameter]->(p:Parameter) "
                    f"RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                    f"c.object_freetext_variants, c.confidence, c.support_level, "
                    f"c.is_current, c.valid_until, c.assertion_time, "
                    f"e.canonical_name, p.canonical_name",
                    parameters={"eid": ent["canonical_id"]},
                )
                while result.has_next():
                    row = result.get_next()
                    at = row[8]
                    if isinstance(at, datetime):
                        at = at.isoformat()
                    claims.append({
                        "canonical_claim_id": row[0],
                        "predicate": row[1],
                        "object_freetext": row[2],
                        "confidence": row[4] or 0,
                        "support_level": row[5],
                        "is_current": row[6],
                        "valid_until": row[7],
                        "assertion_time": str(at) if at else None,
                        "subject_name": row[9],
                        "object_name": row[10],
                    })
            except Exception:
                logger.debug("Deep dive failed for %s", ent["canonical_id"], exc_info=True)
        return claims

    def _strategy_component_bug_scan(self, entities: list[dict]) -> list[dict]:
        """Scan for AFFECTS/CAUSES current claims, grouped by component."""
        claims = []
        entity_ids = [
            e for e in entities
            if e.get("type") in ("Component", "HardwarePlatform")
        ]

        if entity_ids:
            # Scoped: only bugs for matched entities
            for ent in entity_ids:
                table = ent["type"]
                try:
                    result = self.conn.execute(
                        f"MATCH (comp:{table} {{canonical_id: $eid}})"
                        f"-[:{table}_HAS_SUBJECT]->(c:Claim) "
                        f"WHERE c.is_current = true "
                        f"AND (c.predicate = 'affects' OR c.predicate = 'causes') "
                        f"RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                        f"c.confidence, c.support_level, c.is_current, c.valid_until, "
                        f"c.assertion_time, comp.canonical_name "
                        f"ORDER BY c.confidence DESC LIMIT 30",
                        parameters={"eid": ent["canonical_id"]},
                    )
                    while result.has_next():
                        row = result.get_next()
                        at = row[7]
                        if isinstance(at, datetime):
                            at = at.isoformat()
                        claims.append({
                            "canonical_claim_id": row[0],
                            "predicate": row[1],
                            "object_freetext": row[2],
                            "confidence": row[3] or 0,
                            "support_level": row[4],
                            "is_current": row[5],
                            "valid_until": row[6],
                            "assertion_time": str(at) if at else None,
                            "subject_name": row[8],
                            "object_name": None,
                        })
                except Exception:
                    logger.debug("Bug scan failed for %s", ent["canonical_id"], exc_info=True)
        else:
            # Global scan: all components
            try:
                result = self.conn.execute(
                    "MATCH (comp:Component)-[:Component_HAS_SUBJECT]->(c:Claim) "
                    "WHERE c.is_current = true AND (c.predicate = 'affects' OR c.predicate = 'causes') "
                    "RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                    "c.confidence, c.support_level, c.is_current, c.valid_until, "
                    "c.assertion_time, comp.canonical_name "
                    "ORDER BY c.confidence DESC LIMIT 30"
                )
                while result.has_next():
                    row = result.get_next()
                    at = row[7]
                    if isinstance(at, datetime):
                        at = at.isoformat()
                    claims.append({
                        "canonical_claim_id": row[0],
                        "predicate": row[1],
                        "object_freetext": row[2],
                        "confidence": row[3] or 0,
                        "support_level": row[4],
                        "is_current": row[5],
                        "valid_until": row[6],
                        "assertion_time": str(at) if at else None,
                        "subject_name": row[8],
                        "object_name": None,
                    })
            except Exception:
                logger.debug("Bug scan failed", exc_info=True)
        return claims

    def _strategy_parameter_inventory(self, entities: list[dict]) -> list[dict]:
        """HAS_PARAMETER claims with Parameter object join."""
        claims = []
        entity_ids = [e["canonical_id"] for e in entities if e.get("type") in ("Component", "HardwarePlatform")]

        if entity_ids:
            # Scoped to matched entities
            for eid in entity_ids:
                try:
                    result = self.conn.execute(
                        "MATCH (comp)-[:Component_HAS_SUBJECT]->(c:Claim)"
                        "-[:Claim_HAS_OBJECT_Parameter]->(p:Parameter) "
                        "WHERE comp.canonical_id = $eid AND c.predicate = 'has_parameter' "
                        "RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                        "c.confidence, c.support_level, c.is_current, c.valid_until, "
                        "c.assertion_time, comp.canonical_name, p.canonical_name",
                        parameters={"eid": eid},
                    )
                    while result.has_next():
                        row = result.get_next()
                        at = row[7]
                        if isinstance(at, datetime):
                            at = at.isoformat()
                        claims.append({
                            "canonical_claim_id": row[0],
                            "predicate": row[1],
                            "object_freetext": row[2],
                            "confidence": row[3] or 0,
                            "support_level": row[4],
                            "is_current": row[5],
                            "valid_until": row[6],
                            "assertion_time": str(at) if at else None,
                            "subject_name": row[8],
                            "object_name": row[9],
                        })
                except Exception:
                    logger.debug("Parameter inventory failed for %s", eid, exc_info=True)
        else:
            # Global: all HAS_PARAMETER claims
            try:
                result = self.conn.execute(
                    "MATCH (comp:Component)-[:Component_HAS_SUBJECT]->(c:Claim)"
                    "-[:Claim_HAS_OBJECT_Parameter]->(p:Parameter) "
                    "WHERE c.predicate = 'has_parameter' "
                    "RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                    "c.confidence, c.support_level, c.is_current, c.valid_until, "
                    "c.assertion_time, comp.canonical_name, p.canonical_name "
                    "ORDER BY c.confidence DESC LIMIT 20"
                )
                while result.has_next():
                    row = result.get_next()
                    at = row[7]
                    if isinstance(at, datetime):
                        at = at.isoformat()
                    claims.append({
                        "canonical_claim_id": row[0],
                        "predicate": row[1],
                        "object_freetext": row[2],
                        "confidence": row[3] or 0,
                        "support_level": row[4],
                        "is_current": row[5],
                        "valid_until": row[6],
                        "assertion_time": str(at) if at else None,
                        "subject_name": row[8],
                        "object_name": row[9],
                    })
            except Exception:
                logger.debug("Parameter inventory global scan failed", exc_info=True)
        return claims

    def _strategy_person_ranking(self, entities: list[dict]) -> list[dict]:
        """Person nodes by mention_count, their mentioned documents."""
        claims = []
        try:
            result = self.conn.execute(
                "MATCH (p:Person) "
                "OPTIONAL MATCH (p)-[:Person_MENTIONED_IN]->(d:Document) "
                "RETURN p.canonical_id, p.canonical_name, p.github_handle, "
                "p.mention_count, collect(d.doc_id) AS doc_ids "
                "ORDER BY p.mention_count DESC LIMIT 10"
            )
            while result.has_next():
                row = result.get_next()
                # Package person info as a pseudo-claim for uniform output
                claims.append({
                    "canonical_claim_id": f"person_{row[0]}",
                    "predicate": "person_info",
                    "object_freetext": f"handle=@{row[2]}, docs={len(row[4] or [])}",
                    "confidence": 1.0,
                    "support_level": "strong",
                    "is_current": True,
                    "valid_until": None,
                    "assertion_time": None,
                    "subject_name": row[1],
                    "object_name": None,
                })
        except Exception:
            logger.debug("Person ranking failed", exc_info=True)
        return claims

    def _strategy_evidence_chain(self, entities: list[dict]) -> list[dict]:
        """CAUSES claims for matched entities with full evidence provenance."""
        ENTITY_TYPES_SET = set(ENTITY_TYPES)
        claims = []
        for ent in entities:
            table = ent.get("type")
            if table not in ENTITY_TYPES_SET:
                continue
            try:
                result = self.conn.execute(
                    f"MATCH (e:{table} {{canonical_id: $eid}})"
                    f"-[:{table}_HAS_SUBJECT]->(c:Claim) "
                    f"WHERE c.predicate = 'causes' "
                    f"RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                    f"c.confidence, c.support_level, c.is_current, c.valid_until, "
                    f"c.assertion_time, e.canonical_name",
                    parameters={"eid": ent["canonical_id"]},
                )
                while result.has_next():
                    row = result.get_next()
                    at = row[7]
                    if isinstance(at, datetime):
                        at = at.isoformat()
                    claims.append({
                        "canonical_claim_id": row[0],
                        "predicate": row[1],
                        "object_freetext": row[2],
                        "confidence": row[3] or 0,
                        "support_level": row[4],
                        "is_current": row[5],
                        "valid_until": row[6],
                        "assertion_time": str(at) if at else None,
                        "subject_name": row[8],
                        "object_name": None,
                    })
            except Exception:
                logger.debug("Evidence chain failed for %s", ent["canonical_id"], exc_info=True)
        return claims

    def _strategy_entity_issue_crossref(self, entities: list[dict]) -> list[dict]:
        """Documents via MENTIONED_IN + claims involving the entity."""
        claims = []
        for ent in entities:
            table = ent.get("type")
            if table not in ENTITY_TYPES:
                continue
            try:
                # Get claims where entity is subject
                result = self.conn.execute(
                    f"MATCH (e:{table} {{canonical_id: $eid}})"
                    f"-[:{table}_HAS_SUBJECT]->(c:Claim) "
                    f"RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                    f"c.confidence, c.support_level, c.is_current, c.valid_until, "
                    f"c.assertion_time, e.canonical_name",
                    parameters={"eid": ent["canonical_id"]},
                )
                while result.has_next():
                    row = result.get_next()
                    at = row[7]
                    if isinstance(at, datetime):
                        at = at.isoformat()
                    claims.append({
                        "canonical_claim_id": row[0],
                        "predicate": row[1],
                        "object_freetext": row[2],
                        "confidence": row[3] or 0,
                        "support_level": row[4],
                        "is_current": row[5],
                        "valid_until": row[6],
                        "assertion_time": str(at) if at else None,
                        "subject_name": row[8],
                        "object_name": None,
                    })
            except Exception:
                logger.debug("Issue crossref failed for %s", ent["canonical_id"], exc_info=True)

            # Also get documents where entity is mentioned
            try:
                result = self.conn.execute(
                    f"MATCH (e:{table} {{canonical_id: $eid}})"
                    f"-[:{table}_MENTIONED_IN]->(d:Document) "
                    f"RETURN d.doc_id, d.title, d.source_url "
                    f"LIMIT 10",
                    parameters={"eid": ent["canonical_id"]},
                )
                while result.has_next():
                    row = result.get_next()
                    claims.append({
                        "canonical_claim_id": f"doc_{row[0]}",
                        "predicate": "mentioned_in",
                        "object_freetext": f"title={row[1]}, url={row[2]}",
                        "confidence": 1.0,
                        "support_level": "strong",
                        "is_current": True,
                        "valid_until": None,
                        "assertion_time": None,
                        "subject_name": ent.get("name"),
                        "object_name": None,
                    })
            except Exception:
                logger.debug("Mentioned-in crossref failed for %s", ent["canonical_id"], exc_info=True)
        return claims

    def _strategy_temporal_query(
        self, entities: list[dict], intent: QueryIntent | None = None,
    ) -> list[dict]:
        """Temporal strategy: version-specific or general fixed_in/regressed_in claims."""
        claims: list[dict] = []
        version = intent.temporal_filter if intent else None

        if version:
            # Version-specific: find claims connected to FirmwareVersion matching version
            try:
                result = self.conn.execute(
                    "MATCH (comp)-[]->(c:Claim)"
                    "-[:Claim_HAS_OBJECT_FirmwareVersion]->(v:FirmwareVersion) "
                    "WHERE v.canonical_name CONTAINS $version "
                    "AND c.predicate IN ['fixed_in', 'regressed_in'] "
                    "RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                    "c.confidence, c.support_level, c.is_current, c.valid_until, "
                    "c.assertion_time, comp.canonical_name, v.canonical_name",
                    parameters={"version": version},
                )
                while result.has_next():
                    row = result.get_next()
                    at = row[7]
                    if isinstance(at, datetime):
                        at = at.isoformat()
                    claims.append({
                        "canonical_claim_id": row[0],
                        "predicate": row[1],
                        "object_freetext": row[2],
                        "confidence": row[3] or 0,
                        "support_level": row[4],
                        "is_current": row[5],
                        "valid_until": row[6],
                        "assertion_time": str(at) if at else None,
                        "subject_name": row[8],
                        "object_name": row[9],
                    })
            except Exception:
                logger.debug("Temporal version query failed for %s", version, exc_info=True)
        else:
            # Scoped or global temporal: fixed_in and regressed_in claims
            entity_ids = [
                e for e in entities
                if e.get("type") in ("Component", "HardwarePlatform")
            ]

            if entity_ids:
                # Scoped to matched entities
                for ent in entity_ids:
                    table = ent["type"]
                    for pred in ("fixed_in", "regressed_in"):
                        try:
                            result = self.conn.execute(
                                f"MATCH (comp:{table} {{canonical_id: $eid}})"
                                f"-[:{table}_HAS_SUBJECT]->(c:Claim) "
                                f"WHERE c.predicate = $pred AND c.is_current = true "
                                f"RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                                f"c.confidence, c.support_level, c.is_current, c.valid_until, "
                                f"c.assertion_time, comp.canonical_name",
                                parameters={"eid": ent["canonical_id"], "pred": pred},
                            )
                            while result.has_next():
                                row = result.get_next()
                                at = row[7]
                                if isinstance(at, datetime):
                                    at = at.isoformat()
                                claims.append({
                                    "canonical_claim_id": row[0],
                                    "predicate": row[1],
                                    "object_freetext": row[2],
                                    "confidence": row[3] or 0,
                                    "support_level": row[4],
                                    "is_current": row[5],
                                    "valid_until": row[6],
                                    "assertion_time": str(at) if at else None,
                                    "subject_name": row[8],
                                    "object_name": None,
                                })
                        except Exception:
                            logger.debug("Temporal %s scan failed for %s", pred, ent["canonical_id"], exc_info=True)
            else:
                # Global scan
                for pred in ("fixed_in", "regressed_in"):
                    try:
                        result = self.conn.execute(
                            f"MATCH (comp)-[]->(c:Claim) "
                            f"WHERE c.predicate = $pred AND c.is_current = true "
                            f"RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                            f"c.confidence, c.support_level, c.is_current, c.valid_until, "
                            f"c.assertion_time, comp.canonical_name",
                            parameters={"pred": pred},
                        )
                        while result.has_next():
                            row = result.get_next()
                            at = row[7]
                            if isinstance(at, datetime):
                                at = at.isoformat()
                            claims.append({
                                "canonical_claim_id": row[0],
                                "predicate": row[1],
                                "object_freetext": row[2],
                                "confidence": row[3] or 0,
                                "support_level": row[4],
                                "is_current": row[5],
                                "valid_until": row[6],
                                "assertion_time": str(at) if at else None,
                                "subject_name": row[8],
                                "object_name": None,
                            })
                    except Exception:
                        logger.debug("Temporal %s scan failed", pred, exc_info=True)

            # Also fetch superseded claims
            try:
                result = self.conn.execute(
                    "MATCH (comp)-[]->(c:Claim) "
                    "WHERE c.superseded_by IS NOT NULL "
                    "RETURN c.canonical_claim_id, c.predicate, c.object_freetext, "
                    "c.confidence, c.support_level, c.is_current, c.valid_until, "
                    "c.assertion_time, comp.canonical_name"
                )
                while result.has_next():
                    row = result.get_next()
                    at = row[7]
                    if isinstance(at, datetime):
                        at = at.isoformat()
                    claims.append({
                        "canonical_claim_id": row[0],
                        "predicate": row[1],
                        "object_freetext": row[2],
                        "confidence": row[3] or 0,
                        "support_level": row[4],
                        "is_current": row[5],
                        "valid_until": row[6],
                        "assertion_time": str(at) if at else None,
                        "subject_name": row[8],
                        "object_name": None,
                    })
            except Exception:
                logger.debug("Temporal superseded scan failed", exc_info=True)

        # Sort by assertion_time desc
        claims.sort(key=lambda c: c.get("assertion_time") or "", reverse=True)
        return claims

    # ── main query ─────────────────────────────────────────────────

    def query(self, question: str) -> ContextPack:
        """Answer a natural language question using the memory graph."""
        t0 = time.time()
        logger.debug("=== Query start: %r ===", question)

        # 1. Classify intent (strategy + entity names in one LLM call)
        intent = self.classify_intent(question)
        logger.debug("Intent: strategy=%s entities=%s temporal=%s",
                      intent.strategy, intent.entity_names, intent.temporal_filter)

        # 2. Match entities from intent names (or fallback to full extraction)
        if intent.entity_names:
            entities = self._match_entities_by_names(intent.entity_names)
        else:
            entities = self._match_entities(question)

        # 2b. Resolve parent entities (Parameter → Component)
        entities = self._resolve_parent_entities(entities)

        # 3. Select strategy: use intent, but validate it exists
        strategy_fn = {
            "component_deep_dive": self._strategy_component_deep_dive,
            "component_bug_scan": self._strategy_component_bug_scan,
            "parameter_inventory": self._strategy_parameter_inventory,
            "person_ranking": self._strategy_person_ranking,
            "evidence_chain": self._strategy_evidence_chain,
            "entity_issue_crossref": self._strategy_entity_issue_crossref,
        }
        strategy = intent.strategy
        if strategy != "temporal_query" and strategy not in strategy_fn:
            strategy = self._select_strategy(question, entities)
        logger.info("Strategy: %s | Entities: %s", strategy, [e.get("name") for e in entities])

        # 4. Run strategy
        if strategy == "temporal_query":
            raw_claims = self._strategy_temporal_query(entities, intent)
        else:
            raw_claims = strategy_fn[strategy](entities)
        logger.debug("Strategy %s returned %d raw claims", strategy, len(raw_claims))

        # 4a. Deduplicate claims by canonical_claim_id
        seen_ids: set[str] = set()
        deduped: list[dict] = []
        for c in raw_claims:
            cid = c["canonical_claim_id"]
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped.append(c)
        raw_claims = deduped

        # 4b. Score and sort
        for claim in raw_claims:
            claim["score"] = self._score_claim(claim)
        raw_claims.sort(key=lambda c: c["score"], reverse=True)

        if raw_claims:
            logger.debug("Score range: %.3f – %.3f", raw_claims[-1]["score"], raw_claims[0]["score"])
            for c in raw_claims[:5]:
                logger.debug("  %.3f  %s | %s → %s",
                             c["score"], c["predicate"],
                             c.get("subject_name"), (c.get("object_freetext") or "")[:60])

        # 5. Truncate to top 10
        raw_claims = raw_claims[:10]

        # 6. Fetch evidence for top claims
        claim_ids = [
            c["canonical_claim_id"] for c in raw_claims
            if not c["canonical_claim_id"].startswith(("person_", "doc_"))
        ]
        evidence_map = self._fetch_evidence(claim_ids) if claim_ids else {}
        total_ev = sum(len(v) for v in evidence_map.values())
        logger.debug("Fetched %d evidence records for %d claims", total_ev, len(claim_ids))

        # 7. Build ranked claims
        ranked = []
        for c in raw_claims:
            cid = c["canonical_claim_id"]
            ranked.append(RankedClaim(
                canonical_claim_id=cid,
                predicate=c["predicate"],
                subject_name=c.get("subject_name"),
                object_name=c.get("object_name"),
                object_freetext=c.get("object_freetext"),
                confidence=c["confidence"],
                support_level=c.get("support_level"),
                is_current=c.get("is_current", True),
                valid_until=c.get("valid_until"),
                assertion_time=c.get("assertion_time"),
                score=c["score"],
                evidence=evidence_map.get(cid, []),
            ))

        # 8. Synthesize answer
        synthesized = self.synthesize_answer(question, ranked)

        elapsed = (time.time() - t0) * 1000

        return ContextPack(
            question=question,
            strategy=strategy,
            matched_entities=entities,
            claims=ranked,
            generated_at=datetime.now(timezone.utc).isoformat(),
            query_time_ms=elapsed,
            synthesized_answer=synthesized,
        )
