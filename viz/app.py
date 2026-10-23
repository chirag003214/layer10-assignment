"""Streamlit visualization app for the ArduPilot memory graph.

Three tabs:
  1. Graph View — interactive entity/claim graph via st_link_analysis
  2. Retrieval — natural language querying via MemoryRetriever
  3. Merge Inspector — browse entity/claim merge decisions
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ensure project root is on sys.path so `graph.*` imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st
from st_link_analysis import NodeStyle, EdgeStyle, st_link_analysis

# ── paths ────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CANONICAL_DIR = DATA_DIR / "canonical"
MENTIONS_DIR = DATA_DIR / "mentions"
GRAPH_DB_PATH = str(DATA_DIR / "graph" / "memory.kuzu")

# ── page config ──────────────────────────────────────────────────
st.set_page_config(page_title="ArduPilot Memory Graph", layout="wide")
st.title("ArduPilot Memory Graph Explorer")

# ── data loading ─────────────────────────────────────────────────


@st.cache_resource
def load_data():
    """Load all canonical data files once."""
    with open(CANONICAL_DIR / "canonical_entities.json") as f:
        entities = json.load(f)
    with open(CANONICAL_DIR / "canonical_claims.json") as f:
        claims = json.load(f)

    # evidence records keyed by evidence_id
    evidence_by_id = {}
    ev_path = MENTIONS_DIR / "evidence_records.jsonl"
    if ev_path.exists():
        with open(ev_path) as f:
            for line in f:
                rec = json.loads(line)
                evidence_by_id[rec["evidence_id"]] = rec

    # merge logs
    merge_log = []
    ml_path = CANONICAL_DIR / "merge_log.jsonl"
    if ml_path.exists():
        with open(ml_path) as f:
            merge_log = [json.loads(line) for line in f]

    freetext_merge_log = []
    ft_path = CANONICAL_DIR / "freetext_merge_log.jsonl"
    if ft_path.exists():
        with open(ft_path) as f:
            freetext_merge_log = [json.loads(line) for line in f]

    claim_merge_log = []
    cm_path = CANONICAL_DIR / "claim_merge_log.jsonl"
    if cm_path.exists():
        with open(cm_path) as f:
            claim_merge_log = [json.loads(line) for line in f]

    # mention_to_canonical
    m2c_path = CANONICAL_DIR / "mention_to_canonical.json"
    mention_to_canonical = {}
    if m2c_path.exists():
        with open(m2c_path) as f:
            mention_to_canonical = json.load(f)

    # lookup dicts
    entities_by_id = {e["canonical_id"]: e for e in entities}
    claims_by_subject = {}
    for c in claims:
        sid = c["subject_canonical_id"]
        claims_by_subject.setdefault(sid, []).append(c)

    return {
        "entities": entities,
        "claims": claims,
        "evidence_by_id": evidence_by_id,
        "merge_log": merge_log,
        "freetext_merge_log": freetext_merge_log,
        "claim_merge_log": claim_merge_log,
        "mention_to_canonical": mention_to_canonical,
        "entities_by_id": entities_by_id,
        "claims_by_subject": claims_by_subject,
    }


data = load_data()

# ── styling constants ────────────────────────────────────────────

ENTITY_TYPE_STYLES = {
    "person":            {"color": "#FF7F3E", "icon": "person"},
    "component":         {"color": "#2A629A", "icon": "dns"},
    "parameter":         {"color": "#9C27B0", "icon": "key"},
    "firmware_version":  {"color": "#4CAF50", "icon": "inventory"},
    "hardware_platform": {"color": "#FF5722", "icon": "router"},
}

PREDICATE_COLORS = {
    "causes":        "#E53935",
    "affects":       "#FB8C00",
    "fixed_in":      "#43A047",
    "regressed_in":  "#8E24AA",
    "has_parameter": "#1E88E5",
    "duplicate_of":  "#757575",
}

CLAIM_ICON = "description"
CLAIM_COLOR = "#78909C"


# ── graph builder ────────────────────────────────────────────────

def build_graph_elements(
    selected_types: list[str],
    selected_predicates: list[str],
    min_confidence: float,
    show_historical: bool,
    show_claims: bool,
) -> dict:
    """Build {nodes, edges} for st_link_analysis from canonical data."""
    nodes = []
    edges = []

    # filter claims
    visible_claims = []
    for c in data["claims"]:
        if c["predicate"] not in selected_predicates:
            continue
        conf = c.get("confidence") or 0
        if conf < min_confidence:
            continue
        if not show_historical and not c.get("is_current", True):
            continue
        visible_claims.append(c)

    # collect entity IDs referenced by visible claims
    referenced_ids = set()
    for c in visible_claims:
        referenced_ids.add(c["subject_canonical_id"])
        if c.get("object_canonical_id"):
            referenced_ids.add(c["object_canonical_id"])

    # filter entities: must be in selected types and referenced by visible claims
    visible_entities = {}
    for e in data["entities"]:
        if e["entity_type"] not in selected_types:
            continue
        if e["canonical_id"] not in referenced_ids:
            continue
        visible_entities[e["canonical_id"]] = e

    # build entity nodes
    for eid, e in visible_entities.items():
        nodes.append({
            "data": {
                "id": eid,
                "label": e["entity_type"],
                "name": e["canonical_name"],
                "mention_count": len(e.get("mention_ids", [])),
                "entity_type": e["entity_type"],
            }
        })

    # build claim nodes and edges
    for c in visible_claims:
        subj = c["subject_canonical_id"]
        obj = c.get("object_canonical_id")
        # skip if subject entity not visible
        if subj not in visible_entities:
            continue
        # skip if object entity required but not visible
        if obj and obj not in visible_entities:
            continue

        cid = c["canonical_claim_id"]
        conf = c.get("confidence") or 0

        if show_claims:
            # hypergraph: subject → claim_node → object/freetext
            nodes.append({
                "data": {
                    "id": cid,
                    "label": "claim",
                    "name": c["predicate"],
                    "predicate": c["predicate"],
                    "confidence": conf,
                    "is_current": c.get("is_current", True),
                }
            })
            edges.append({
                "data": {
                    "id": f"e_{cid}_subj",
                    "label": c["predicate"],
                    "source": subj,
                    "target": cid,
                }
            })
            if obj:
                edges.append({
                    "data": {
                        "id": f"e_{cid}_obj",
                        "label": c["predicate"],
                        "source": cid,
                        "target": obj,
                    }
                })
        else:
            # simple mode: direct edge from subject to object
            if obj:
                edges.append({
                    "data": {
                        "id": f"e_{cid}_direct",
                        "label": c["predicate"],
                        "source": subj,
                        "target": obj,
                    }
                })

    return {"nodes": nodes, "edges": edges}


def get_node_styles() -> list[NodeStyle]:
    """Build NodeStyle list for all entity types + claim type."""
    styles = []
    for etype, cfg in ENTITY_TYPE_STYLES.items():
        styles.append(NodeStyle(etype, cfg["color"], "name", cfg["icon"]))
    styles.append(NodeStyle("claim", CLAIM_COLOR, "name", CLAIM_ICON))
    return styles


def get_edge_styles() -> list[EdgeStyle]:
    """Build EdgeStyle list for all predicates."""
    styles = []
    for pred in PREDICATE_COLORS:
        styles.append(EdgeStyle(pred, directed=True, caption="label"))
    return styles


# ── cached retriever ──────────────────────────────────────────────


@st.cache_resource
def get_retriever():
    """Cached MemoryRetriever instance for situation panels and retrieval tab."""
    from graph.retrieval import MemoryRetriever
    return MemoryRetriever(db_path=GRAPH_DB_PATH)


# ── evidence rendering helper ────────────────────────────────────

def _render_evidence_section(evidence_list):
    """Render enriched evidence grouped by parent issue."""
    from graph.retrieval import MAINTAINER_HANDLES

    # Group by issue_number
    by_issue: dict[int | None, list] = {}
    for ev in evidence_list:
        key = ev.issue_number
        by_issue.setdefault(key, []).append(ev)

    seen_ids = set()
    for issue_num, evs in by_issue.items():
        # Issue header
        first = evs[0]
        if issue_num and first.issue_title:
            state_badge = f" ({first.issue_state})" if first.issue_state else ""
            st.markdown(f"**#{issue_num}** {first.issue_title}{state_badge}")
        for ev in evs:
            if ev.evidence_id in seen_ids:
                continue
            seen_ids.add(ev.evidence_id)
            author_display = f"**{ev.author}**" if ev.author_is_maintainer else (ev.author or "?")
            doc_type_tag = f" [{ev.doc_type}]" if ev.doc_type else ""
            ts_tag = f" {ev.timestamp}" if ev.timestamp else ""
            st.markdown(f"> {ev.excerpt}" if ev.excerpt else "> *(no excerpt)*")
            meta = f"*{author_display}{doc_type_tag}{ts_tag}*"
            if ev.source_url:
                meta += f" — [source]({ev.source_url})"
            st.markdown(meta)


# ── situation panel ──────────────────────────────────────────────

def show_situation_panel(node_id: str):
    """Show situation panel for entity nodes, simplified panel for claim nodes."""
    from graph.retrieval import SituationPack

    ent = data["entities_by_id"].get(node_id)
    if ent:
        retriever = get_retriever()
        situation = retriever.get_entity_situation(node_id)

        st.subheader(f"{ent['canonical_name']}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Type**: {ent['entity_type']}")
            st.write(f"**ID**: `{ent['canonical_id']}`")
            st.write(f"**Mentions**: {situation.entity.get('mention_count', 0)}")
        with col2:
            if ent.get("aliases"):
                st.write(f"**Aliases**: {', '.join(ent['aliases'])}")

        # Section 1: What is known
        st.markdown("---")
        st.markdown("### What is known")

        if situation.claims_as_subject:
            # Group by predicate
            by_pred: dict[str, list] = {}
            for claim in situation.claims_as_subject:
                by_pred.setdefault(claim.predicate, []).append(claim)

            current_claims = [c for c in situation.claims_as_subject if c.is_current]
            historical_claims = [c for c in situation.claims_as_subject if not c.is_current]

            for pred, pred_claims in by_pred.items():
                current_in_pred = [c for c in pred_claims if c.is_current]
                for claim in current_in_pred:
                    obj = claim.object_name or claim.object_freetext or ""
                    conf_bar = f"conf={claim.confidence:.2f}"
                    support = claim.support_level or "?"
                    st.markdown(
                        f"- `{claim.predicate}` {obj} "
                        f"({conf_bar}, {support})"
                    )

            if historical_claims:
                with st.expander(f"Historical beliefs ({len(historical_claims)})"):
                    for claim in historical_claims:
                        obj = claim.object_name or claim.object_freetext or ""
                        st.markdown(
                            f"- ~~`{claim.predicate}` {obj}~~ "
                            f"(conf={claim.confidence:.2f})"
                        )
        else:
            st.info("No claims found as subject.")

        if situation.claims_as_object:
            st.markdown("**Claims referencing this entity:**")
            for claim in situation.claims_as_object:
                subj = claim.subject_name or "?"
                st.markdown(
                    f"- {subj} `{claim.predicate}` this "
                    f"(conf={claim.confidence:.2f})"
                )

        # Section 2: Why we believe it
        st.markdown("---")
        st.markdown("### Why we believe it")

        all_evidence = []
        for claim in situation.claims_as_subject + situation.claims_as_object:
            all_evidence.extend(claim.evidence)

        if all_evidence:
            _render_evidence_section(all_evidence)
        else:
            st.info("No evidence records found.")

        # Section 3: Related
        st.markdown("---")
        st.markdown("### Related")

        if situation.related_entities:
            by_type: dict[str, list] = {}
            for rel in situation.related_entities:
                by_type.setdefault(rel.entity_type, []).append(rel)
            for etype, rels in by_type.items():
                names = ", ".join(r.canonical_name for r in rels)
                st.markdown(f"**{etype}**: {names}")

        if situation.mentioned_in_docs:
            with st.expander(f"Mentioned in {len(situation.mentioned_in_docs)} documents"):
                for doc in situation.mentioned_in_docs:
                    title = doc.get("title") or doc.get("doc_id", "?")
                    num = doc.get("issue_number")
                    prefix = f"#{num} " if num else ""
                    st.markdown(f"- {prefix}{title}")

        if situation.conflicts:
            st.markdown("---")
            st.markdown("### Conflicts")
            for conflict in situation.conflicts:
                st.warning(conflict.explanation)
                col_old, col_new = st.columns(2)
                with col_old:
                    old_text = conflict.older_claim.object_freetext or "?"
                    st.markdown(f"~~{conflict.older_claim.canonical_claim_id}: {old_text}~~")
                with col_new:
                    new_text = conflict.newer_claim.object_freetext or "?"
                    st.markdown(f"**{conflict.newer_claim.canonical_claim_id}: {new_text}**")
        return

    # check if it's a claim node
    for c in data["claims"]:
        if c["canonical_claim_id"] == node_id:
            subj = data["entities_by_id"].get(c["subject_canonical_id"], {})
            subj_name = subj.get("canonical_name", c["subject_canonical_id"])
            pred = c["predicate"]

            obj_name = None
            if c.get("object_canonical_id"):
                obj_ent = data["entities_by_id"].get(c["object_canonical_id"], {})
                obj_name = obj_ent.get("canonical_name", c["object_canonical_id"])
            freetext = c.get("object_freetext") or ""

            if obj_name and freetext:
                headline = f"{subj_name} **{pred}** {obj_name}: {freetext}"
            elif obj_name:
                headline = f"{subj_name} **{pred}** {obj_name}"
            elif freetext:
                headline = f"{subj_name} **{pred}**: {freetext}"
            else:
                headline = f"{subj_name} **{pred}**"

            st.subheader("Claim")
            st.markdown(headline)

            status = "current" if c.get("is_current", True) else "historical"
            conf = c.get("confidence", 0)
            st.write(
                f"Confidence: **{conf:.2f}** | Status: **{status}** | "
                f"ID: `{c['canonical_claim_id']}`"
            )
            if c.get("superseded_by"):
                st.write(f"**Superseded by**: `{c['superseded_by']}`")

            # evidence via retriever
            retriever = get_retriever()
            ev_map = retriever._fetch_evidence([c["canonical_claim_id"]])
            evidence = ev_map.get(c["canonical_claim_id"], [])
            if evidence:
                st.write(f"**Evidence** ({len(evidence)} records):")
                _render_evidence_section(evidence)
            return

    st.info(f"No details found for node `{node_id}`")


# ── tabs ─────────────────────────────────────────────────────────

tab_graph, tab_retrieval, tab_merge = st.tabs(
    ["Graph View", "Retrieval", "Merge Inspector"]
)

# ──────────────────────────────────────────────────────────────────
# TAB 1: Graph View
# ──────────────────────────────────────────────────────────────────

with tab_graph:
    all_types = sorted({e["entity_type"] for e in data["entities"]})
    all_predicates = sorted({c["predicate"] for c in data["claims"]})

    with st.sidebar:
        st.header("Graph Filters")
        sel_types = st.multiselect(
            "Entity types", all_types, default=all_types, key="graph_types"
        )
        sel_predicates = st.multiselect(
            "Predicates", all_predicates, default=all_predicates, key="graph_preds"
        )
        min_conf = st.slider(
            "Min confidence", 0.0, 1.0, 0.0, 0.05, key="graph_conf"
        )
        show_hist = st.checkbox("Show historical claims", value=True, key="graph_hist")
        show_claims_toggle = st.checkbox(
            "Show claim nodes (hypergraph)", value=True, key="graph_claims"
        )

    elements = build_graph_elements(
        sel_types, sel_predicates, min_conf, show_hist, show_claims_toggle
    )

    # summary metrics
    n_entity_nodes = sum(
        1 for n in elements["nodes"] if n["data"].get("label") != "claim"
    )
    n_claim_nodes = sum(
        1 for n in elements["nodes"] if n["data"].get("label") == "claim"
    )
    n_current = sum(
        1 for n in elements["nodes"]
        if n["data"].get("label") == "claim" and n["data"].get("is_current")
    )
    n_historical = n_claim_nodes - n_current
    avg_conf_vals = [
        n["data"]["confidence"]
        for n in elements["nodes"]
        if n["data"].get("label") == "claim" and n["data"].get("confidence") is not None
    ]
    avg_conf = sum(avg_conf_vals) / len(avg_conf_vals) if avg_conf_vals else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Visible Entities", n_entity_nodes)
    m2.metric("Visible Claims", n_claim_nodes)
    m3.metric("Current / Historical", f"{n_current} / {n_historical}")
    m4.metric("Avg Confidence", f"{avg_conf:.2f}")

    if elements["nodes"]:
        result = st_link_analysis(
            elements,
            layout="cose",
            node_styles=get_node_styles(),
            edge_styles=get_edge_styles(),
            height=600,
        )

        # show detail panel when a node is selected
        if result and isinstance(result, dict):
            selected = result.get("selected", {}).get("nodes", [])
            if selected:
                show_situation_panel(selected[0])
    else:
        st.info("No nodes match the current filters. Adjust filters in the sidebar.")


# ──────────────────────────────────────────────────────────────────
# TAB 2: Retrieval
# ──────────────────────────────────────────────────────────────────

with tab_retrieval:
    st.header("Query the Memory Graph")

    show_debug = st.checkbox("Show debug log", value=False, key="retrieval_debug")

    EXAMPLE_QUESTIONS = [
        "What's wrong with DDFP?",
        "What components have known bugs?",
        "What parameters does OSD5 have?",
        "Who are the active contributors?",
        "Why do we believe DDFP has this bug?",
        "Which issues involve AM32 ESCs?",
        "Has the DShot arming bug been fixed?",
    ]

    question = st.text_input("Ask a question about ArduPilot issues:", key="query_input")

    st.write("**Example questions:**")
    cols = st.columns(3)
    for i, eq in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 3].button(eq, key=f"eq_{i}"):
            question = eq

    if question:
        with st.spinner("Querying memory graph..."):
            try:
                import logging as _logging
                from graph.retrieval import MemoryRetriever

                # capture debug logs into a buffer
                _log_buffer = _logging.handlers.MemoryHandler(capacity=1000)
                _log_fmt = _logging.Formatter("%(levelname)s  %(message)s")
                _log_buffer.setFormatter(_log_fmt)
                _retrieval_logger = _logging.getLogger("graph.retrieval")
                _prev_level = _retrieval_logger.level
                if show_debug:
                    _retrieval_logger.setLevel(_logging.DEBUG)
                    _retrieval_logger.addHandler(_log_buffer)

                retriever = get_retriever()
                result = retriever.query(question)

                # flush and display debug log
                if show_debug:
                    _log_buffer.flush()
                    log_lines = []
                    for rec in _log_buffer.buffer:
                        log_lines.append(_log_fmt.format(rec))
                    _retrieval_logger.removeHandler(_log_buffer)
                    _retrieval_logger.setLevel(_prev_level)
                    if log_lines:
                        with st.expander("Debug Log", expanded=True):
                            st.code("\n".join(log_lines), language=None)

                st.success(
                    f"Strategy: **{result.strategy}** | "
                    f"Query time: {result.query_time_ms:.0f}ms | "
                    f"Claims found: {len(result.claims)}"
                )

                # Synthesized answer
                if result.synthesized_answer:
                    import re as _re
                    # Bold any [cc_...] claim ID patterns
                    answer_html = _re.sub(
                        r'\[(cc_[a-f0-9_]+)\]',
                        r'**[\1]**',
                        result.synthesized_answer,
                    )
                    st.info(answer_html)

                if result.matched_entities:
                    st.subheader("Matched Entities")
                    for ent in result.matched_entities:
                        cid = ent.get("canonical_id")
                        canonical = data["entities_by_id"].get(cid, {})
                        aliases = canonical.get("aliases", [])
                        attrs = canonical.get("attributes", {})
                        line = (
                            f"- **{ent.get('name', '?')}** "
                            f"({ent.get('type', '?')}, "
                            f"mentions: {ent.get('mention_count', 0)})"
                        )
                        if aliases:
                            line += f"  \n  Aliases: {', '.join(aliases)}"
                        # show useful attributes inline
                        attr_parts = []
                        for k, v in attrs.items():
                            if v is not None and v != "" and v is not False:
                                attr_parts.append(f"{k}={v}")
                        if attr_parts:
                            line += f"  \n  {', '.join(attr_parts)}"
                        st.markdown(line)

                if result.claims:
                    # Collapse sources when synthesized answer is shown
                    sources_container = st
                    if result.synthesized_answer:
                        sources_container = st.expander(
                            f"Sources ({len(result.claims)} claims)", expanded=False
                        )

                    with sources_container:
                        st.subheader("Ranked Claims")
                        for rank, claim in enumerate(result.claims, 1):
                            status = "current" if claim.is_current else "historical"
                            subj = claim.subject_name or "?"
                            obj_name = claim.object_name
                            freetext = claim.object_freetext or ""
                            pred = claim.predicate

                            if obj_name and freetext:
                                headline = f"{subj} **{pred}** {obj_name}: {freetext}"
                            elif obj_name:
                                headline = f"{subj} **{pred}** {obj_name}"
                            elif freetext:
                                headline = f"{subj} **{pred}**: {freetext}"
                            else:
                                headline = f"{subj} **{pred}**"

                            exp_obj = obj_name or freetext[:50]
                            with st.expander(
                                f"#{rank} {subj} {pred} {exp_obj} "
                                f"(score={claim.score:.2f}, {status})"
                            ):
                                st.markdown(headline)
                                st.write(
                                    f"Confidence: **{claim.confidence:.2f}** | "
                                    f"Support: **{claim.support_level or 'unknown'}** | "
                                    f"Status: **{status}** | "
                                    f"Score: **{claim.score:.2f}**"
                                )
                                st.write(f"ID: `{claim.canonical_claim_id}`")
                                if claim.valid_until:
                                    st.write(f"**Valid until**: {claim.valid_until}")
                                if claim.assertion_time:
                                    st.write(f"**Assertion time**: {claim.assertion_time}")
                                if claim.evidence:
                                    st.write(f"**Evidence** ({len(claim.evidence)} records):")
                                    _render_evidence_section(claim.evidence)
                else:
                    st.warning("No claims found for this query.")

            except Exception as exc:
                st.error(f"Query failed: {exc}")

# ──────────────────────────────────────────────────────────────────
# TAB 3: Merge Inspector
# ──────────────────────────────────────────────────────────────────

with tab_merge:
    st.header("Merge Inspector")

    merge_section = st.radio(
        "Section",
        ["Entity Merges", "Claim Dedup (Freetext)", "Claim Merges (Evidence)"],
        horizontal=True,
    )

    if merge_section == "Entity Merges":
        st.subheader("Entity Merge Decisions")
        merge_log = data["merge_log"]
        if not merge_log:
            st.info("No entity merge records found.")
        else:
            search = st.text_input("Search entities:", key="merge_search")
            strategies = sorted({m.get("strategy", "?") for m in merge_log})
            sel_strategy = st.multiselect(
                "Filter by strategy", strategies, default=strategies, key="merge_strat"
            )

            filtered = merge_log
            if search:
                search_lower = search.lower()
                filtered = [
                    m for m in filtered
                    if search_lower in m.get("canonical_name", "").lower()
                    or search_lower in m.get("canonical_id", "").lower()
                ]
            filtered = [m for m in filtered if m.get("strategy") in sel_strategy]

            st.write(f"Showing {len(filtered)} of {len(merge_log)} merge records")

            for m in filtered[:50]:
                n_mentions = len(m.get("merged_mention_ids", []))
                with st.expander(
                    f"{m.get('canonical_name', '?')} — "
                    f"{m.get('strategy', '?')} — "
                    f"{n_mentions} mentions"
                ):
                    st.write(f"**Canonical ID**: `{m.get('canonical_id')}`")
                    st.write(f"**Strategy**: {m.get('strategy')}")
                    if m.get("similarity_score") is not None:
                        st.write(f"**Similarity**: {m['similarity_score']:.4f}")
                    st.write(f"**Decided at**: {m.get('decided_at')}")
                    st.write(f"**Merged mentions**: {m.get('merged_mention_ids', [])}")

                    # show the entity details if available
                    ent = data["entities_by_id"].get(m.get("canonical_id"))
                    if ent:
                        if ent.get("aliases"):
                            st.write(f"**Aliases**: {', '.join(ent['aliases'])}")
                        st.write(f"**Type**: {ent['entity_type']}")

    elif merge_section == "Claim Dedup (Freetext)":
        st.subheader("Freetext Claim Dedup Decisions")
        ft_log = data["freetext_merge_log"]
        if not ft_log:
            st.info("No freetext merge records found.")
        else:
            st.write(f"{len(ft_log)} freetext dedup records")
            for rec in ft_log[:50]:
                merged = rec.get("merged_claim_ids", [])
                with st.expander(
                    f"{rec.get('predicate', '?')} — "
                    f"{rec.get('subject_name', '?')} — "
                    f"score={rec.get('similarity_score', 0):.3f} — "
                    f"{rec.get('strategy', '?')}"
                ):
                    st.write(f"**Canonical claim**: `{rec.get('canonical_claim_id')}`")
                    st.write(f"**Merged claims**: {merged}")
                    st.write(f"**Strategy**: {rec.get('strategy')}")
                    st.write(f"**Similarity**: {rec.get('similarity_score')}")
                    if rec.get("llm_decision"):
                        st.write(f"**LLM decision**: {rec['llm_decision']}")
                    if rec.get("llm_reasoning"):
                        st.write(f"**LLM reasoning**: {rec['llm_reasoning']}")
                    if rec.get("llm_confidence"):
                        st.write(f"**LLM confidence**: {rec['llm_confidence']}")

    else:  # Claim Merges (Evidence)
        st.subheader("Claim Evidence Merge Log")
        cm_log = data["claim_merge_log"]
        if not cm_log:
            st.info("No claim merge records found.")
        else:
            st.write(f"{len(cm_log)} claim merge records")
            for rec in cm_log[:50]:
                with st.expander(
                    f"{rec.get('predicate', '?')} — "
                    f"`{rec.get('canonical_claim_id', '?')}` — "
                    f"{rec.get('evidence_count', 0)} evidence"
                ):
                    st.write(f"**Canonical claim**: `{rec.get('canonical_claim_id')}`")
                    st.write(f"**Predicate**: {rec.get('predicate')}")
                    st.write(f"**Merged docs**: {rec.get('merged_doc_ids', [])}")
                    st.write(f"**Evidence count**: {rec.get('evidence_count')}")
