# Grounded Long-Term Memory via Structured Extraction, Deduplication, and a Context Graph

Layer10 take-home assignment. Builds a queryable memory system over a real-world corpus by extracting entities and claims from issue threads, resolving duplicates, and loading the result into a property graph with temporal validity tracking.

---

## What this is

Given a GitHub issue corpus (ArduPilot/ardupilot — 1,000 issues, 6,960 documents), the system:

1. Extracts entities (components, firmware versions, contributors, parameters, hardware) and structured claims (causes, fixes, regressions, aliases) from each document using a 3-pass LLM pipeline
2. Deduplicates entities using per-type structural resolvers and merges claims using per-predicate grouping keys
3. Loads the resolved data into an embedded Kuzu property graph with full grounding chains (claim → evidence → source document)
4. Retrieves relevant claims for natural-language questions using strategy-based Cypher traversal and LLM synthesis
5. Monitors source freshness — detecting when a GitHub document has been edited since extraction and propagating staleness to affected claims

ArduPilot is a good fit for this problem: components get renamed, bugs regress across firmware versions, the same issue surfaces in multiple threads months apart, and contributors refer to the same subsystem by different names ("EKF3", "Extended Kalman Filter 3", "AP_NavEKF3").

---

## Architecture

Six phases connected by JSONL files:

```
GitHub API
    │
    ▼
┌─────────────┐
│  Ingestion  │  fetch + normalize → data/processed/documents.jsonl
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Extraction │  3-pass LLM: entities → evidence → claims
│             │  → data/extractions/
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Resolution │  per-type entity dedup, per-predicate claim merge
│             │  → data/resolved/
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Graph    │  Kuzu embedded graph, idempotent MERGE loading
│             │  → data/graph/
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieval  │  strategy-based Cypher + LLM synthesis → ContextPack
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   Viz/UI    │  Streamlit: graph explorer, retrieval tab, merge inspector
└─────────────┘
```

Each stage reads from and writes to JSONL files. Any stage can be re-run independently.

---

## Corpus

- **Source**: ArduPilot/ardupilot GitHub Issues (1,000 issues)
- **Documents**: 6,960 — issue bodies, comments, and filtered timeline events
- **Why**: Component aliases, firmware-version regressions, cross-issue duplicates at scale — the properties that make naive extraction and deduplication fail

---

## Setup

**Prerequisites**: Python 3.12, pip

```bash
pip install -r requirements.txt
```

**Environment** (`.env` or exported):

| Variable | Required | Description |
|---|---|---|
| `LLM_API_KEY` | yes | API key for LLM gateway |
| `LLM_API_BASE` | yes | LiteLLM-compatible gateway URL |
| `LLM_MODEL` | yes | Model identifier (e.g. `claude-haiku-4-5`) |
| `USE_LOCAL_EMBEDDINGS` | no | Set to `1` to use local sentence-transformers instead of API embeddings |

All LLM calls go through `config.LLM_KWARGS` via litellm — swap the model by changing `LLM_MODEL`.

---

## Running end-to-end

```bash
# 1. Fetch raw issues (resumable — tracks last-fetched issue number)
python -m ingestion.fetch --repo ArduPilot/ardupilot --count 1000

# 2. Normalize into documents
python -m ingestion.preprocess
# → data/processed/documents.jsonl (6960 lines)

# 3. Extract entities, evidence, and claims (3 LLM passes per document)
python -m extraction.pipeline

# 4. Resolve duplicates and merge claims
python -m resolution.pipeline

# 5. Build the property graph
python -m graph.pipeline

# 6. Run retrieval demo (example questions against the graph)
python -m graph.demo

# 7. Launch the Streamlit visualization
streamlit run viz/app.py

# 8. Export graph snapshot for submission
python -m viz.export
# → data/outputs/graph_export.json
```

> **Development tip**: Always run with a document limit when testing changes. Full pipeline (6,960 docs × 3 LLM passes) takes 30+ minutes. Use `limit=10` to iterate in seconds.
>
> ```bash
> python -c "import extraction.pipeline as p; p.run(limit=10)"
> ```

---

## Project structure

| Module | Key files | Status |
|---|---|---|
| `ingestion/` | `preprocessor.py`, `raw_store.py` | complete |
| `extraction/` | `schema.py`, `tools.py`, `prompts.py`, `extractor.py`, `pipeline.py`, `validator.py` | complete |
| `resolution/` | `entity_resolver.py`, `merge_resolver.py`, `pipeline.py` | complete |
| `graph/` | `memory_graph.py`, `temporal.py`, `serializer.py` | complete |
| `retrieval/` | `embedder.py`, `retriever.py`, `formatter.py` | complete |
| `viz/` | `app.py`, `graph_renderer.py`, `export.py` | complete |
| `monitoring/` | `update_monitor.py` | complete |
| `config.py` | env vars, `LLM_KWARGS` dict | complete |

---

## Key design highlights

- **3-pass extraction**: Separate LLM passes for entities, evidence (verbatim excerpts with character offsets and source hashes), then claims. Focused schemas produce better results than a single "extract everything" prompt; the grounding chain (claim → evidence → source document) is preserved in the graph.

- **Per-type entity resolvers**: A `BaseEntityResolver` ABC with five subclasses — each encoding the correct identification rule for its domain (exact GitHub handle for persons, normalized name + structural rules for components, version string + vehicle type tuple for firmware). Fuzzy string matching was replaced after it caused systematic false merges on ArduPilot's compositional component names (EKF vs EKF3, SITL vs SITL Rover).

- **Per-predicate claim grouping**: Each predicate has a distinct grouping key that preserves the information that matters. `CAUSES` groups per symptom (a component can cause multiple distinct bugs), `DUPLICATE_OF` excludes the subject (multiple people noting the same duplicate produce one canonical claim), `REGRESSED_IN` groups per version. Followed by embedding-based freetext deduplication for near-synonym symptoms.

- **Strategy-based retrieval**: Seven predefined Cypher traversal patterns selected by LLM intent classification. Avoids fragile dynamic Cypher generation while covering the main question types (component deep dive, bug scan, parameter inventory, temporal queries, evidence chains, person ranking, cross-reference).

---

## Testing

```bash
python -m pytest tests/ -v
```

80+ unit tests covering all pipeline phases. All LLM calls are mocked — unit tests run in milliseconds with no API access required. Integration tests in `tests/integration/` run against the live API gateway.

Test coverage includes: schema round-trip serialization, 3-pass extraction pipeline, all 5 entity resolver classes, per-predicate claim grouping, temporal annotation, graph loading idempotency, retrieval scoring, `ContextPack` validation, and update monitor status propagation.

---

## Full write-up

Design rationale, architectural decisions with evidence, and worked examples are in [`writeup-draft.md`](writeup-draft.md).
