# Grounded Long-Term Memory via Structured Extraction, Deduplication, and a Context Graph

---

## 1. Corpus & Reproduction

**Corpus**: 1,000 issues from [ArduPilot/ardupilot](https://github.com/ArduPilot/ardupilot) on GitHub, fetched via the GitHub REST API and normalized into **6,960 documents** — a mix of issue bodies, comments, and timeline events.

ArduPilot is a good fit for this task because it has the properties that make long-term memory hard: components get renamed, bugs regress across firmware versions, the same problem surfaces in different issues months apart, and contributors refer to the same subsystem by different names ("EKF3", "Extended Kalman Filter 3", "AP_NavEKF3"). It's a corpus where naive extraction produces plausible-looking garbage and naive deduplication merges things that shouldn't be merged.

**To reproduce:**

```bash
# Fetch raw issues (resumable — stores last-fetched issue number)
python -m ingestion.fetch --repo ArduPilot/ardupilot --count 1000

# Normalize into documents
python -m ingestion.preprocess
# Output: data/processed/documents.jsonl (6960 lines)
```

The fetcher handles GitHub rate limits with backoff and writes each issue atomically (temp file → rename) so a crash mid-fetch doesn't corrupt data. Timeline events are filtered to semantic types only — `labeled`, `assigned`, `renamed`, `milestoned`, `referenced` — discarding noise like `subscribed` and `mentioned` that carry no information about the issue itself.

---

## 2. Ontology & Schema Design

The schema captures five entity types and six predicates. These aren't arbitrary — they map to the concepts that actually recur in ArduPilot issue discussions and the relationships that matter for answering questions about what's broken, where, and since when.

### Entity types

| Type | What it represents | Canonical identifier |
|---|---|---|
| `PERSON` | GitHub user (contributor, reporter, maintainer) | `github_handle` (lowercased) |
| `COMPONENT` | ArduPilot subsystem or module (EKF3, GPS, DShot) | Normalized name after resolution |
| `PARAMETER` | Configuration parameter (AHRS_EKF_TYPE, TKOFF_RPM_MIN) | Uppercased, underscores normalized |
| `FIRMWARE_VERSION` | A released version, often tied to bugs/fixes | `(version_string, vehicle_type)` |
| `HARDWARE_PLATFORM` | Physical board or frame (CubeOrange, Pixhawk) | Normalized name after resolution |

The choice of five types is deliberate. Fewer types (say, collapsing COMPONENT and HARDWARE_PLATFORM) would lose the distinction between software modules and physical boards — a distinction that matters when asking "what hardware is affected by this bug?" More types (say, splitting COMPONENT into library/driver/mode) would create extraction ambiguity that the LLM handles poorly without extensive few-shot examples.

### Predicates

| Predicate | Subject type | Object type | What it captures |
|---|---|---|---|
| `CAUSES` | Component/Hardware/Firmware | freetext (symptom) | Bug or failure mode |
| `AFFECTS` | Component/Hardware/Firmware | optional | Component involvement without specific mechanism |
| `FIXED_IN` | Component/Hardware | FirmwareVersion | Bug resolution |
| `REGRESSED_IN` | Component/Hardware/Firmware | FirmwareVersion | Bug reintroduction |
| `HAS_PARAMETER` | Component/Hardware | Parameter | Configuration relationship |
| `DUPLICATE_OF` | Person | freetext (issue number) | Duplicate declaration by a specific person |

`AFFECTS` deserves a note: it's an intentional low-precision fallback. When the LLM can see that a component is involved in a bug discussion but can't pin down the specific causal mechanism, `AFFECTS` captures the involvement without overcommitting. A `CAUSES` claim asserts a specific failure mode; `AFFECTS` says "this component is relevant here." In a production system, singleton `AFFECTS` claims would be routed to a more capable model for refinement.

### ID generation

All IDs are deterministic SHA1 hashes with type prefixes (`m_` for mentions, `ev_` for evidence, `cm_` for claims). This makes the pipeline idempotent — re-running extraction on the same document produces the same IDs, which means downstream resolution and graph loading can safely re-merge without creating duplicates. Content-addressability also means two different documents producing the same claim about the same entities will naturally converge.

### Extensibility

Adding a new entity type means: define it in the `EntityType` enum, add a keying rule to the entity resolver (one class), add a node table to the graph schema (one line in a list comprehension), and update the extraction prompt. No existing data or code paths change. Same for predicates: add to the enum, define a grouping key in `ClaimResolver._make_group_key`, and update the claim extraction prompt.

---

## 3. Extraction System

### 3a. Three-pass LLM architecture

Each document goes through three sequential LLM passes, each building on the previous:

**Pass 1 (Entities)** extracts entity mentions with types, surface forms, and attributes. **Pass 2 (Evidence)** extracts verbatim excerpts from the document that ground each entity — exact text, character offsets, source hash. **Pass 3 (Claims)** extracts structured subject-predicate-object triples, referencing the entities and evidence from the previous passes.

The separation matters. A single "extract everything" prompt produces worse results because the LLM tries to do entity recognition, evidence selection, and relationship extraction simultaneously, leading to hallucinated evidence and malformed claims. Three focused passes create a grounding chain: claims reference evidence, evidence references entities, and each pass has a simpler output schema that the LLM follows more reliably.

The cost is 3× LLM calls per document. On a corpus of 6,960 documents, that's roughly 20,000 calls — manageable with Claude Haiku and batching.

#### The author rule: a problem found and fixed

Short comments like *"I believe this is a duplicate of #17383"* contain no technical entities — no component, parameter, or firmware version. Without explicit instruction, the LLM returns an empty entity array, which triggers an early-exit gate in the extractor (`if not entity_mentions: return`) that skips passes 2 and 3 entirely. This meant `DUPLICATE_OF` claims were never extracted from these comments.

The fix was an "author rule": always extract the document author as a `PERSON` entity. This guarantees at least one entity exists for claim attribution, since `DUPLICATE_OF` requires a person as subject ("who declared the duplicate").

But this introduced a second failure mode. When a comment author describes system behavior in first person and no component is explicitly named, the LLM has only the PERSON entity available and uses it as the subject of `CAUSES` claims. Example from issue #32342:

> "This does not solve my problem as the DZ is only around a trim. When preparing the payload my trim could be elsewhere..."

The LLM produced `JeyPi causes "dead zone is only around trim"` — semantically wrong. JeyPi is *reporting* the behavior, not causing it.

Three complementary fixes resolved this:

1. **Author tagging** (deterministic) — after pass 1, any PERSON matching `doc.author` gets `is_document_author=True`. No LLM call.
2. **Subject-type validation** (deterministic) — per-predicate type guards enforce that `CAUSES`, `AFFECTS`, `REGRESSED_IN` require COMPONENT/HARDWARE/FIRMWARE as subject; `DUPLICATE_OF` requires PERSON. Author-flagged persons are blocked from non-`DUPLICATE_OF` predicates.
3. **Implicit component extraction** (prompt) — the entity prompt now instructs the LLM to infer component names from behavioral descriptions even when the text doesn't name them explicitly. "The servo can not be operated by the RC when in Auto modes" → extract `servo`, `Auto modes`, `RC` as components.

The prompt change did the heavy lifting — re-extraction of the same document produced four inferred components and zero person-as-subject claims. The validation layer is a safety net for cases where the prompt still fails.

This is the kind of problem that only surfaces at scale. The three-layer fix (tag → validate → improve extraction) is deliberate: each layer catches failures the others miss, and the deterministic layers are zero-cost insurance against prompt regressions.
<div style="page-break-after: always;"></div>
### 3b. Structured extraction for timeline events

About 4,000 of the 6,960 documents are timeline events — fully structured GitHub API objects with typed fields like `event_type`, `event_detail.assignee`, `event_detail.label`. There's no free text to analyze. Sending these through the LLM wastes API calls and produces inconsistent results.

A `StructuredExtractor` handles these deterministically:

| Event type | Entity extracted | Entity type |
|---|---|---|
| `assigned` / `unassigned` | `event_detail.assignee` | `PERSON` |
| `labeled` / `unlabeled` | `event_detail.label` (if in known-component set) | `COMPONENT` |
| `milestoned` | `event_detail.milestone` | `FIRMWARE_VERSION` |

Every mention gets `mention_source=STRUCTURED_RULE` (versus `LLM_EXTRACTED` for LLM-produced mentions), enabling downstream filtering and quality analysis by provenance. The structured extractor produces no claims — timeline events lack the textual evidence required by the claim schema, and manufacturing synthetic evidence would undermine the grounding guarantee.

This saves roughly 12,000 LLM calls (4,000 docs × 3 passes) at zero quality loss.

### 3c. Validation, grounding, and quality gates

**Verbatim evidence** is the core grounding mechanism. Every evidence record contains an exact excerpt copied from the source document, plus character offsets and a SHA1 hash of the source text at extraction time.

Character offsets are computed via `body.find(excerpt)` after extraction. When the LLM slightly paraphrases — strips markdown, changes punctuation, drops a trailing word — `find()` returns -1. The design decision here was to accept -1 offsets as an honest signal rather than force the LLM to predict character positions (which LLMs do poorly) or implement fuzzy alignment (which adds complexity for marginal gain at this scale). The claim still exists, the evidence text still exists, the source hash still enables edit detection. You lose precise character citation, which matters for a UI but not for grounding verification.

**Validation pipeline** runs after each extraction pass:
- Entity filtering removes false positives (git commit hashes, C struct references that the LLM misidentifies as entities)
- Excerpt verification confirms evidence text actually appears in the source document
- Claims with confidence < 0.5 are dropped
- Two retries per pass with graceful degradation — if all retries fail, the document is skipped rather than blocking the pipeline

**Quality gates** operate on the extraction output:
- Schema versioning is embedded in extraction metadata. When the schema changes, quality gates flag claims extracted under an older version as stale, enabling targeted re-extraction instead of full re-runs
- Source hash comparison detects when a document has been edited since extraction, flagging the associated evidence as potentially outdated

**Bitemporal fields** on claims distinguish *when something happened* (`event_time` — the firmware version for `FIXED_IN`/`REGRESSED_IN`) from *when it was stated* (`assertion_time` — timestamp of the most recent evidence). A bug fixed in firmware 4.3 may have been reported months later. Collapsing these into a single timestamp produces incorrect temporal ordering.

**Claim provenance** (`ClaimProvenance`) replaces a simple list of contributing mention IDs with full extraction context: document ID, source hash, predicate, schema version, prompt version, extraction timestamp. This enables targeted re-extraction (only re-run documents whose claims used an older prompt version) and full audit trails (trace any canonical claim back to the exact extraction event).

---

## 4. Deduplication & Canonicalization

### 4a. Entity resolution

Entity resolution went through three iterations, each driven by observed failures on real data. The final design replaces all fuzzy string matching with structural rules, corpus evidence, and context-aware LLM decisions.

#### Why fuzzy matching fails for this domain

The initial resolver used `rapidfuzz` `token_set_ratio` with auto-merge at ≥90% similarity and LLM review at 75–89%. This caused systematic false merges in ArduPilot's corpus:

- "SITL" and "SITL Rover" — token_set_ratio ≈100, but SITL is a simulator framework and SITL Rover is a specific vehicle configuration
- "EKF" and "EKF3" — substring containment auto-merged them, but they're distinct components with different behavior
- "GPS" and "GPS Yaw" — token overlap suggests similarity, but GPS is a subsystem and GPS Yaw is a specific feature

ArduPilot component names are compositional (prefix + qualifier). Fuzzy string matching treats "SITL" as a subset of "SITL Rover" when they're actually in a parent-child relationship. The right abstraction is structural analysis, not character-level similarity.

#### Per-type resolver architecture

The monolithic resolver was replaced with a `BaseEntityResolver` ABC and five subclasses, each encoding the domain-appropriate identification rule:

| Resolver | Identification rule | Rationale |
|---|---|---|
| `PersonResolver` | `github_handle.lower()` | Handles are canonical identifiers |
| `ParameterResolver` | `surface_form.upper().replace(" ", "_")` | Parameters are exact technical names |
| `FirmwareVersionResolver` | `(version_string, vehicle_type_or_null)` | Null vehicle_type treated as wildcard |
| `ComponentResolver` | Structural rules + LLM for ambiguous cases | Names are compositional, not fuzzy |
| `HardwarePlatformResolver` | Same framework as ComponentResolver | Same compositional naming patterns |

For Person and Parameter, exact-match keying is sufficient — these identifiers don't have surface-form variation worth worrying about (after normalization). Firmware versions needed one fix: a mention with `vehicle_type=None` and one with `vehicle_type="Copter"` for the same version number were splitting into separate entities. The fix treats null vehicle_type as a wildcard that merges with any match.

#### Structural rules for semantic entities

The `ComponentResolver` and `HardwarePlatformResolver` use a four-step structural decision tree that replaced all fuzzy scoring:

1. **Numeric variant guard** — "EKF" vs "EKF3" → always distinct (trailing integers indicate versioned components)
2. **Exact normalized match** → merge
3. **Structural key match** — strip namespace prefixes (`ap_`, `ardu`) and generic suffixes (`module`, `system`, `handler`), compare stripped forms. "AP_GPS" and "GPS Module" both reduce to `gps` → merge
4. **Co-document Levenshtein-1** — edit distance of 1 AND both mentions in the same document → route to LLM (catches typos like "Compas"/"Compass"). Different documents → keep distinct

The `rapidfuzz` dependency was removed entirely. A simple `_edit_distance()` method handles the short-string comparisons needed for step 4.

#### Corpus-grounded merging via ALIAS_OF

A new `ALIAS_OF` predicate was added to the extraction schema. The LLM extracts alias claims when text contains explicit evidence: parenthetical aliases ("EKF (Extended Kalman Filter)"), slash notation, "also called", "formerly known as".

During resolution, an alias pass runs between entity resolution and claim resolution. For each `ALIAS_OF` claim, the subject and object are resolved to canonical entities. If they're different entities, they're merged — with the alias claim serving as explicit textual evidence for the merge. `ALIAS_OF` claims are then filtered out before claim resolution; they're merge evidence, not queryable facts.

This inverts the usual approach: instead of the resolver guessing whether two strings mean the same thing, the extractor finds explicit statements in the corpus that they do. The merge is grounded in evidence, not heuristics.

#### Context-aware LLM merging

For ambiguous cases that reach the LLM (Levenshtein-1 candidates, shared-prefix pairs), the merge prompt now includes `context_sentence` — the exact sentence where each entity appears. This was added as a field on `EntityMention` during extraction. With context, the LLM can distinguish "RC" meaning "radio control" from "RC" meaning "release candidate" based on surrounding text.

#### Coherence splitting (diagnostic)

A `CoherenceSplitter` runs after all merges to detect possible over-merging. For each entity with ≥5 claims, it embeds the claim descriptions, builds a similarity graph, and finds connected components. An entity whose claims form two disconnected clusters may have been incorrectly merged. Results are written to `data/reports/split_candidates.json` for manual review — the splitter flags but doesn't auto-fix, since splitting an entity has cascading effects on claims and evidence.

#### Results

251 mentions resolved to 127 canonical entities (49.4% deduplication rate): 29 persons, 33 components, 40 parameters, 14 firmware versions, 11 hardware platforms.

### 4b. Claim deduplication

Claim deduplication operates at two levels: structural grouping by predicate-specific keys, and semantic deduplication of freetext within groups.

#### Per-predicate grouping keys

The original claim resolver used a two-way branch: `CAUSES`/`REGRESSED_IN` grouped by `(subject, predicate)`, everything else by `(subject, predicate, object, freetext)`. This caused four distinct bugs:

| Predicate | Bug | Fix |
|---|---|---|
| `CAUSES` | All symptoms for a component collapsed into one claim — distinct failure modes lost | Key on `(subject, predicate, object_freetext)` |
| `REGRESSED_IN` | All regressions for a component (across versions) collapsed | Key on `(subject, predicate, object_canonical_id)` |
| `HAS_PARAMETER` | Same parameter with different values shared IDs due to identical hash input | Key includes `context_hash = sha1(freetext)[:8]` |
| `DUPLICATE_OF` | Two people independently noting the same duplicate produced two claims | Key on `(predicate, object_freetext)` — subject excluded |

`CAUSES` is intentionally kept per-symptom: a component can genuinely cause multiple distinct failure modes simultaneously. A motor stall and a GPS dropout from the same component are not the same bug. Collapsing them loses information valuable for root cause analysis.

For `DUPLICATE_OF`, excluding the subject from the grouping key means multiple people declaring the same duplicate produce a single canonical claim. The canonical subject is chosen as the person with the earliest `assertion_time` — the first person to state the duplicate is most authoritative.

#### Freetext semantic deduplication

After structural grouping, `CAUSES` claims with different freetext can still be near-synonyms ("motor stops responding" vs "motor becomes unresponsive under load"). A `FreetextClaimDeduper` handles this using embedding similarity:

- Score ≥ 0.92 → auto-merge (clear synonym)
- 0.75 ≤ score < 0.92 → LLM review via `ClaimMergeResolver`
- Score < 0.75 → keep distinct

These thresholds are slightly tighter than the entity resolver's (0.90/0.75) because claim freetext is more varied — a 0.88 embedding score on entity names almost always means the same entity, but the same score on symptom descriptions can mean related-but-distinct failure modes.

Pairwise merge decisions are combined via union-find, handling transitive merges correctly. The claim with the most evidence IDs serves as the merge base; evidence, freetext variants, and provenance records are unioned across the group.

### 4c. Temporal reasoning & conflicts

Three temporal mechanisms maintain correctness over time:

**FIXED_IN / valid_until**: When a bug is marked fixed in a version, related `CAUSES` and `AFFECTS` claims for the same component get `valid_until` set to that version and `is_current=False`. The system knows these claims are historical.

**REGRESSED_IN / valid_from**: When a bug regresses, related claims get `valid_from` set to the regression version. Combined with `valid_until`, this creates a temporal window of applicability.

**Superseding for REGRESSED_IN**: When multiple regression claims exist for the same component, a `SupersedingResolver` marks older ones with `is_current=False, superseded_by=<latest_claim_id>`. Only `REGRESSED_IN` is auto-superseded — multiple `CAUSES` claims can genuinely coexist (different bugs), and `FIXED_IN` superseding is already handled by the temporal resolver.

The pipeline order matters: `FreetextClaimDeduper` runs before `TemporalResolver` (so temporal logic operates on clean, deduplicated claims), and `SupersedingResolver` runs after `TemporalResolver` (it needs `assertion_time` to be set and relies on `FIXED_IN` having already marked some claims as historical).

### 4d. Reversibility & audit

Every merge decision is logged with the strategy that triggered it (deterministic, suffix_strip, namespace_prefix, fuzzy_llm, alias_of) and the similarity score where applicable. This enables post-hoc review: if a canonical entity looks wrong, you can trace back to which merge produced it and why.

`ClaimProvenance` records on each canonical claim track the full extraction context — document ID, source hash, schema version, prompt version, extraction timestamp. If you change the extraction prompt, you can identify exactly which claims were produced by the old prompt and target them for re-extraction.

A coherence report runs after all resolution phases, flagging:
- Orphan entities (components with no claims — possible extraction gaps)
- Mixed-predicate components (both current `CAUSES` and `FIXED_IN` — the fix should have marked the cause as historical)
- High alias counts (≥4 aliases — possible over-merging)
- Low-confidence entities (mean confidence < 0.65 — noisy extractions)

The report includes a `grounding_status_distribution` showing how many claims are fully grounded, partially grounded, or unverifiable based on source freshness checks.

---

## 5. Memory Graph

### Why a property graph

The memory needs to answer multi-hop questions: "what components are affected by bugs that regressed in version X?" This requires traversing entity→claim→evidence→document chains efficiently. A property graph with Cypher query support handles this naturally.

The implementation uses Kuzu — an embedded graph database (no server process, single pip install, on-disk persistence). The schema has 8 node tables and 18 relationship tables.

*Note: Kuzu has since been archived by its authors. A production deployment would migrate to a maintained embedded graph (e.g., DuckPG, or a hosted option like Neo4j Aura).*

### Schema: typed tables vs. generic entity table

The schema uses separate node tables per entity type rather than a single `Entity` table with a `type` column:

- **Type-specific properties**: Person has `github_handle`, Parameter has `value`, FirmwareVersion has `version_string` and `vehicle_type`. A generic table would need nullable columns for every variant.
- **Typed relationships**: `Component_HAS_SUBJECT` only connects Component→Claim. The schema enforces this — you can't accidentally create a Person→Claim subject edge with the wrong type.
- **Query clarity**: `MATCH (c:Component)-[:Component_HAS_SUBJECT]->(cl:Claim)` is self-documenting.

The tradeoff is more DDL: 5 entity tables × 4 relationship patterns = 20 relationship tables. But the schema is generated from list comprehensions, so the maintenance cost is a few extra lines of Python, not manual SQL.

### Loading and idempotency

`GraphLoader` reads all resolved JSONL files and loads via `MERGE` statements — idempotent by design. Ten load methods execute in dependency order: documents → evidence → entities → claims → subject rels → object rels → supported_by → sourced_from → mentioned_in → superseded_by.

### Verification

The pipeline verifies the complete evidence chain works end-to-end:

```
DShot --causes--> [ev:ev_0ab03b51_0] --> doc:ardupilot/ardupilot#32351
DDFP --has_parameter--> [ev:ev_0ab03b51_0] --> doc:ardupilot/ardupilot#32351
DShot --regressed_in--> [ev:ev_0ab03b51_1] --> doc:ardupilot/ardupilot#32351
```

Every claim in the graph traces through evidence back to the original GitHub issue. The grounding chain from extraction is preserved and queryable.

### Graph statistics

| | Count |
|---|---|
| Entity nodes | 153 (27 person, 68 component, 38 parameter, 8 firmware, 12 hardware) |
| Claim nodes | 81 |
| Evidence nodes | 151 |
| Document nodes | 6,715 |
| Relationships | 649 |

---

## 6. Retrieval & Grounding

### Entity matching: two-stage approach

A question hits two stages before any graph traversal:

**Stage 1 — LLM intent classification**: A single tool-calling LLM request extracts entity names, selects a query strategy, and optionally applies temporal or confidence filters. This replaced an earlier design with separate LLM calls for entity extraction and strategy selection — one call is faster and produces more coherent results. A 3-second timeout with keyword-based fallback keeps latency bounded.

**Stage 2 — Graph name search**: Each extracted entity name is searched across all 5 entity tables with case-insensitive `CONTAINS` matching. Results are deduplicated by `canonical_id` and ranked by `mention_count` (top 5).

Neither stage alone is sufficient. The LLM handles linguistic variation ("What's wrong with DDFP?" → extracts `DDFP` as a component) but might hallucinate entity names. Graph search validates against real entities and provides the canonical IDs needed for Cypher traversal.

### Strategy-based query routing

Different questions need fundamentally different graph traversals. Rather than generating Cypher dynamically (fragile, hard to test), the retriever selects from seven predefined strategies:

| Strategy | When selected | What it queries |
|---|---|---|
| `component_deep_dive` | "bug"/"problem" + specific component | All claims where component is subject |
| `component_bug_scan` | Bug keywords, no specific component | Current AFFECTS/CAUSES across all components |
| `parameter_inventory` | "parameter"/"setting"/"config" | HAS_PARAMETER claims with Parameter join |
| `person_ranking` | "who"/"contributor" | Person nodes by mention count |
| `evidence_chain` | "cause"/"why"/"root cause" + component | CAUSES claims with full evidence chain |
| `entity_issue_crossref` | "mention"/"reference" | MENTIONED_IN documents + related claims |
| `temporal_query` | "fixed"/"regressed"/"version" | Version-specific or general temporal claims |

Each strategy is a tested, optimized Cypher pattern. Adding a new one is a single method addition with no changes to existing strategies.

### Scoring

Each claim is scored with four weighted signals:

```
score = confidence × 0.4 + is_current × 0.3 + support_level × 0.2 + recency × 0.1
```

**Confidence** (0.4) is the most reliable signal — it comes directly from extraction quality. **Temporal currency** (0.3) ranks active claims above historical ones. **Support level** (0.2) rewards claims backed by multiple evidence records (`overwhelming=1.0`, `strong=0.75`, `single=0.4`, `unsupported=0.0`). **Recency** (0.1) is a tiebreaker — recent assertions preferred, with `None` mapped to 0.5 (neutral).

### Conflict detection and synthesis

The retriever checks for superseded claims via `Claim_SUPERSEDED_BY` traversal. When conflicts exist (e.g., a `CAUSES` claim superseded by a `FIXED_IN`), both the older and newer claims are surfaced.

An LLM synthesis step produces a natural-language answer from the top 5 ranked claims. The system prompt enforces citation with `[claim_id]` references, 3–5 sentences, and no invention beyond what the claims state. Conflict information is fed into the synthesis prompt so the answer naturally acknowledges when something was true previously but has been updated.

### Output: ContextPack

Every retrieval returns a `ContextPack` (Pydantic model):

- Question text and selected strategy
- Matched entities with canonical IDs and mention counts
- Ranked claims with scores, and for each claim: evidence snippets with document context (issue title, number, state, author, maintainer status)
- Synthesized answer with claim citations
- Query timing metadata

Every item in the pack traces to evidence. There are no synthesized facts without provenance.

### Example

**Question**: "What's wrong with DDFP?"

- Strategy: `component_deep_dive`
- Matched: DDFP (23 mentions), DDFP_SPIN_MIN, DDFP_SPIN_MAX
- Top claim: `DDFP --affects--> (none)` — score 0.910, strong support, evidence from 2 contributors across issues #32351 and #32288
- Second claim: `DDFP --causes--> "linearization curve must be adjusted..."` — score 0.888, 2 excerpts from lthall in #32288
- Synthesized answer cites both claims and notes the lack of a specific `FIXED_IN` claim

---

## 7. Visualization

A Streamlit web UI with three tabs covers the main exploration use cases.

### Graph View

Uses `st_link_analysis` (Cytoscape.js wrapper) to render entities and claims as a force-directed graph. Claims are rendered as intermediate nodes between subject and object entities (hypergraph representation) — this makes multi-evidence claims, freetext-only objects, and claim metadata (confidence, predicate) visible on the graph rather than crammed into edge labels.

Sidebar filters control: entity types (multiselect), predicates (multiselect), confidence threshold (slider), and whether to show historical/superseded claims. Clicking a node opens a situation panel with three sections:

- **What is known** — claims grouped by predicate, current claims first, historical collapsed
- **Why we believe it** — evidence grouped by issue, maintainer contributions highlighted
- **Related** — connected entities by type, document list, conflict banner if superseded claims exist

### Retrieval Tab

Wraps the `MemoryRetriever` from phase 5. Users type questions or click example buttons. Results show matched entities, the selected strategy, ranked claims with expandable evidence, and the synthesized answer in a highlighted box above the sources.

### Merge Inspector

Browsable view of all merge decisions from the resolution phase: entity merges (filterable by strategy and similarity score), freetext claim dedup results (with embedding scores), and claim evidence merges. This is the auditability interface — you can trace any canonical entity or claim back to the merge decisions that created it.

### Export

`viz/export.py` produces `data/outputs/graph_export.json` — a self-contained snapshot with entities, claims (evidence embedded inline), and summary statistics. This is the serialized form of the memory graph for submission.

---

## 8. Update Monitoring & Source Freshness

A memory system that can't detect when its sources change will silently serve stale information. The `UpdateMonitor` addresses this by auditing the grounding chain from evidence back to raw documents.

### How it works

For each evidence record, the monitor:

1. Resolves the `doc_id` to the appropriate raw file (issue body, comment, or timeline event)
2. Computes `sha1(current_body)` and compares to the `source_hash` stored at extraction time
3. Classifies the evidence status:
   - Hash matches → `active`
   - Hash differs, excerpt still present in body → `edited_excerpt_intact`
   - Hash differs, excerpt gone → `edited_excerpt_removed`
   - File missing → `deleted`

Evidence statuses are propagated to claim-level `grounding_status` using priority logic:

| Claim grounding status | Condition |
|---|---|
| `fully_grounded` | All evidence active |
| `partially_grounded` | Some evidence active, some not |
| `edited_excerpt_intact` | No active evidence, but excerpts still present in edited docs |
| `unverifiable` | No active or intact evidence, some edited with excerpts removed |
| `ungrounded` | All evidence deleted |

A JSON cache keyed by `doc_id → current_hash` skips unchanged documents on repeated runs, making incremental monitoring cheap.

### Why this matters

This is the "long-term correctness" mechanism. The retrieval layer can filter or downrank claims by `grounding_status`. The coherence report includes a `grounding_status_distribution`, making source freshness part of the system's observability surface. In a production system, `ungrounded` claims would trigger re-extraction or manual review rather than being silently served.

---

## 9. Adapting to Layer10's Environment

The pipeline was built against GitHub issues, but the architecture is designed to generalize. Here's what changes — and what doesn't — for email, Slack, Jira, and docs.

### Ontology changes

The five entity types would expand:

| New type | Source | Identification rule |
|---|---|---|
| `TEAM` | Slack channels, Jira project membership | Channel/project name |
| `PROJECT` | Jira project, doc folder, email thread subject | Project key or normalized name |
| `CUSTOMER` | CRM references in email/Slack | Email domain or CRM ID |
| `DECISION` | Meeting notes, Slack threads with reactions | Content-addressable from decision text |

New predicates: `DECIDED` (person decided X in meeting/thread), `ASSIGNED_TO` (person→ticket), `BLOCKED_BY` (ticket→ticket), `OWNED_BY` (component→team). The existing predicate framework (enum + per-predicate grouping key + subject-type guard) extends without structural changes.

### Structured extractors for structured sources

The `StructuredExtractor` pattern scales directly. Jira webhook payloads, Slack message metadata, and email headers are all structured data that should bypass the LLM:

- Jira status transitions → `FIXED_IN`, `REGRESSED_IN` claims with deterministic extraction
- Slack channel membership changes → `PERSON` mentions with team context
- Email headers → `PERSON` entities with identity linkage (email address → Slack handle → Jira username)
- Calendar events → meeting entities with attendee relationships

LLM extraction is reserved for free-text content: email bodies, Slack messages, Jira descriptions and comments, document prose. The `MentionSource` enum already distinguishes provenance; retrieval can weight structured-rule evidence higher than LLM-extracted evidence for reliability-sensitive queries.

### Cross-system entity resolution

The hardest problem in a multi-source environment is identity resolution across systems. The same person has an email address, a Slack handle, a Jira username, and possibly a GitHub account. The per-type resolver architecture handles this: `PersonResolver` would accept multiple identifier types and merge on any shared identifier, with a priority order (SSO email > Slack handle > display name).

For components and projects, the structural rules (namespace prefix stripping, suffix normalization) generalize well — "PROJ-123" in Jira and "#proj-123" in Slack should resolve to the same entity after normalization.

### Permission-aware retrieval

The current system serves all evidence to all users. In a production environment with email and Slack data, retrieval must respect source access controls. The approach:

- Every evidence record already carries a `doc_id` that identifies the source system and document
- Add an `access_scope` field to evidence (derived from source system ACLs at ingestion time)
- At retrieval time, filter evidence by the querying user's access scopes *before* scoring
- Claims with no accessible evidence are excluded from results
- The `grounding_status` computation runs per-user: a claim might be `fully_grounded` for one user and `ungrounded` for another based on their access

This is a filter on the existing retrieval pipeline, not a redesign. The key constraint is that access scopes must be kept current — a Slack channel made private should invalidate evidence from that channel for non-members.

### Incremental ingestion and cost

The pipeline's JSONL interchange format and idempotent graph loading support incremental operation: new documents are appended, extraction runs on new docs only, resolution re-runs on the changed entity/claim sets, and graph loading merges new data.

For cost at scale, the structured-extractor-first strategy is critical. In a typical organization, the majority of data volume is structured (Jira transitions, Slack metadata, email headers). Reserving LLM calls for free-text content keeps costs proportional to the amount of unstructured text, not total event volume.

The update monitor's caching mechanism extends to incremental operation: only re-scan documents whose raw content has changed since the last run.

### What stays the same

The core pipeline — 3-pass extraction → per-type entity resolution → per-predicate claim grouping → temporal reasoning → graph loading → strategy-based retrieval — doesn't change. The architecture was built around the assumption that sources are heterogeneous (structured and unstructured), entities need domain-specific resolution rules, and claims need temporal validity tracking. These assumptions hold for Layer10's target environment.

---

## Test Coverage

80+ unit tests covering all pipeline phases:

- **Schema** — Pydantic round-trip serialization, ID determinism, enum handling
- **Extraction** — full 3-pass pipeline with mocked LLM, validation logic, author rule, subject-type guards
- **Entity resolution** — 26 tests across all 5 resolver classes, structural rules, numeric variant guards, alias resolution, coherence splitting
- **Claim resolution** — per-predicate grouping, evidence merging, temporal annotations, superseding
- **Graph** — 20 tests for schema validation, node/relationship loading, idempotency
- **Retrieval** — 10 tests for scoring, strategy selection, entity matching, ContextPack validation
- **Update monitor** — 12 tests for status classification, propagation, caching, idempotency
- **Integration** — end-to-end pipeline with real LLM calls against the API gateway

All LLM calls in unit tests are mocked. Integration tests run against the live API with real ArduPilot data.

---

## Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment
export LLM_API_KEY=<your-key>
export LLM_API_BASE=<truefoundry-gateway-url>

# 3. Ingest
python -m ingestion.fetch --repo ArduPilot/ardupilot --count 1000
python -m ingestion.preprocess

# 4. Extract
python -m extraction.pipeline

# 5. Resolve
python -m resolution.pipeline

# 6. Build graph
python -m graph.pipeline

# 7. Run retrieval demo
python -m graph.demo

# 8. Launch visualization
streamlit run viz/app.py

# 9. Export graph
python -m viz.export
```

Each stage reads from and writes to JSONL files in `data/`. Any stage can be re-run independently.