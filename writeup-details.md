# Writeup Details — Grounded Long-Term Memory System

A living document tracking architectural decisions and implementation details across all phases.

---

## Project Overview

**Corpus**: ArduPilot/ardupilot GitHub issues — 1000 issues fetched via the GitHub API, normalized into 6960 documents (issues, comments, timeline events).

**LLM**: Claude Haiku 4.5 via TrueFoundry/litellm gateway. All LLM calls route through `config.LLM_KWARGS` for centralized API key and endpoint management.

**Architecture**: Six-phase pipeline with JSONL files as the interchange format between stages:

1. Ingestion — fetch and normalize raw GitHub data
2. Extraction — LLM-powered entity/claim/evidence extraction
3. Resolution — deduplicate entities, merge claims, temporal reasoning
4. Graph — build in-memory context graph from resolved data
5. Retrieval — embedding search + graph traversal for queries
6. Visualization — web UI for exploring the graph

**Why JSONL between stages**: Each phase reads from and writes to JSONL files, making the pipeline resumable, inspectable, and debuggable. Any stage can be re-run independently without re-running upstream stages.

---

## Phase 1: Ingestion

**Commit**: `fff0953` (Mar 4)
**Output**: 6960 documents in `data/processed/documents.jsonl`

### What was built

- **GitHub API client** (`ingestion/raw_store.py`) — paginated fetching of issues, comments, and timeline events with rate-limit handling and resumable fetches (tracks last-fetched issue number)
- **RawStore** — atomic writes using temp-file-then-rename pattern, per-issue JSON storage, metadata tracking for fetch state
- **Preprocessor** (`ingestion/preprocessor.py`) — normalizes raw API responses into `RawDocument` Pydantic models; filters timeline events to semantic-only types (labeled, assigned, renamed, milestoned, referenced, etc.), discarding noise like subscribed/mentioned events

### Design decisions

| Decision | Rationale |
|---|---|
| **Event filtering** | GitHub timeline events include many non-semantic events (subscribed, mentioned). Filtering to labeled/assigned/renamed/etc. keeps only events that carry meaningful information about the issue. |
| **Atomic writes** | Write to a temp file then rename — prevents corrupted files if the process crashes mid-write. |
| **Resumable fetches** | Store the last-fetched issue number so re-running ingestion picks up where it left off, avoiding redundant API calls. |
| **Flat JSONL output** | One document per line makes streaming, counting, and debugging trivial. Tools like `wc -l`, `head`, `jq` work directly. |
| **Pydantic v2 models** | Type safety and serialization for free. All downstream phases can trust the shape of ingested data. |

---

## Phase 2: Extraction

**Commit**: `de02c98` (Mar 4)
**Output**: 251 entity mentions, 168 evidence records, 99 claim mentions

### Schema design

Defined in `extraction/schema.py`:

- **EntityType** enum: `PERSON`, `COMPONENT`, `PARAMETER`, `FIRMWARE_VERSION`, `HARDWARE_PLATFORM` — covers the core concepts in ArduPilot issue discussions
- **Predicate** enum: `CAUSES`, `AFFECTS`, `FIXED_IN`, `REGRESSED_IN`, `HAS_PARAMETER`, `DUPLICATE_OF` — captures the relationships between entities that matter for a memory system
- **SourceStatus** enum: tracks whether evidence is `ACTIVE`, `EDITED`, or `DELETED`
- **IDGenerator**: deterministic SHA1-based IDs with type prefixes (`m_`, `ev_`, `cm_`) — ensures reproducibility and makes IDs content-addressable

### 3-pass LLM architecture

Each document goes through three sequential LLM passes:

1. **Entity pass** — extract entity mentions with types, surface forms, and attributes
2. **Evidence pass** — extract verbatim excerpts grounding each entity, with character offsets and source hashes
3. **Claim pass** — extract structured claims (subject-predicate-object triples) grounded in the entities and evidence from previous passes

**Why 3 passes instead of 1**: Separation of concerns means each pass has a focused prompt and simpler output schema, improving LLM reliability. Each pass builds on the previous one's output, creating a grounding chain: claims reference evidence, evidence references entities.

### Evidence grounding

- **Verbatim excerpts**: Evidence records contain exact text copied from the source document
- **Character offsets**: Start/end positions for locating excerpts in the original text
- **Source hashes**: SHA1 of the source text at extraction time, enabling detection of edits/deletions downstream

**Why verbatim evidence**: This is the core "grounding" mechanism — every claim in the system can be traced back to exact text in the original documents. No hallucinated or paraphrased evidence.

### Validation pipeline

Implemented in `extraction/validator.py`:

- **Entity filtering**: Removes false-positive entities like git commit hashes and C struct field references that the LLM sometimes extracts as entities
- **Excerpt verification**: Confirms that evidence excerpts actually appear in the source document
- **Confidence thresholds**: Claims with confidence < 0.5 are dropped — removes low-quality extractions
- **Retry logic**: 2 retries per LLM pass with graceful degradation (if all retries fail, skip the document rather than crash)

### Quality gates

Implemented in `extraction/quality.py`:

- **Stale claim detection**: Flags claims extracted with an older schema version (schema version is embedded in extraction metadata)
- **Ungrounded claim detection**: Flags claims whose evidence has been edited or deleted (detected via source hash mismatch)

### Design decisions

| Decision | Rationale |
|---|---|
| **3-pass separation** | Simpler prompts per pass, better LLM accuracy, each pass grounded in previous outputs. Trade-off: 3x LLM calls per document. |
| **Verbatim evidence** | Enables true grounding — claims can always be traced to exact source text. |
| **Soft validation** | Low-confidence claims are dropped but don't block the pipeline. Maximizes throughput while maintaining quality. |
| **Local IDs** | SHA1-based deterministic IDs mean re-running extraction produces the same IDs for the same content. |
| **Schema versioning** | Embedding schema version in extractions lets quality gates detect stale data after schema changes. |

---

## Phase 3: Resolution

**Commits**: `6e28ef9` + `9cbf0a9` (Mar 5)
**Output**: 251 mentions → 127 canonical entities (49.4% deduplication rate)

Breakdown by type:
- Person: 29 canonical entities
- Component: 33 canonical entities
- Parameter: 40 canonical entities
- Firmware version: 14 canonical entities
- Hardware platform: 11 canonical entities

### Entity resolution — 4-stage pipeline

Implemented in `resolution/entity_resolver.py`:

**Stage 1: Deterministic grouping**
- Groups mentions by exact-match rules: GitHub handles (for PERSON), parameter names (for PARAMETER), firmware version strings, and exact normalized surface forms
- Fast, zero-cost, high-precision — handles the easy cases first

**Stage 2: Fuzzy matching**
- Applied to COMPONENT and HARDWARE_PLATFORM types only (the types most prone to surface-form variation)
- **Substring auto-merge**: If one entity name is a substring of another (e.g., "GPS" and "GPS module"), auto-merge
- **90%+ similarity auto-merge**: Normalized string similarity above 90% triggers automatic merge
- **75-89% similarity → LLM decision**: Uses `MergeResolver` to ask the LLM whether two entities refer to the same thing, with the original context provided for grounding

**Stage 3: Canonical entity formation**
- Most common surface form becomes the canonical name
- Attributes merged from all mentions
- Content-addressable IDs: canonical entity ID is derived from the merged content, so identical merges always produce the same ID

**Stage 4: Audit trail**
- Every merge decision is logged with the strategy that triggered it (deterministic, fuzzy-auto, fuzzy-llm)
- Enables post-hoc review of resolution quality

### Claim resolution — 3 stages

**Stage 1: Rewrite** — Replace mention IDs in claims with canonical entity IDs from entity resolution

**Stage 2: Group** — Group claims by predicate-specific keys (e.g., CAUSES claims grouped by subject+object, HAS_PARAMETER claims grouped by subject+parameter)

**Stage 3: Merge** — Within each group: union all evidence sets, take max confidence, collect freetext description variants

### Temporal reasoning

- **FIXED_IN** claims: When a bug is marked as fixed in a version, related claims (CAUSES, AFFECTS) are annotated as historical with `valid_until` set
- **REGRESSED_IN** claims: When a bug regresses, related claims get `valid_from` set to the regression version

**Why temporal annotations**: In a long-term memory system, knowing that "X causes Y" was true in firmware 4.2 but fixed in 4.3 is critical for answering queries accurately. Without temporal reasoning, the system would return stale information.

### Design decisions

| Decision | Rationale |
|---|---|
| **Content-addressable IDs** | Canonical entity IDs derived from content ensure idempotent resolution — re-running produces the same output. |
| **Multi-stage pipeline** | Deterministic rules first (fast, high-precision), then fuzzy matching (slower, needs more care), then LLM for ambiguous cases. Minimizes LLM calls while maximizing recall. |
| **Conservative LLM merging** | Only use LLM for the 75-89% similarity band. Below 75% is too risky; above 90% doesn't need LLM confirmation. |
| **Union-Find for transitive merges** | If A merges with B and B merges with C, then A-B-C all become one entity. Union-Find handles this efficiently without explicit transitive closure computation. |
| **Temporal validity annotations** | Annotating claims with valid_from/valid_until enables time-aware retrieval later. Trade-off: adds complexity to resolution, but essential for a long-term memory system. |
| **Audit trails** | Logging merge decisions with strategy attribution enables debugging resolution quality and tuning thresholds. |

---

## Testing

**Commits**: `d6ff9dc`, `e05ba23`, `beb15a8` (Mar 5)

### Unit tests

- **Schema tests** (`tests/test_schema.py`): Pydantic model round-trip serialization, ID generation determinism, enum serialization
- **Extraction tests** (`tests/test_extraction.py`): Full extraction pipeline with mocked LLM responses, validation logic, error handling
- **Entity resolution tests** (`tests/test_entity_resolver.py`): Deterministic grouping, fuzzy matching thresholds, transitive merge behavior
- **Claim resolution tests** (`tests/test_claim_resolver.py`): ID rewriting, grouping logic, evidence merging, temporal annotations
- **Quality tests** (`tests/test_quality.py`): Stale claim detection, ungrounded claim detection

### Integration tests

- **End-to-end pipeline** (`tests/integration/`): Runs extraction → resolution with real LLM calls against TrueFoundry gateway
- Validates that the full pipeline produces well-formed output on real ArduPilot issue data

### Test isolation

- `tests/conftest.py` sets dummy environment variables (`LLM_API_KEY`, `LLM_API_BASE`) so unit tests never hit the real API
- All LLM calls in unit tests are mocked via `unittest.mock.patch`

---

## Prompt Fix: DUPLICATE_OF Claim Extraction

**Commit**: (current)

### What was changed

Two targeted prompt edits in `extraction/prompts.py`:

1. **Evidence prompt** (`EVIDENCE_SYSTEM_PROMPT`) — added a new bullet to WHAT TO EXTRACT for issue relationship statements (duplicate declarations, fix confirmations, regression confirmations). Also added a DO NOT EXTRACT exclusion for speculative duplicate suggestions ("might be related to").

2. **Claim prompt** (`CLAIM_SYSTEM_PROMPT`) — replaced the vague DUPLICATE_OF definition with an expanded version that specifies: use a PERSON entity as subject (the person who stated the duplicate), set `object_freetext` to the plain issue number without "#", confidence thresholds (0.95 explicit, 0.75 hedged, never below 0.7), and positive/negative examples.

### Why this was needed

DUPLICATE_OF claims were not being extracted despite duplicate statements existing in the corpus. Root cause analysis revealed a two-part failure:

- **Pass 2 (evidence)** only looked for technical facts (bug symptoms, parameters, versions) and didn't recognize duplicate statements as extractable evidence
- **Pass 3 (claims)** had a vague DUPLICATE_OF definition that didn't tell the LLM to use a PERSON entity as subject or how to format the issue number, so even if evidence existed, the claim wouldn't be well-formed

### Author rule — always extract document author as PERSON

Short comments like "I believe this is a duplicate of #17383." contain no technical entities (no component, parameter, or version). Without explicit instruction, the LLM sometimes returns an empty entity array for these docs, which triggers an early-exit gate in the extractor (`if not entity_mentions: return`) that skips passes 2 and 3 entirely.

**Fix**: Added an AUTHOR RULE to `ENTITY_SYSTEM_PROMPT` instructing the LLM to always extract the document Author as a PERSON entity. This guarantees at least one entity exists for claim attribution (DUPLICATE_OF requires a PERSON subject).

**Trade-off**: Every document now produces at least one PERSON entity, and passes 2/3 always run (3x LLM calls even for trivial docs). The extra entity cost is minimal — entity resolution deduplicates authors across documents anyway. The extra LLM calls are acceptable given the corpus size.

### Evolution — Author rule causes Person-as-subject for CAUSES claims

The author rule solved the empty-entity problem but introduced a new failure mode: when a comment author describes system behavior in first person and no component entity is named, the LLM has only the PERSON entity available and uses it as the subject of `causes` claims.

**Observed example** — issue #32342, comment by `JeyPi`:
> "This does not solve my problem as the DZ is only around a trim. When preparing the payload my trim could be elswhere I would rather see it done in a way that the servo can not be operated by the RC when in Auto modes. The servo would only move when given input from the mission and reverts to RC when in manual."

The LLM extracted three `causes` claims with `JeyPi` (PERSON) as subject:
- `JeyPi causes "servo cannot be operated by RC when in Auto modes"`
- `JeyPi causes "servo only moves when given input from mission and reverts to RC in manual mode"`
- `JeyPi causes "dead zone is only around trim"`

All three are semantically wrong. JeyPi is *reporting* the behavior, not *causing* it. The actual subject is an unnamed ArduPilot subsystem (servo control / Auto mode logic). Because no component entity was extracted from this comment, the PERSON was the only candidate.

**Root cause chain**:
1. Author rule guarantees a PERSON entity is always extracted — good for DUPLICATE_OF
2. Comment names no component explicitly — so no COMPONENT entity is extracted
3. Claim extraction pass has no rule preventing PERSON as subject of causal predicates
4. LLM picks the only available entity (PERSON) as subject — producing a semantically invalid claim

**What should happen instead**: `causes`, `has_bug`, `fixed_in`, and `regressed_in` predicates imply a technical component or system as their subject. If no component can be identified, the claim should either be skipped or attributed to a synthetic "Unknown Component" entity rather than the comment author.

### Fix — three layered defences

Three complementary changes were implemented together (`extraction/extractor.py` + `extraction/prompts.py`):

**Fix 1 — Mark document author in Pass 1 (`_mark_document_author`)**

After `_parse_entity_blocks` returns, a new post-processing step walks the entity list and finds any PERSON whose `github_handle` or `surface_form` matches `doc.author` (case-insensitive). That entity gets `attributes["is_document_author"] = True`. No LLM call — pure deterministic tagging.

**Fix 2 — Subject-type validation in Pass 3 (`_run_pass3_claims`)**

After the predicate is parsed for each raw claim, two checks run before the claim is created:

- *Check 1 (author guard)*: if subject is a PERSON with `is_document_author=True` and predicate is not `DUPLICATE_OF` → drop, log warning. This is the narrowest possible gate — it only blocks the exact failure mode (author as causal subject) and leaves non-author PERSONs to be handled by check 2.
- *Check 2 (per-predicate type guard)*: enforces allowed subject types per predicate via a module-level lookup table (`_PREDICATE_SUBJECT_TYPES`):
  - `CAUSES`, `AFFECTS`, `REGRESSED_IN` → subject must be `COMPONENT`, `HARDWARE_PLATFORM`, or `FIRMWARE_VERSION`
  - `FIXED_IN`, `HAS_PARAMETER` → subject must be `COMPONENT` or `HARDWARE_PLATFORM`
  - `DUPLICATE_OF` → subject must be `PERSON`

**Fix 3 — Prompt improvements**

- `ENTITY_SYSTEM_PROMPT` gained an **IMPLICIT COMPONENT EXTRACTION** section: if the text describes subsystem behavior without naming the component explicitly, extract the most specific inferable name. Explicitly forbids vague placeholders like "the system" or "the firmware".
- `CLAIM_SYSTEM_PROMPT` gained a **SUBJECT ENTITY RULES** section: states the author PERSON may only be subject of `DUPLICATE_OF`; lists the required subject types per predicate; instructs the LLM to infer a component from context before omitting the claim.

**Result** — same document re-extracted:

| | Before | After |
|---|---|---|
| Entities | `JeyPi` (PERSON) | `JeyPi` (PERSON, `is_document_author=True`), `servo` (COMPONENT), `Auto modes` (COMPONENT), `RC` (COMPONENT), `mission` (COMPONENT) |
| Claims | 3× `JeyPi causes "..."` | `servo affects`, `Auto modes causes "servo cannot be operated by RC input"`, `mission causes "servo moves only when given mission input..."` |
| Bad PERSON-as-subject | 3 | 0 |

The prompt change (implicit component extraction) did the heavy lifting: the LLM inferred four components that the comment never names explicitly. The validation layer is a safety net — it would have dropped any remaining PERSON-as-subject claims even if the prompt failed to extract components.

### Trade-offs

- Slightly longer prompts (+12 lines in evidence prompt, +14 lines in claim prompt), but both changes are targeted and don't affect other predicate types
- The PERSON-as-subject convention for DUPLICATE_OF is a design choice — alternatively the component could be the subject, but PERSON better captures "who said it" which is important for authority/trust in a memory system
- The author rule and the subject-type constraint pull in opposite directions: one guarantees a PERSON entity exists, the other forbids PERSON as subject of causal claims. Both are correct — they apply to different predicates. The fix is predicate-scoped validation, not removing the author rule.
- The implicit component extraction prompt may produce more speculative component entities (e.g. "mission", "RC") that are harder to resolve across documents. This is a worthwhile trade: a slightly noisier entity list is better than semantically invalid claims.

### AFFECTS as a low-precision fallback

The `AFFECTS` predicate serves an important architectural role: **low-precision fallback when component involvement is clear but no specific causal chain can be identified**.

**Observed pattern** — Issue #32288 (#32288/c/3961296798, #32288/c/3963785027):
- Comments mention "AP_MotorsHeli_Single" and "heli DDFP" in the context of voltage scaling issues
- The LLM correctly identifies the component as involved but cannot pin down the specific failure mechanism (voltage scaling before/after mixing, actuator limits, yaw compensation, etc.)
- Extraction produces: `AFFECTS` claims with no object

**Why this is correct behavior**:
- `AFFECTS` is intentionally object-agnostic (grouping key: `(subject, predicate, None)`)
- It encodes "this component is involved in the system behavior" without overcommitting to a specific causal chain
- It is better to have an underspecified claim than a false positive (e.g., claiming DDFP **causes** "voltage scaling" when the actual bug is that the scaling step is **missing**)

**Trade-off for a production system**:
A production system would add a post-extraction quality gate:
1. Flag singleton `AFFECTS` claims (no other claims on that component) with no corresponding `CAUSES` or `FIXED_IN` claims
2. Route these to a higher-capacity model (Claude Sonnet/Opus) with a more specific prompt targeting the failure mechanism
3. Either upgrade to a `CAUSES`/`FIXED_IN` claim, or annotate the `AFFECTS` claim with low confidence

This turns a limitation into a **demonstrated understanding of extraction failure modes** — exactly what a memory system evaluation looks for. The baseline system correctly identifies component involvement; the next iteration would specialize the extraction per component and predicate type.

---

## Phase 2 — Extraction: StructuredExtractor for timeline_event docs

### What was added

A `StructuredExtractor` class (`extraction/structured_extractor.py`) that derives entity mentions deterministically from the structured fields of `timeline_event` documents, bypassing the LLM entirely. The extraction pipeline (`extraction/pipeline.py`) now routes `doc_type == "timeline_event"` docs to `StructuredExtractor` and all other doc types to the existing `Extractor` (LLM).

### Why this approach

Timeline events are fully structured GitHub API objects — they carry typed fields like `event_type`, `event_detail.assignee`, `event_detail.label`, `event_detail.milestone`. There is no free text to analyze. Sending them through the LLM wasted API calls and produced inconsistent results.

Rule-based extraction is both faster and more reliable for structured data: an `assigned` event always yields a PERSON mention, a `labeled` event with a known component name always yields a COMPONENT mention, a `milestoned` event always yields a FIRMWARE_VERSION mention.

### What it extracts

| event_type | Entity extracted | Type |
|---|---|---|
| `assigned` / `unassigned` | `event_detail.assignee` | `PERSON` |
| `labeled` / `unlabeled` | `event_detail.label` (if in known-component set) | `COMPONENT` |
| `milestoned` | `event_detail.milestone` | `FIRMWARE_VERSION` |
| all others (`locked`, `closed`, etc.) | — | (empty result) |

### Trade-offs

- **Coverage vs. noise**: Labels not in `KNOWN_COMPONENT_LABELS` (e.g., `"NeedAnswer"`, `"Priority"`) produce empty results rather than noisy entities. The component set can be extended as needed.
- **No claim extraction**: Timeline events lack the evidence (quotes, excerpts) required by `ClaimMention.evidence_ids`. The structured extractor deliberately returns no claims rather than manufacturing synthetic evidence.
- **Zero API cost**: Every `timeline_event` doc (roughly 4000 of the 6960 total) is now processed in microseconds at zero cost.

---

## Phase 3 — Resolution: Entity Resolver Design Analysis

### The core design problem

The current `EntityResolver` mixes two different questions: "are these two strings similar?" (a generic text comparison) and "by what rule is this entity type identified?" (a domain rule). The right question is the second one.

On inspection, the resolver already uses domain rules for three of the five entity types:

| Type | Current keying strategy | Correct? |
|---|---|---|
| `PERSON` | `github_handle.lower()` | ✓ — handle is the canonical identifier |
| `PARAMETER` | `surface_form.upper().strip()` | ✓ — parameter names are precise technical identifiers |
| `FIRMWARE_VERSION` | `(stripped_version_string, vehicle_type)` | mostly ✓ — one gap (see below) |
| `COMPONENT` | normalized form → fuzzy string similarity | ✗ — wrong abstraction |
| `HARDWARE_PLATFORM` | normalized form → fuzzy string similarity | ✗ — wrong abstraction |

The real problem is narrower: **COMPONENT and HARDWARE_PLATFORM are treated as natural language strings when they are technical identifiers.** "EKF3" is not a fuzzy string — it is a precise name. The substring and token-set-ratio logic is appropriate for paraphraseable text; it is the wrong tool for ArduPilot component names, which are enumerable and exact.

### Known failure modes

**1. COMPONENT substring matching causes false merges (Stage 2)**

```python
if nf_i in nf_j or nf_j in nf_i:
    union_track(mi.mention_id, mj.mention_id, True)  # "substring_match"
```

`"ekf" in "ekf3"` → `True` → EKF and EKF3 are auto-merged. Same problem hits any numbered variant: EKF/EKF2/EKF3, GPS/GPS1/GPS2, Compass/Compass2. These are distinct components with different behaviour — this is a systematic false positive.

**2. FIRMWARE_VERSION splits on `vehicle_type` mismatch (Stage 1)**

The key is `(stripped_version_string, vehicle_type)`. A mention with `vehicle_type="Copter"` and one with `vehicle_type=None` for the same version number become separate canonical entities. No Stage 2 fuzzy merging applies to `FIRMWARE_VERSION`, so the split is permanent.

**3. PARAMETER has no fuzzy merge (Stage 1 only)**

Exact match on `surface_form.upper()`. If the LLM produces `"AHRS_EKF_TYPE"` in one doc and `"AHRS EKF TYPE"` (spaces instead of underscores) in another, they remain as two canonical PARAMETER entities. Not a structured-extractor issue (it produces no PARAMETERs), but a resolver gap.

### Proposed solution: type-specific resolver routing

Rather than one resolver with branching heuristics, decompose into per-type resolvers, each implementing the identification rule appropriate for that type, and route mentions by entity type before resolution. This makes the identification logic explicit, testable in isolation, and easy to extend or replace per type.

---

## Phase 3 — Resolution: Entity Resolver Refactor (per-type resolver classes)

**Commit**: `0a3b454`

### What was built

The monolithic `EntityResolver` was replaced with a `BaseEntityResolver` ABC and five per-type subclasses, each implementing a `make_key` method with the domain-appropriate identification rule:

| Class | Entity type | Identification rule |
|---|---|---|
| `PersonResolver` | `PERSON` | `github_handle.lower()` |
| `ParameterResolver` | `PARAMETER` | `surface_form.upper().replace(" ", "_")` |
| `FirmwareVersionResolver` | `FIRMWARE_VERSION` | `(stripped_version, vehicle_type_or_null)` |
| `ComponentResolver` | `COMPONENT` | normalized form + fuzzy scoring (no substring containment) |
| `HardwarePlatformResolver` | `HARDWARE_PLATFORM` | normalized form + fuzzy scoring |

`SemanticEntityResolver` is a shared base for `ComponentResolver` and `HardwarePlatformResolver` that provides fuzzy scoring logic. `EntityResolver` becomes a thin orchestrator that dispatches mentions by type to the appropriate subclass.

### Three bugs fixed

**1. False merge: numeric variants** — the old substring check (`"ekf" in "ekf3"`) auto-merged EKF and EKF3. Fixed by a `_is_numeric_variant` check in `SemanticEntityResolver`: if two names differ only by a trailing integer, they skip auto-merge and go through fuzzy scoring or LLM decision instead.

**2. False split: firmware version with null vehicle_type** — a mention with `vehicle_type=None` and one with `vehicle_type="Copter"` for the same version never merged. Fixed in `FirmwareVersionResolver`: null vehicle_type is treated as a wildcard and merged with any matching version via `firmware_null_vehicle_merge` strategy.

**3. False split: PARAMETER space/underscore variants** — `AHRS_EKF_TYPE` and `AHRS EKF TYPE` now share a key because `ParameterResolver.make_key` normalizes spaces to underscores before hashing.

### Test coverage

Tests reorganized by resolver class: `TestPersonResolver` (3), `TestParameterResolver` (4), `TestFirmwareVersionResolver` (6), `TestComponentResolver` (8), `TestHardwarePlatformResolver` (2), `TestEntityResolverOrchestrator` (3). All 26 new tests + 54 existing tests pass (80 total).

### Trade-offs

- More files of code but each class is independently testable and readable in isolation
- The `ResolutionDecision` dataclass was introduced to avoid naming collision with `MergeDecision` from claim resolution
- The orchestrator pattern means adding a new entity type is a one-class change with no modification to existing resolvers

---

## Phase 2 — Extraction: Schema additions — MentionSource, ClaimProvenance, bitemporal fields

**Commit**: `89b4af9`

### What was added

Three categories of schema additions to `extraction/schema.py` and `resolution/merge_resolver.py`:

**1. `MentionSource` enum** (`llm_extracted` | `structured_rule`) — added as a field on `EntityMention`. Every mention now records how it was produced. `StructuredExtractor` sets `mention_source=STRUCTURED_RULE` on all mentions it creates; `Extractor` defaults to `LLM_EXTRACTED`.

**Why**: Downstream quality analysis and debugging need to distinguish LLM-generated mentions (which may have noise) from rule-derived mentions (which are deterministic). This also enables filtering: if retrieval wants only high-confidence structured mentions, it can filter on `mention_source`.

**2. `ClaimProvenance`** — replaces the old `supporting_claim_mention_ids` field on `CanonicalClaim`. Each `ClaimProvenance` record captures: `doc_id`, `source_hash`, `predicate`, `schema_version`, `prompt_version`, `extracted_at`.

**Why**: `supporting_claim_mention_ids` only tracked which mention IDs contributed to a merged claim. `ClaimProvenance` tracks the full extraction context — which document, which version of the schema/prompt, and when. This enables:
- Detecting stale claims when schema or prompts change (compare `schema_version`)
- Re-extraction targeting: only re-run documents whose claims are stale
- Audit trail: trace any canonical claim back to the exact extraction event that produced it

**3. Bitemporal fields** on `CanonicalClaim`:
- `assertion_time` — timestamp of the most recent evidence (computed in `ClaimResolver._merge()` as `max(ev.timestamp for ev in evidence_set)`)
- `event_time` — version string (set by `TemporalResolver` for `FIXED_IN`/`REGRESSED_IN` claims from the version field)

**Why**: A grounded memory system needs two time axes: *when something happened* (event_time) and *when it was asserted* (assertion_time). A bug fixed in firmware 4.3 (event_time) may have been reported months later (assertion_time). Using only one axis produces incorrect temporal ordering for queries.

### Trade-offs

- `ClaimProvenance` adds per-claim storage overhead (one record per contributing claim mention). On the current corpus (99 claim mentions → ~40 canonical claims) this is negligible.
- `assertion_time` is only as accurate as the evidence timestamps, which come from GitHub API `created_at` fields — these are reliable for this corpus.
- `event_time` is only set for version-keyed claims (`FIXED_IN`, `REGRESSED_IN`). `CAUSES` and `AFFECTS` claims have `event_time=None` — they have no inherent version anchor.

---

## Phase 2 — Extraction: Excerpt Offset Extraction Design

**Current approach**: Pass 2 (evidence extraction) asks the LLM to copy verbatim text from the document body. The extractor then uses `body.find(excerpt)` to compute character offsets (start, end).

### The Problem

When the LLM slightly paraphrases — strips markdown, changes punctuation, drops a word — `body.find()` returns -1 because the copied text doesn't exactly match what's in the body. Offsets become -1, -1.

Example:
```
Original text:   "The TKOFF_RPM_MIN parameter controls..."
LLM extracts:    "The TKOFF_RPM_MIN parameter controls"  (trailing ellipsis removed)
body.find() result: -1 (exact match not found)
```

### The Alternative: Span-First Approach

Instead of asking the model to copy text and searching for it, ask the model to return character positions directly: `(start: 847, end: 893)`. Then extract the excerpt via `doc.body[start:end]`. The excerpt is always verbatim by construction because Python does the slicing, not the model.

### Why This Is Not Implemented Now

LLMs are poor at counting characters precisely. "The word TKOFF_RPM_MIN starts at character 847" is the kind of task they get wrong consistently. You would need:
- Validation logic to check if offsets are within document bounds
- Fallback strategies when offsets are invalid
- Multiple retries for failed offset predictions
- This could degrade a pipeline that is currently working well

### Design Decision: Accept -1 Offsets as Honest Signal

**Tradeoff accepted**:
- The current -1 fallback is honest — it signals "excerpt not precisely located" without breaking the claim
- The claim still exists, the evidence still exists, the text is still there
- You just lose the exact character position for citation purposes
- At the current scale (99 claims, 168 evidence records), this is an acceptable tradeoff

**When this would need to change**:
- In production with a proper tokenizer-based span extraction approach (using a library like `spaCy` or `transformers.pipelines` for token alignment)
- If downstream retrieval or UI features require precise offsets (currently they don't)

The current approach maximizes pipeline reliability and throughput. Migrating to span-first would be a different implementation, requiring validation and error-recovery logic that isn't justified yet.

---

## Phase 3 — Resolution: Freetext Claim Deduplication + Grouping Fixes

### What was changed

Four grouping bugs in `ClaimResolver._make_group_key` were fixed, a new freetext semantic deduplication stage was added, superseding logic for multi-version REGRESSED_IN claims was added, and a coherence report was introduced.

**Fix 1 — `_make_group_key` (was: two-way branch in `_group`)**

The old `_group` method used a two-way branch: CAUSES/REGRESSED_IN got `(subject, predicate)` and everything else got `(subject, predicate, object, freetext)`. This caused four distinct bugs:

| Predicate | Old key | Bug | New key |
|---|---|---|---|
| `CAUSES` | `(subject, predicate)` | All symptoms for a component collapsed into one claim — distinct failure modes lost | `(subject, predicate, object_freetext)` |
| `REGRESSED_IN` | `(subject, predicate)` | All regression claims for a component (across different versions) collapsed together | `(subject, predicate, object_canonical_id)` |
| `HAS_PARAMETER` | `(subject, predicate, object, freetext)` | Same parameter with different values (e.g. `value=100` vs `value=200`) shared the same `canonical_claim_id` due to identical hash input | `(subject, predicate, object, context_hash)` where `context_hash = sha1(freetext)[:8]` |
| `DUPLICATE_OF` | `(subject, predicate, object)` | Two people independently noting the same duplicate produced two separate canonical claims | `(predicate, object_freetext)` — subject excluded |

All grouping logic now lives in `_make_group_key` — no branching elsewhere.

**Fix 2 — `DUPLICATE_OF` subject selection**

With subject excluded from the key, the merged claim needs a canonical subject. `_select_duplicate_of_subject` picks the subject from the claim with the earliest `assertion_time` (earliest mention is most authoritative for duplicate tracking).

**Fix 3 — `superseded_by` on `CanonicalClaim`**

Added `superseded_by: str | None = None` field. Set by `SupersedingResolver` for older REGRESSED_IN claims.

### Why this approach

The old two-way branch conflated two different intentions: "collect freetext variants" (for CAUSES) and "group by version" (for REGRESSED_IN). Fixing these required making each predicate's grouping rule explicit. A `_make_group_key` method makes each rule a single line — readable, testable in isolation, and easy to extend.

CAUSES is intentionally kept per-symptom: a component can genuinely cause multiple distinct failure modes simultaneously (motor stall and GPS dropout are not the same bug). Collapsing them loses information that is valuable for graph traversal and root cause analysis.

### `resolution/freetext_deduper.py` — semantic deduplication

Runs after `ClaimResolver` and before `TemporalResolver`. With the CAUSES fix, each distinct symptom is now a separate `CanonicalClaim` — but two claims can still be near-synonyms in wording ("motor stops responding" vs "motor stops responding under load"). The deduper handles this.

**Embedder**: Calls `litellm.embedding(model=config.EMBEDDING_MODEL)` first; falls back to a local `SentenceTransformer(config.EMBEDDING_LOCAL_MODEL)` on failure. Embeddings are unit-normalized so cosine similarity = dot product.

**Two-threshold design**:
- `score >= 0.92` → auto-merge (clear synonym, no LLM needed)
- `0.75 <= score < 0.92` → call `ClaimMergeResolver` (LLM tool-use) — borderline cases that require semantic judgment
- `score < 0.75` → keep distinct (no LLM needed)

The 0.92/0.75 thresholds mirror the entity resolver thresholds (0.90/0.75) but are slightly tighter because claim freetext is more varied than entity names — a 0.88 score on entity names almost always means the same entity, but the same score on claim freetext can mean related-but-distinct failure modes.

**Union-find merging**: Pairwise decisions are combined via union-find. This handles transitive merges (if A~B and B~C then A+B+C merge) without inconsistency.

**Merge result**: The claim with the most evidence_ids is used as the base. Evidence, freetext variants, and provenance are unioned. Longest freetext is kept as canonical (more descriptive).

**Trade-off**: Embedding every CAUSES/REGRESSED_IN group adds latency. On the full 6960-document corpus this is ~few seconds for local embeddings. The LLM borderline band calls add more time but are rare (only pairs in 0.75–0.92 band). The payoff is a cleaner graph — the coherence report's `mixed_predicate_component` flag would produce false positives without this step (two slightly different wordings of the same fixed bug would both appear as current CAUSES claims).

### `SupersedingResolver` — REGRESSED_IN superseding

Runs after `TemporalResolver` (needs `assertion_time` to be set). For each subject, REGRESSED_IN claims are sorted by `assertion_time` descending. The most recent is `is_current=True`; older ones get `is_current=False, superseded_by=<latest_id>`.

**Why only REGRESSED_IN**: Multiple bugs genuinely coexist (CAUSES claims), so CAUSES is not auto-superseded. FIXED_IN superseding already happens in `TemporalResolver._apply_fixed_in`. REGRESSED_IN is the only predicate where "the latest known regression version" clearly supersedes older ones for query purposes.

### `monitoring/quality.py` — coherence report

Generated after all resolution phases. Flags:
- `singleton_entity` (info): only one mention — may be noise or a genuinely rare entity
- `orphan_entity` (warning): COMPONENT with no claims — likely an extraction gap or a merge that orphaned the entity
- `high_alias_count` (info): ≥4 aliases — may indicate over-merging in entity resolution
- `mixed_predicate_component` (warning): COMPONENT with both current CAUSES and FIXED_IN claims — the FIXED_IN should have marked the CAUSES as historical; signals a temporal resolution miss
- `low_confidence_claims` (info): mean confidence < 0.65 — entity may have noisy extractions

### Pipeline order rationale

FreetextClaimDeduper runs **before** TemporalResolver so temporal logic operates on clean, deduplicated data. SupersedingResolver runs **after** TemporalResolver because it needs `assertion_time` (set by ClaimResolver) and relies on FIXED_IN already having marked some CAUSES as historical.

---

## Phase 4: Graph — Kuzu Property Graph Loader

**Output**: Kuzu database at `data/graph/memory.kuzu` — 153 entity nodes, 81 claim nodes, 151 evidence nodes, 6715 document nodes, 649 relationships

### What was built

Three files in `graph/`:

- **`graph/schema.py`** — DDL constants for 8 node tables and 18 relationship tables. Entity types (Person, Component, Parameter, FirmwareVersion, HardwarePlatform) each get a typed node table with attributes relevant to that entity type. Claim, Evidence, and Document are the other three node tables. Relationship tables encode the semantic edges: subject/object connections between entities and claims, evidence support chains, document sourcing, entity-document mentions, and claim supersession.

- **`graph/loader.py`** — `GraphLoader` class that reads all resolved pipeline data and loads it into Kuzu via MERGE statements (idempotent — safe to re-run). 10 load methods execute in dependency order: documents → evidence → entities → claims → subject rels → object rels → supported_by → sourced_from → mentioned_in → superseded_by.

- **`graph/pipeline.py`** — CLI entry point (`python -m graph.pipeline`) that creates the database, loads all data, and prints summary counts and verification queries.

### Why Kuzu

| Consideration | Kuzu | NetworkX (alternative) |
|---|---|---|
| **Schema enforcement** | Typed node/rel tables with primary keys — invalid data rejected at insert time | Arbitrary dicts on nodes/edges — schema errors surface at query time |
| **Query language** | Cypher — pattern matching, multi-hop traversal, aggregation built-in | Python iteration — complex traversals require manual BFS/DFS |
| **Persistence** | On-disk database, survives process restarts | In-memory only, must serialize/deserialize |
| **Scale** | Handles millions of nodes efficiently | Degrades above ~100K nodes |
| **Dependency** | Single pip package, embedded (no server process) | Already in requirements.txt |

For a memory system that needs to answer multi-hop queries ("what components are affected by bugs that regressed in version X?"), Cypher pattern matching is significantly more expressive than manual graph traversal. The embedded architecture (no server) keeps deployment simple.

Kuzu was chosen for its embedded graph semantics and zero-infrastructure setup. The project has since been archived by its authors, and a production deployment would migrate to a maintained alternative.

### Schema design: 5 entity tables vs 1 generic entity table

The schema uses **separate node tables per entity type** rather than a single `Entity` table with a `type` column. This was a deliberate choice:

- **Type-specific properties**: Person has `github_handle`, Parameter has `value`, FirmwareVersion has `version_string` and `vehicle_type`. A single table would need nullable columns for all variants.
- **Typed relationships**: `Component_HAS_SUBJECT` only connects Component→Claim. Kuzu enforces this at the schema level — you cannot accidentally create a `Person_HAS_SUBJECT` edge with the wrong source type. A generic `HAS_SUBJECT` from a single Entity table would lose this constraint.
- **Query clarity**: `MATCH (c:Component)-[:Component_HAS_SUBJECT]->(cl:Claim)` is self-documenting. The type is in the query, not a WHERE filter.

**Trade-off**: 5 entity tables × 4 relationship patterns = 20 relationship tables (vs. 4 with a generic Entity table). This is more DDL but the schema is generated from a list comprehension, so maintenance cost is minimal.

### Graph statistics

| Node table | Count |
|---|---|
| Person | 27 |
| Component | 68 |
| Parameter | 38 |
| FirmwareVersion | 8 |
| HardwarePlatform | 12 |
| Claim | 81 |
| Evidence | 151 |
| Document | 6715 |

| Relationship | Count |
|---|---|
| Component_HAS_SUBJECT | 76 |
| HardwarePlatform_HAS_SUBJECT | 5 |
| Claim_HAS_OBJECT_Parameter | 28 |
| Claim_HAS_OBJECT_Component | 2 |
| Claim_HAS_OBJECT_HardwarePlatform | 1 |
| SUPPORTED_BY | 99 |
| SOURCED_FROM | 151 |
| Person_MENTIONED_IN | 100 |
| Component_MENTIONED_IN | 115 |
| Parameter_MENTIONED_IN | 42 |
| FirmwareVersion_MENTIONED_IN | 14 |
| HardwarePlatform_MENTIONED_IN | 16 |

### Verification: full path traversal

The pipeline verifies the complete evidence chain works: entity → claim → evidence → document.

```
DShot --causes--> [ev:ev_0ab03b51_0] --> doc:ardupilot/ardupilot#32351
DDFP --has_parameter--> [ev:ev_0ab03b51_0] --> doc:ardupilot/ardupilot#32351
DShot --regressed_in--> [ev:ev_0ab03b51_1] --> doc:ardupilot/ardupilot#32351
```

Every claim in the graph can be traced through evidence back to the original GitHub issue document — the grounding chain from Phase 2 is preserved and queryable.

### Test coverage

20 unit tests in `tests/test_graph.py`:
- Schema validation (3 tests): node/rel table counts, IF NOT EXISTS clauses, table name correctness
- Loader initialization (3 tests): database creation, schema idempotency
- Node loading (7 tests): documents, evidence, entities with type-specific attributes, claims with support levels
- Relationship loading (7 tests): subject/object edges, supported_by, sourced_from, mentioned_in (with deduplication), superseded_by, full load_all integration

---

## Phase 5: Retrieval — Strategy-Based Graph Querying

**Output**: `ContextPack` JSON responses with ranked claims, evidence snippets, and provenance for 6 demo questions. Written to `data/outputs/context_packs.json`.

### What was built

Two files in `graph/`:

- **`graph/retrieval.py`** — `MemoryRetriever` class: accepts natural language questions, matches entities via LLM + graph search, selects a query strategy, executes Cypher traversals, scores results, and returns a `ContextPack` with ranked claims and evidence.
- **`graph/demo.py`** — CLI entry point (`python -m graph.demo`) that runs 6 demo questions and writes formatted output.

### Architecture: two-stage entity matching

**Stage 1 — LLM extraction**: The question is sent to Claude Haiku via a tool call (`identify_entities`) that extracts entity names and types. This handles natural language variations ("What's wrong with DDFP?" → extracts `DDFP` as a component).

**Stage 2 — Graph name search**: For each extracted name, all 5 entity tables are queried with `lower(n.canonical_name) CONTAINS lower($term)`. Results are deduplicated by `canonical_id` and sorted by `mention_count` (top 5 returned).

**Why two stages**: LLM extraction handles linguistic variation (abbreviations, informal names) while graph search validates against actual entities and provides the canonical IDs needed for Cypher traversal. Neither stage alone is sufficient — LLM might hallucinate entity names, and pure string matching misses paraphrased references.

### Strategy pattern for query routing

Different questions require fundamentally different graph traversals. Rather than generating Cypher dynamically (fragile, hard to debug), the retriever selects from 6 predefined strategies based on keyword rules:

| Strategy | Trigger keywords | What it queries |
|---|---|---|
| `component_deep_dive` | "wrong", "bug", "problem" + component entity | All claims where matched component is subject |
| `component_bug_scan` | "bug", "wrong" (no specific component) | AFFECTS/CAUSES current claims across all components |
| `parameter_inventory` | "parameter", "setting", "config" | HAS_PARAMETER claims with Parameter object join |
| `person_ranking` | "who", "contributor", "author" | Person nodes by mention_count with document counts |
| `evidence_chain` | "cause", "why", "root cause" + component | CAUSES claims for component with full evidence chain |
| `entity_issue_crossref` | "mention", "reference", "document" | MENTIONED_IN documents + claims involving entity |

**Why strategy pattern over dynamic Cypher**: Each strategy is a tested, optimized Cypher query pattern. Dynamic Cypher generation would need to handle schema variations (5 entity tables × different relationship names), NULL handling, and aggregation — complexity that strategies encapsulate cleanly. Adding a new strategy is a single method addition.

### Scoring formula

Each claim is scored with a weighted combination of four signals:

```
score = confidence × 0.4 + is_current × 0.3 + support_level × 0.2 + recency × 0.1
```

| Signal | Weight | Rationale |
|---|---|---|
| `confidence` | 0.4 | Most reliable signal — directly from LLM extraction quality |
| `is_current` | 0.3 | Temporal relevance — historical claims should rank lower |
| `support_level` | 0.2 | Evidence quality — claims with multiple evidence are more reliable |
| `recency` | 0.1 | Tiebreaker — recent assertions preferred over old ones |

Support level mapping: `overwhelming=1.0`, `strong=0.75`, `single=0.4`, `unsupported=0.0`.

Recency: `1.0 - (days_since_assertion / 365)`, clamped to [0, 1]. `None` → 0.5 (neutral).

### Example output

**Question**: "What's wrong with DDFP?"

- Strategy: `component_deep_dive`
- Matched 3 entities: DDFP (23 mentions), DDFP_SPIN_MIN, DDFP_SPIN_MAX
- Top claim: `DDFP --affects--> (none)` (score=0.910, confidence=0.90, strong support)
  - Evidence: quotes from 2 contributors across issue #32351 and #32288
- Second claim: `DDFP --causes--> "linearization curve must be adjusted..."` (score=0.888)
  - Evidence: 2 excerpts from lthall in issue #32288

**Question**: "Who are the main contributors?"

- Strategy: `person_ranking`
- Top 5: LupusTheCanine (14 docs), IamPete1 (12 docs), lthall (7 docs), bnsgeyer (7 docs), peterbarker (6 docs)

### Pydantic output models

Three nested models ensure type-safe, serializable output:

- `EvidenceSnippet` — evidence_id, excerpt, author, timestamp, doc_id, source_url, source_status
- `RankedClaim` — claim fields + score + list of EvidenceSnippet
- `ContextPack` — question, strategy, matched_entities, ranked claims, generated_at, query_time_ms

### Trade-offs

- **LLM call per query**: Entity matching requires one LLM call (~1s latency). Alternative: pure string matching would be faster but miss linguistic variations. The LLM call is cached in litellm if the same question is repeated.
- **Strategy selection by keywords**: Simple and debuggable but brittle for ambiguous questions. An LLM-based strategy selector would be more flexible but adds another API call. Keywords work well for the demo question set.
- **Top 10 truncation**: Limits response size for practical use. Some strategies (bug scan) find 30+ claims — the top 10 by score are the most relevant.
- **Evidence fetched per claim**: Each claim triggers a Cypher traversal for evidence. For 10 claims this is fast (<100ms total), but wouldn't scale to hundreds of claims without batching.

### Test coverage

10 unit tests in `tests/test_retrieval.py`:
- Scoring (2): current+strong beats historical+single; confidence weight = 0.4
- Recency (2): recent > old; None → 0.5
- Entity matching (1): deduplication by canonical_id
- Strategy selection (2): keyword→strategy mapping
- ContextPack (1): Pydantic model validation
- Query integration (2): max 10 claims; query_time_ms > 0

All tests use mocked kuzu and litellm — no real database or API calls.

---

## Phase 6: Visualization

**Commit**: (current)
**Files**: `viz/app.py`, `viz/export.py`, `viz/__init__.py`

### What was added

A Streamlit web UI with three tabs for exploring the memory graph interactively, plus a JSON export script for submission artifacts.

### Why this approach

A single-file Streamlit app keeps the visualization self-contained while providing immediate interactive exploration of the graph. Three tabs cover the three main use cases: visual exploration (Graph View), natural language querying (Retrieval), and quality inspection of the deduplication pipeline (Merge Inspector).

### Architecture

**Tab 1 — Graph View**: Uses `st_link_analysis` (Cytoscape.js wrapper) to render entities and claims as an interactive force-directed graph. Sidebar filters control entity types, predicates, confidence thresholds, and whether to show historical (superseded) claims. Claim nodes use a hypergraph representation: subject→claim→object, making the predicate and confidence visible on the graph. Clicking a node shows entity details, claims, and evidence excerpts.

**Tab 2 — Retrieval**: Wraps the existing `MemoryRetriever` from phase 5. Users type questions or click example buttons. The retriever performs LLM entity extraction → graph matching → strategy selection → claim scoring, displaying matched entities and ranked claims with expandable evidence.

**Tab 3 — Merge Inspector**: Browsable view of all merge decisions from the resolution phase. Three sections: entity merges (with strategy/similarity filtering), freetext claim dedup (embedding-based), and claim evidence merges. Enables auditing the deduplication quality.

**Export** (`viz/export.py`): Produces a self-contained `data/outputs/graph_export.json` with entities, claims (evidence embedded inline), and summary statistics.

### Design decisions

- **Hypergraph rendering**: Claims are intermediate nodes between subject and object entities, not simple edges. This makes multi-evidence claims, freetext objects (no target entity), and claim metadata (confidence, predicate) visible in the graph.
- **Sidebar filters**: Entity type multiselect, predicate multiselect, confidence slider, and historical toggle let users focus on specific slices of the graph without re-querying.
- **Cached data loading**: `@st.cache_resource` loads canonical JSON/JSONL files once at startup. The Kuzu database is only opened when the Retrieval tab is used (on-demand via `MemoryRetriever`).
- **`st_link_analysis` over pyvis**: Cytoscape.js provides better interactivity (node selection events, toolbar, zoom) and integrates cleanly with Streamlit's component model. pyvis generates standalone HTML files which are harder to integrate with Streamlit state.

### Results

- Graph renders 61 entity nodes + 81 claim nodes with 112 edges
- 154 entities and 81 claims loaded from canonical data
- Export produces 1,078 evidence records embedded across claims
- All 5 entity types and 5 predicates are represented in the visualization

---

## Eight Retrieval & Visualization Improvements

**Commits**: `b6f0875` through `f9b14a3` (8 commits, Mar 5)

### What was built

Eight improvements transforming the basic query-to-ranked-claims system into a rich, context-aware memory explorer:

1. **EnrichedEvidence** — Extended evidence snippets with document context (issue title, number, state, doc type, maintainer detection). Lazy doc cache avoids N+1 queries for comment/timeline parent issue lookups.

2. **SituationPack & get_entity_situation()** — Full 360-degree view of any entity: claims as subject, claims as object, related entities (2-hop via claims), mentioned-in documents, chronological timeline. Four Kuzu query phases with object entity name resolution.

3. **Situation Panel** — Replaced flat detail panel with 3-section view: "What is known" (claims grouped by predicate, current first, historical collapsed), "Why we believe it" (evidence grouped by issue, maintainer highlighting), "Related" (entities by type, doc list, conflict banner).

4. **synthesize_answer()** — LLM-powered natural language synthesis from top 5 ranked claims. System prompt enforces citation with [claim_id], 3-5 sentences, no invention. Answer shown in highlighted box above collapsed sources.

5. **Conflict Detection** — Identifies superseded claims via `Claim_SUPERSEDED_BY` traversal. Conflict model tracks older/newer claims with explanation. Conflict info fed to LLM synthesis prompt so answers acknowledge updates. Viz shows strikethrough/highlight comparison.

6. **LLM Intent Classification** — Single LLM call with tool calling replaces separate entity extraction + keyword strategy. `classify_query` tool returns strategy + entity names + optional temporal/confidence filters. Falls back to keyword matching on failure/timeout.

7. **temporal_query Strategy** — Two paths: version-specific (finds fixed_in/regressed_in claims connected to a FirmwareVersion) and general temporal (all current fix/regress claims + superseded). Results sorted by assertion_time desc.

8. **Demo Updates** — 7 demo questions including temporal query. Synthesized answers printed in demo output.

### Why these design choices

- **EnrichedEvidence over raw doc_id** — Users need context (which issue, open/closed, who said it) to evaluate evidence quality. Maintainer detection adds trust signal without extra UI. Lazy cache loads all docs once (~6960) rather than per-evidence N+1 queries.

- **Single classify_intent() call** — Reduces LLM round-trips from 2 (entity extraction + strategy selection) to 1. Tool calling with structured output gives both strategy and entity names simultaneously. 3-second timeout with keyword fallback keeps latency bounded.

- **SituationPack as a standalone model** — Decouples entity exploration from query-based retrieval. Same model serves both graph click interactions and future API endpoints. Timeline view (all claims sorted by assertion_time) enables temporal browsing without the temporal_query strategy.

- **Conflict detection at claim level** — Checking `superseded_by` is cheap (field check + optional Kuzu lookup). Feeding conflicts into synthesis prompt means the LLM naturally acknowledges outdated information. Alternative: full claim-pair comparison was rejected as too expensive for real-time.

### Trade-offs

- **Intent classification adds latency** — One extra LLM call before any graph queries. Mitigated by 3s timeout and keyword fallback.
- **Doc cache memory** — Caching all 6960 documents in-memory is acceptable for this corpus size but wouldn't scale to millions.
- **Temporal strategy is simple** — No date-range filtering or version comparison logic. Works well for "was X fixed?" but not "what changed between v4.3 and v4.5?".

### Results

- 35 unit tests pass, covering all 8 improvements
- EnrichedEvidence provides issue context for comment/timeline docs via parent issue resolution
- Intent classification correctly routes "Has the DShot arming bug been fixed?" → temporal_query
- Situation panel shows 3 sections with real claim data, evidence, and related entities
- Synthesized answers cite specific claim IDs from the ranked results
- Conflict detection identifies superseded claims and surfaces them in both viz and synthesis

---

## Fix: Replace Substring Matching with Targeted Rules

### What

Replaced the single `substring_match` rule in `SemanticEntityResolver.should_merge()` with three targeted rules:

1. **Suffix strip match** — strips generic suffixes like "module", "system", "handler" from normalized forms and merges if the stripped cores are identical (e.g., "gps module" + "gps" → merge)
2. **Namespace prefix match** — strips ArduPilot namespace prefixes (`ap_`, `ardu`) and merges if remainders match (e.g., "ap_gps" + "gps" → merge)
3. **Length-guarded substring** — only considers substring containment when the shorter string is >= 6 chars and the length ratio is >= 0.6, routing to the LLM for review instead of auto-merging

### Why

The original `substring_match` rule auto-merged any pair where one normalized form was a substring of the other. This caused cascading union-find chains through short common tokens like "ap", "rc", "baro" — producing canonical entities with 300+ aliases. For example, "AP" would match "AP_GPS", which would match "GPS", which would match "GPS Module", creating ever-growing chains of false merges.

### Trade-offs

- **May miss some valid merges** — pairs like "DShot" / "DShot ESC" (where "esc" is not a generic suffix) are no longer auto-merged. However, if the fuzzy score is high enough (>= 90), they'll still merge via `fuzzy_auto_merge`.
- **Namespace prefixes are ArduPilot-specific** — the `("ap_", "ardu")` list is domain-specific. A different corpus would need different prefixes.
- **Length-guarded substring routes to LLM** — adds LLM cost for borderline cases, but prevents false positives.

### Examples

Tests demonstrate the rules working correctly:

- `"gps module" + "gps"` → merge via `suffix_strip_match` (stripped "module" suffix)
- `"ap_gps" + "gps"` → merge via `namespace_prefix_match` (stripped "ap_" prefix)
- `"ap" + "ap_gps"` → kept distinct (empty remainder after prefix strip)
- `"baro" + "barometer"` → kept distinct (4 chars < 6 minimum for substring rule)
- `"rc" + "rc_channels"` → kept distinct (2 chars < 6 minimum)
- `"sitl" + "sitl rover"` → kept distinct (4 chars < 6 minimum)
- `"throttle" + "throttle_ctrl"` → routed to LLM via `substring_candidate` (8/13 = 0.62 ratio)

---

## Entity Resolution Redesign — Structural Rules + Corpus Evidence

**Commit**: entity-resolution-rewrite branch

### Problem

The original `SemanticEntityResolver.should_merge()` used rapidfuzz `token_set_ratio` with automatic merge at >=90 and LLM candidates at 75-89. This caused false merges in the ArduPilot corpus where many component names share tokens:

- "SITL" and "SITL Rover" merged (token_set_ratio ≈ 100) — SITL is a simulator, SITL Rover is a specific vehicle configuration
- "Rover" and "SITL Rover" merged — completely different scopes
- "GPS" and "GPS Yaw" at risk — GPS is a subsystem, GPS Yaw is a specific feature

Fuzzy string matching is fundamentally unsafe for a domain where entity names are compositional (prefix + qualifier patterns).

### What changed

Four-phase redesign replacing all fuzzy matching with structural rules, corpus evidence, and context-aware LLM decisions.

#### Phase 1: Structural-only merging

Replaced the fuzzy decision tree with a strict 5-step structural decision tree:

1. **Numeric variant guard** — "EKF" vs "EKF3" → always keep distinct (no longer fuzzy-merged)
2. **Exact normalized match** — case-insensitive exact match → merge
3. **Structural key match** — strip namespace prefix (`ap_`, `ardu`) and generic suffixes (`module`, `system`, etc.), compare stripped forms. "AP_GPS" and "GPS Module" both reduce to "gps" → merge
4. **Co-document Levenshtein-1** — edit distance of 1 AND same document → LLM candidate (catches typos like "Compas" / "Compass"). Different documents → distinct (too risky without context)
5. **Shared prefix co-document** — structural keys sharing >=4 char prefix AND same document → LLM candidate

Removed `rapidfuzz` dependency entirely. Added a simple O(mn) `_edit_distance()` static method for short entity name comparisons.

#### Phase 2: ALIAS_OF corpus evidence

Added a new `ALIAS_OF` predicate to the extraction schema. The LLM extracts alias claims when text contains explicit evidence: parenthetical aliases ("EKF (Extended Kalman Filter)"), slash notation, "also called", "formerly known as".

In the resolution pipeline, a new alias resolution pass runs between entity resolution and claim resolution:
- For each ALIAS_OF claim, resolves subject → canonical entity A and object_freetext → canonical entity B
- If A and B are different entities, absorbs B into A (merges mention_ids, aliases, timestamps)
- ALIAS_OF claims are filtered out before claim resolution (they serve as merge evidence, not queryable claims)

#### Phase 3: Context-aware LLM merging

Added `context_sentence` field to `EntityMention` — the exact sentence from the document body where the entity appears. The LLM extraction prompt now requests this for each entity.

Updated `MergeResolver` to include context sentences in the LLM merge prompt, enabling the model to make more informed decisions about whether two entity names refer to the same real-world subsystem.

#### Phase 4: Coherence splitting

Added `CoherenceSplitter` that detects entities whose claims form disconnected semantic clusters:
- For each entity with >= 5 claims, embeds claim descriptions using the existing `Embedder`
- Builds a similarity graph (edge if cosine similarity >= 0.3)
- Finds connected components via BFS
- Entities with 2+ components are flagged as split candidates
- Results written to `data/reports/split_candidates.json`

This is a diagnostic tool — it doesn't modify the data, just flags entities that may have been over-merged.

### Why this approach

1. **Structural rules are deterministic and auditable** — no threshold tuning, no score distributions to monitor
2. **Corpus evidence (ALIAS_OF) grounds merges in explicit textual evidence** — the LLM saw "EKF (Extended Kalman Filter)" in the document and extracted it as an alias claim
3. **Co-document requirement for fuzzy matches** — entities mentioned in the same document are more likely to be typo variants; entities in different documents are more likely to be genuinely different
4. **Coherence splitting catches over-merging** — even if the structural rules merge too aggressively, the splitter can detect it post-hoc

### Trade-offs

- **More conservative merging** — some valid merges (e.g., "CubeOrange" / "Cube Orange") now require LLM confirmation instead of auto-merging. This increases LLM cost but eliminates false positives.
- **ALIAS_OF depends on extraction quality** — if the LLM fails to extract alias claims, those merges won't happen. However, this is a feature: we only merge when there's explicit evidence.
- **Coherence splitter is diagnostic only** — it flags but doesn't fix. Manual review is required.
- **No fuzzy fallback** — entities that don't match structurally and aren't in the same document will remain distinct. This is intentional: false positives corrupt the graph, false negatives can be fixed with additional extraction.

### Pipeline order

1. EntityResolver (structural matching)
2. Alias resolution (ALIAS_OF claim merging)
3. ClaimResolver (per-predicate grouping)
4. FreetextClaimDeduper (embedding-based claim dedup)
5. TemporalResolver (valid_from/valid_until from version entities)
6. SupersedingResolver (marks older REGRESSED_IN claims)
7. CoherenceSplitter → `data/reports/split_candidates.json`
8. Write canonical outputs
9. CoherenceReport → `coherence_report.json`

### Test results

44 unit tests covering all 4 phases:
- Phase 1: 10 structural merging tests (exact match, suffix strip, namespace prefix, numeric variant, Levenshtein-1, SITL/Rover distinct, etc.)
- Phase 2: 2 alias resolution tests (merge via alias, no-match safety)
- Phase 3: 3 LLM context tests (context in prompt, reasoning recorded, failure handling)
- Phase 4: 4 coherence splitter tests (split detection, coherent not flagged, threshold, structure)

---

## Phase 3b: UpdateMonitor + grounding_status

**Commit**: `5f9c30a`

### What was added

**`UpdateMonitor`** (`monitoring/update_monitor.py`) — a trustworthiness auditing component that checks whether each `EvidenceRecord`'s source document still matches the hash captured at extraction time, then propagates a `grounding_status` to canonical claims.

**`grounding_status`** field added to `CanonicalClaim` (default `"fully_grounded"`).

### Why this approach

GitHub issues can be edited or deleted after extraction. An evidence record extracted from a document that has since been modified or deleted may no longer be verifiable — the claim it supports could be wrong, outdated, or unconfirmable. The `UpdateMonitor` makes this staleness visible without requiring a full re-extraction.

### What problem it solves

**Trustworthiness auditing**: the memory graph can be queried not just by topic but by source freshness. A claim with `grounding_status = "ungrounded"` signals that all its evidence has been deleted; `"partially_grounded"` means some sources remain; `"edited_excerpt_intact"` means the document changed but the specific excerpt is still present (the claim likely still holds).

### How it works

1. **`_resolve_doc_body(doc_id)`** — maps doc_id format to the appropriate raw file:
   - `owner/repo#{n}` → `data/raw/issues/{n}.json` → `body`
   - `...#{n}/c/{cid}` → `data/raw/comments/{n}.json` list → item matching `id == cid`
   - `...#{n}/t/{eid}` → `data/raw/timeline/{n}.json` list → item matching `id == eid`
   - Returns `None` if file is missing (→ `"deleted"`)

2. **`scan_evidence()`** — groups evidence by `doc_id`, computes `sha1(body)`, compares to `source_hash` stored at extraction time:
   - Hash matches → `"active"`
   - Hash differs, excerpt still in body → `"edited_excerpt_intact"`
   - Hash differs, excerpt gone → `"edited_excerpt_removed"`
   - File missing → `"deleted"`
   - Uses a JSON cache (`scan_cache_path`) keyed by `doc_id → current_hash` to skip unchanged docs on repeated runs.

3. **`propagate_to_claims()`** — maps evidence statuses to claim-level `grounding_status` using priority logic:
   - Any active → all active → `"fully_grounded"`; mixed → `"partially_grounded"`
   - No active, some intact → `"edited_excerpt_intact"`
   - No active/intact, some removed → `"unverifiable"`
   - All deleted → `"ungrounded"`
   - No evidence found in results → `"fully_grounded"` (default, safe)

4. **`generate_report()`** — writes `data/reports/update_monitor_report.json` with status distributions, deleted/edited doc lists, and `grounding_status_distribution`.

5. **`monitoring/quality.py`** — `generate_coherence_report()` now includes `grounding_status_distribution` in its summary section, so the coherence report reflects source freshness alongside entity and claim statistics.

### Trade-offs

- **Excerpt matching is exact substring** — a minor edit to the surrounding paragraph won't invalidate an excerpt, but a paraphrase of the excerpt itself will classify as `"edited_excerpt_removed"`. This is conservative: we'd rather flag uncertainty than silently trust a changed document.
- **Cache is per doc_id, not per evidence record** — if any record in a doc was extracted from an older version, all records get re-checked on the next run after the cache is invalidated by a hash change. This is correct: hash mismatch means the doc changed, so every record needs re-evaluation.
- **`grounding_status` lives on `CanonicalClaim`** — not on individual evidence records. This is intentional: the claim is the queryable unit; per-evidence status is internal to the monitor.

### Test results

12 unit tests in `tests/test_update_monitor.py`, all passing in 0.04s:
1. Active document (hash matches) → `"active"`
2. Edited, excerpt present → `"edited_excerpt_intact"`
3. Edited, excerpt removed → `"edited_excerpt_removed"`
4. Missing file → `"deleted"`
5. All active evidence → claim `"fully_grounded"`
6. Mixed active/deleted → `"partially_grounded"`
7. All deleted → `"ungrounded"`
8. All edited_excerpt_removed → `"unverifiable"`
9. Cache skip: second run writes cache correctly and returns `"active"` without re-hashing
10. Idempotency: two consecutive runs produce identical status and `changed` flag
11. Report JSON has all required fields (`total_evidence_scanned`, `changed_count`, `status_distribution`, `grounding_status_distribution`, `deleted_doc_count`, `edited_doc_count`)
12. Claims file written with `grounding_status` on all claims
