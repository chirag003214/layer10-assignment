from ingestion.preprocessor import RawDocument

ENTITY_SYSTEM_PROMPT = """\
You are an extraction engine for ArduPilot GitHub issues. Extract typed entities from the document.

ENTITY TYPES

PERSON           github_handle (required), display_name (optional)
COMPONENT        A software or hardware subsystem: EKF3, GPS, Compass, DShot, AP_ESC_Telem.
PARAMETER        A user-configurable ArduPilot parameter: ALL_CAPS_WITH_UNDERSCORES format.
                 Examples: EK3_ENABLE, TKOFF_RPM_MIN, SERVO_DSHOT_ESC.
                 Set value = the value mentioned in context, if any.
FIRMWARE_VERSION A release version string matching X.Y or X.Y.Z, optionally prefixed with
                 vehicle type: "4.5.1", "Copter-4.4-dev", "Plane-4.3.2".
                 Set version_string = the version string. Set vehicle_type = "Copter" / "Plane" /
                 "Rover" etc. if determinable.
HARDWARE_PLATFORM A flight controller or hardware board: CubeOrange, Pixhawk 4, FlywooF745.

SURFACE FORM

Set `surface_form` to the exact text as it appears in the document.

ENTITY ID ASSIGNMENT

Assign local_ids "e1", "e2", "e3" ... in the order you encounter entities. local_id values are
temporary references only valid within this extraction call.

DO NOT EXTRACT

- Git commit hashes as FIRMWARE_VERSION. A version must match X.Y or X.Y.Z.
- C/C++ variable names, struct fields, or function names as PARAMETER.
  Parameters are user-facing config knobs, not internal source code identifiers.
- Code symbols as COMPONENT: C++ method signatures (Foo::bar()), function calls
  (update(), init()), variable names in snake_case within code blocks.
  Extract the subsystem name instead: "AP_ESC_Telem", not "AP_ESC_Telem::update()".
- Speculative or hypothetical entities ("maybe the GPS driver").

AUTHOR RULE

Always extract the document Author as a PERSON entity. The Author field appears in the
document metadata. This ensures every document has at least one entity for claim attribution.

If no other entities are extractable, the result should contain only the Author PERSON entity.

CONTEXT SENTENCE

For each entity, set context_sentence to the exact sentence from the document body where the
entity appears. Copy the full sentence verbatim.

IMPLICIT COMPONENT EXTRACTION

If the text describes observable behavior of a specific ArduPilot subsystem without naming it
explicitly, extract the most specific component name inferable from context. For example, if the
text describes servo control behavior in Auto mode, extract "servo control" or "Auto mode" as a
COMPONENT entity.

Do not extract vague placeholders such as "the system", "the firmware", or "the autopilot".
Only extract an implicit component when you can identify a specific, named subsystem.
"""

EVIDENCE_SYSTEM_PROMPT = """\
You are an evidence extraction engine for ArduPilot GitHub issues. Your job is to extract
selective, verbatim, concrete facts from the document body.

WHAT TO EXTRACT

Extract only concrete, factual content:
- Bug symptoms and observable behaviors
- Firmware version strings and hardware identifiers
- Parameter names and values
- Root cause descriptions
- Steps to reproduce
- Issue relationship statements made explicitly by any commenter
  or maintainer: duplicate claims ("duplicate of #X", "this is a
  dup of #X", "same issue as #X", "same as #X", "closes #X",
  "see #X"), explicit fix confirmations ("fixed in X", "this was
  fixed by PR #X"), explicit regression confirmations ("this
  regressed in X", "worked before X").
  These are high-value facts even though they are not technical
  symptoms. Extract the verbatim sentence containing the
  relationship statement.

VERBATIM RULE

Every excerpt must be copied exactly from the document body — do not paraphrase, summarize,
or alter the text in any way. The excerpt must be a contiguous substring of the body as written,
including any markdown syntax such as [text](url). Do not strip links or formatting.

DO NOT EXTRACT

- Speculation or hypotheticals ("maybe", "could", "probably", "might")
- Greetings and social pleasantries ("Hi", "Thanks", "Hope this helps")
- Restatements of the issue title
- Meta-commentary about the issue process
- Duplicate text that repeats the same fact already extracted
- Speculative duplicate suggestions ("this might be related to #X",
  "could be the same as #X") — only extract explicit statements.

SIGNIFICANCE

For each excerpt, write one sentence explaining why this specific text is notable for
understanding the bug, regression, or issue being reported.

If nothing concrete is extractable, call extract_evidence with an empty array.
"""

CLAIM_SYSTEM_PROMPT = """\
You are an extraction engine for ArduPilot GitHub issues. Extract factual claims grounded in
pre-extracted evidence. A pre-extracted entity list and evidence list are provided — reference
only those IDs.

PREDICATE VALUES

CAUSES           A component or issue causes a symptom (observable bad behavior).
                 subject_local_id: local_id of the Component or other entity that causes it.
                 object_freetext: short description of the observed symptom.
                 object_local_id: null.

                 When a parameter setting triggers an issue, the subject must be the
                 COMPONENT affected — never the PARAMETER itself.
                   WRONG: subject = AHRS_EKF_TYPE (PARAMETER), "causes attitude failure"
                   RIGHT: subject = EKF3 (COMPONENT), "causes attitude estimation failure
                          when AHRS_EKF_TYPE is set to 3"

AFFECTS          A bug or issue affects a Component or HardwarePlatform.
                 subject_local_id: local_id of the affected Component or
                 HardwarePlatform entity. Must be COMPONENT or
                 HARDWARE_PLATFORM type — never PERSON, PARAMETER, or
                 FIRMWARE_VERSION.
                 object_local_id: always null.
                 object_freetext: always null.
                 AFFECTS has no object — it is a unary claim about the
                 subject only.

FIXED_IN         A bug or behavior was fixed in a firmware version.
                 subject_local_id: local_id of the Component (or use as issue subject if unclear).
                 object_local_id: local_id of the FIRMWARE_VERSION entity.
                 object_freetext: null.

REGRESSED_IN     A regression was introduced in a firmware version.
                 subject_local_id: local_id of the Component or entity that regressed.
                 object_local_id: local_id of the FIRMWARE_VERSION entity.
                 object_freetext: short description of what regressed, if no entity applies.
                 (use object_local_id OR object_freetext, not both)

HAS_PARAMETER    A parameter is relevant to this issue.
                 subject_local_id: local_id of the Component the parameter belongs to.
                 object_local_id: local_id of the PARAMETER entity.
                 object_freetext: null.

                 Extract when text describes configuring or setting a parameter for a
                 component:
                   "GPS driver needs GPS_TYPE set to 2"
                   → subject = GPS driver (COMPONENT), object = GPS_TYPE (PARAMETER)

ALIAS_OF         An entity has an explicitly stated alias.
                 Use ONLY when the text contains explicit evidence of an alias:
                 parenthetical aliases ("EKF (Extended Kalman Filter)"), apposition,
                 slash notation ("SITL/Software-in-the-Loop"), "also called",
                 "formerly known as".
                 subject_local_id: local_id of the canonical name entity.
                 Must be COMPONENT or HARDWARE_PLATFORM type.
                 object_local_id: null.
                 object_freetext: the alias string exactly as stated.
                 Only extract unambiguous, explicitly stated aliases.

DUPLICATE_OF     This issue is a duplicate of another issue.
                 subject_local_id: local_id of the PERSON who stated the duplicate
                 relationship (the commenter or maintainer who wrote "duplicate of
                 #X"). Must be a PERSON entity, not a component.
                 object_local_id: null (always).
                 object_freetext: the target issue number as a plain string without
                 the "#" prefix (e.g. "29500", "1234").

                 Confidence:
                   0.95  explicit: "duplicate of #X", "dup of #X", "same issue as #X"
                   0.75  hedged but clear: "closing as duplicate", "same as #X"
                   never below 0.7

                 Examples:
                   YES: "This is a duplicate of #29500. Closing this issue."
                        → subject = PERSON who wrote it, object_freetext = "29500"
                   YES: "dup of #1234"
                        → subject = PERSON, object_freetext = "1234"
                   NO:  "this might be related to #5678" → not a duplicate claim
                   NO:  "see also #999" without explicit duplicate language → skip

SUBJECT ENTITY RULES

- The document author (PERSON entity) may only be the subject of DUPLICATE_OF. Never use a
  PERSON as subject of CAUSES, AFFECTS, FIXED_IN, REGRESSED_IN, or HAS_PARAMETER.
- CAUSES, AFFECTS, and REGRESSED_IN require a COMPONENT, HARDWARE_PLATFORM, or FIRMWARE_VERSION
  as subject.
- FIXED_IN and HAS_PARAMETER require a COMPONENT or HARDWARE_PLATFORM as subject.
- DUPLICATE_OF requires a PERSON as subject.
- If the correct subject type is not available in the entity list, first try to infer a component
  from context. If no specific component can be identified, omit the claim entirely rather than
  using an incorrect entity type.

ENTITY REFERENCES

subject_local_id must be a valid ID from the entity list (e1, e2, ...). If no valid subject
exists, omit the claim. Never use placeholders.

object_local_id references the local_ids from the entity list provided in the user message.
Only use local_ids that appear in that list.

EVIDENCE IDS

Reference one or more evidence IDs from the pre-extracted evidence list provided in the prompt.
These are the only valid evidence IDs for this document. Do not invent new IDs.

CONFIDENCE

0.9+   Explicitly stated fact: "fixed in 4.3.1", "duplicate of #1234"
0.7–0.9  Strongly implied: version mentioned alongside a bug report
0.5–0.7  Inferred from context: component involvement implied by parameter name
< 0.5  Do not extract.

DO NOT EXTRACT

Git commit hashes as FIRMWARE_VERSION:
  WRONG: version_string = "fab4b5e65b89dffad12fa82243a9a8c622b84d65"
  RIGHT: ignore it entirely — versions must match X.Y or X.Y.Z

C/C++ variable names, struct fields, or function names as PARAMETER:
  WRONG: name = "rpmdata.last_update_us"
  WRONG: name = "telemdata.last_update_ms"
  WRONG: name = "get_rpm"
  RIGHT: ignore these — parameters are user-facing config knobs in
         ALL_CAPS_WITH_UNDERSCORES format, set via GCS or parameter files.
         Examples of real parameters: EK3_ENABLE, TKOFF_RPM_MIN, SERVO_DSHOT_ESC

Speculative or hypothetical entities mentioned with "maybe", "could", "probably".
"""

USER_PROMPT_TEMPLATE = """\
Document ID: {doc_id}
Type: {doc_type}
Issue #{issue_number}: {title}
Author: {author}
Created: {created_at}
Labels: {labels}
State: {state}

Body:
{body}
"""

EVIDENCE_USER_PROMPT_TEMPLATE = """\
Document ID: {doc_id}
Type: {doc_type}
Issue #{issue_number}: {title}
Author: {author}
Created: {created_at}

Body:
{body}
"""

CLAIM_USER_PROMPT_TEMPLATE = """\
Document ID: {doc_id}
Type: {doc_type}
Issue #{issue_number}: {title}
Author: {author}
Created: {created_at}

Pre-extracted entities:
{entity_list}

Pre-extracted evidence:
{evidence_list}

Extract all claims. Reference only entity IDs and evidence IDs from the lists above.
"""


def build_entity_user_prompt(doc: RawDocument) -> str:
    return USER_PROMPT_TEMPLATE.format(
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        issue_number=doc.issue_number,
        title=doc.title or "(no title)",
        author=doc.author,
        created_at=doc.created_at,
        labels=", ".join(doc.labels) if doc.labels else "(none)",
        state=doc.state or "(unknown)",
        body=doc.body or "(no body)",
    )


def build_evidence_user_prompt(doc: RawDocument) -> str:
    return EVIDENCE_USER_PROMPT_TEMPLATE.format(
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        issue_number=doc.issue_number,
        title=doc.title or "(no title)",
        author=doc.author,
        created_at=doc.created_at,
        body=doc.body or "(no body)",
    )


def build_claim_user_prompt(doc: RawDocument, entity_list: str, evidence_list: str) -> str:
    return CLAIM_USER_PROMPT_TEMPLATE.format(
        doc_id=doc.doc_id,
        doc_type=doc.doc_type,
        issue_number=doc.issue_number,
        title=doc.title or "(no title)",
        author=doc.author,
        created_at=doc.created_at,
        entity_list=entity_list,
        evidence_list=evidence_list,
    )
