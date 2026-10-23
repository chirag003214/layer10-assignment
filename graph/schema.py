from __future__ import annotations

"""DDL constants for the Kuzu property-graph schema.

8 node tables (5 entity types + Claim + Evidence + Document)
18 relationship tables.
"""

ENTITY_TYPES = ["Person", "Component", "Parameter", "FirmwareVersion", "HardwarePlatform"]

NODE_TABLES: list[str] = [
    # ── entity node tables ────────────────────────────────────────
    """CREATE NODE TABLE IF NOT EXISTS Person(
        canonical_id STRING,
        canonical_name STRING,
        github_handle STRING,
        aliases STRING[],
        mention_count INT64,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        mention_source STRING,
        PRIMARY KEY (canonical_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Component(
        canonical_id STRING,
        canonical_name STRING,
        aliases STRING[],
        mention_count INT64,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        mention_source STRING,
        PRIMARY KEY (canonical_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Parameter(
        canonical_id STRING,
        canonical_name STRING,
        value STRING,
        aliases STRING[],
        mention_count INT64,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        PRIMARY KEY (canonical_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS FirmwareVersion(
        canonical_id STRING,
        canonical_name STRING,
        version_string STRING,
        vehicle_type STRING,
        aliases STRING[],
        mention_count INT64,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        PRIMARY KEY (canonical_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS HardwarePlatform(
        canonical_id STRING,
        canonical_name STRING,
        aliases STRING[],
        mention_count INT64,
        first_seen TIMESTAMP,
        last_seen TIMESTAMP,
        mention_source STRING,
        PRIMARY KEY (canonical_id)
    )""",
    # ── claim / evidence / document ───────────────────────────────
    """CREATE NODE TABLE IF NOT EXISTS Claim(
        canonical_claim_id STRING,
        predicate STRING,
        object_freetext STRING,
        object_freetext_variants STRING[],
        confidence DOUBLE,
        support_level STRING,
        is_current BOOLEAN,
        valid_from STRING,
        valid_until STRING,
        assertion_time TIMESTAMP,
        event_time STRING,
        superseded_by STRING,
        PRIMARY KEY (canonical_claim_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Evidence(
        evidence_id STRING,
        excerpt STRING,
        significance STRING,
        char_offset_start INT64,
        char_offset_end INT64,
        timestamp TIMESTAMP,
        author STRING,
        source_hash STRING,
        source_status STRING,
        PRIMARY KEY (evidence_id)
    )""",
    """CREATE NODE TABLE IF NOT EXISTS Document(
        doc_id STRING,
        doc_type STRING,
        issue_number INT64,
        title STRING,
        author STRING,
        created_at TIMESTAMP,
        state STRING,
        source_url STRING,
        PRIMARY KEY (doc_id)
    )""",
]

REL_TABLES: list[str] = [
    # ── {Type}_HAS_SUBJECT  (entity → Claim) ─────────────────────
    *[
        f"CREATE REL TABLE IF NOT EXISTS {t}_HAS_SUBJECT(FROM {t} TO Claim)"
        for t in ENTITY_TYPES
    ],
    # ── Claim_HAS_OBJECT_{Type}  (Claim → entity) ────────────────
    *[
        f"CREATE REL TABLE IF NOT EXISTS Claim_HAS_OBJECT_{t}(FROM Claim TO {t})"
        for t in ENTITY_TYPES
    ],
    # ── evidence / document relationships ─────────────────────────
    "CREATE REL TABLE IF NOT EXISTS SUPPORTED_BY(FROM Claim TO Evidence)",
    "CREATE REL TABLE IF NOT EXISTS SOURCED_FROM(FROM Evidence TO Document)",
    # ── {Type}_MENTIONED_IN  (entity → Document) ─────────────────
    *[
        f"CREATE REL TABLE IF NOT EXISTS {t}_MENTIONED_IN(FROM {t} TO Document)"
        for t in ENTITY_TYPES
    ],
    # ── claim supersession ────────────────────────────────────────
    "CREATE REL TABLE IF NOT EXISTS Claim_SUPERSEDED_BY(FROM Claim TO Claim)",
]

# mapping from extraction entity_type string → Kuzu node table name
ENTITY_TYPE_TO_TABLE: dict[str, str] = {
    "person": "Person",
    "component": "Component",
    "parameter": "Parameter",
    "firmware_version": "FirmwareVersion",
    "hardware_platform": "HardwarePlatform",
}
