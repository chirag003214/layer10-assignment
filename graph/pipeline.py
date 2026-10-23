from __future__ import annotations

"""CLI entry point: python -m graph.pipeline

Loads all resolved data into a Kuzu property graph at data/graph/memory.kuzu.
"""

import logging
import shutil
from pathlib import Path

from graph.loader import GraphLoader
from graph.schema import ENTITY_TYPES

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/graph/memory.kuzu"


def run(*, clean: bool = False) -> None:
    db_dir = Path(DB_PATH)
    if clean and db_dir.exists():
        if db_dir.is_dir():
            shutil.rmtree(db_dir)
        else:
            db_dir.unlink()
        logger.info("Removed existing database at %s", DB_PATH)

    db_dir.parent.mkdir(parents=True, exist_ok=True)

    loader = GraphLoader(DB_PATH)
    loader.init_schema()
    loader.load_all()

    # ── summary counts ────────────────────────────────────────────
    print("\n=== Graph Summary ===")
    node_tables = ENTITY_TYPES + ["Claim", "Evidence", "Document"]
    for table in node_tables:
        result = loader.conn.execute(f"MATCH (n:{table}) RETURN count(n)")
        count = result.get_next()[0]
        print(f"  {table:20s} {count:>6} nodes")

    rel_names = (
        [f"{t}_HAS_SUBJECT" for t in ENTITY_TYPES]
        + [f"Claim_HAS_OBJECT_{t}" for t in ENTITY_TYPES]
        + ["SUPPORTED_BY", "SOURCED_FROM"]
        + [f"{t}_MENTIONED_IN" for t in ENTITY_TYPES]
        + ["Claim_SUPERSEDED_BY"]
    )
    print()
    for rname in rel_names:
        result = loader.conn.execute(f"MATCH ()-[r:{rname}]->() RETURN count(r)")
        count = result.get_next()[0]
        if count > 0:
            print(f"  {rname:35s} {count:>6} edges")

    # ── verification queries ──────────────────────────────────────
    print("\n=== Verification Queries ===")

    # 1. Top entities by mention count
    print("\n-- Top 5 entities by mention count --")
    for table in ENTITY_TYPES:
        result = loader.conn.execute(
            f"MATCH (n:{table}) RETURN n.canonical_name, n.mention_count "
            f"ORDER BY n.mention_count DESC LIMIT 3"
        )
        while result.has_next():
            row = result.get_next()
            print(f"  {table:20s} {row[0]:30s} mentions={row[1]}")

    # 2. Claims with most evidence
    print("\n-- Claims with most evidence (SUPPORTED_BY) --")
    result = loader.conn.execute(
        "MATCH (c:Claim)-[r:SUPPORTED_BY]->(e:Evidence) "
        "RETURN c.canonical_claim_id, c.predicate, count(e) AS n_ev "
        "ORDER BY n_ev DESC LIMIT 5"
    )
    while result.has_next():
        row = result.get_next()
        print(f"  {row[0]}  predicate={row[1]}  evidence_count={row[2]}")

    # 3. Full traversal: entity → claim → evidence → document
    print("\n-- Sample full path: entity → claim → evidence → document --")
    result = loader.conn.execute(
        "MATCH (comp:Component)-[:Component_HAS_SUBJECT]->(cl:Claim)"
        "-[:SUPPORTED_BY]->(ev:Evidence)-[:SOURCED_FROM]->(doc:Document) "
        "RETURN comp.canonical_name, cl.predicate, ev.evidence_id, doc.doc_id "
        "LIMIT 5"
    )
    while result.has_next():
        row = result.get_next()
        print(f"  {row[0]} --{row[1]}--> [ev:{row[2]}] --> doc:{row[3]}")


if __name__ == "__main__":
    run(clean=True)
