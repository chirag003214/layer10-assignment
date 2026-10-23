from __future__ import annotations

"""Demo: run 6 queries against the memory graph and write ContextPacks.

Usage: python -m graph.demo
"""

import json
import logging
from pathlib import Path

from graph.retrieval import MemoryRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/graph/memory.kuzu"
OUTPUT_DIR = Path("data/outputs")

DEMO_QUESTIONS = [
    "What's wrong with DDFP?",
    "What components have known bugs?",
    "What parameters does OSD5 have?",
    "Who are the active contributors?",
    "Why do we believe DDFP has this bug?",
    "Which issues involve AM32 ESCs?",
    "Has the DShot arming bug been fixed?",
]


def _print_separator(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_pack(pack) -> None:
    print(f"  Strategy:  {pack.strategy}")
    print(f"  Time:      {pack.query_time_ms:.0f}ms")
    print(f"  Entities:  {len(pack.matched_entities)}")
    print(f"  Claims:    {len(pack.claims)}")

    if pack.synthesized_answer:
        print(f"\n  Answer:")
        for line in pack.synthesized_answer.split("\n"):
            print(f"    {line}")

    if pack.matched_entities:
        print("\n  Matched entities:")
        for ent in pack.matched_entities:
            print(f"    - {ent.get('name')} ({ent.get('type')}, mentions={ent.get('mention_count', '?')})")

    if pack.claims:
        print("\n  Top claims:")
        for i, claim in enumerate(pack.claims[:5], 1):
            current = "CURRENT" if claim.is_current else "historical"
            print(f"\n    [{i}] {claim.subject_name} --{claim.predicate}--> "
                  f"{claim.object_name or claim.object_freetext or '(none)'}")
            print(f"        confidence={claim.confidence:.2f}  support={claim.support_level}  "
                  f"{current}  score={claim.score:.3f}")

            if claim.evidence:
                for ev in claim.evidence[:2]:
                    excerpt = (ev.excerpt[:80] + "...") if ev.excerpt and len(ev.excerpt) > 80 else ev.excerpt
                    print(f"        evidence: \"{excerpt}\"")
                    print(f"                 by {ev.author} in {ev.doc_id}")
    else:
        print("\n  (no claims found)")


def run() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    retriever = MemoryRetriever(DB_PATH)
    packs = []

    for question in DEMO_QUESTIONS:
        _print_separator(question)
        pack = retriever.query(question)
        _print_pack(pack)
        packs.append(pack.model_dump())

    # Write all ContextPacks to JSON
    output_path = OUTPUT_DIR / "context_packs.json"
    with open(output_path, "w") as f:
        json.dump(packs, f, indent=2, default=str)
    print(f"\n\nWrote {len(packs)} context packs to {output_path}")


if __name__ == "__main__":
    run()
