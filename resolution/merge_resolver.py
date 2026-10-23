from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Literal

import litellm

import config
from extraction.schema import EntityMention

logger = logging.getLogger(__name__)

MERGE_SYSTEM_PROMPT = """\
You are an entity resolution engine for ArduPilot GitHub issues.
Your task is to decide whether two extracted entity mentions refer
to the same real-world entity and should be merged into one
canonical entity in a memory graph.

ArduPilot is an open-source autopilot system. Entities include
software subsystems (EKF, GPS, DShot), hardware platforms
(Pixhawk, CubeOrange), and configuration parameters.

You will receive the surface form of each entity AND the sentence
in which it appeared. Use the sentence context to determine if both
names refer to the same real-world subsystem.

MERGE if:
- The two mentions clearly refer to the same subsystem, component,
  or hardware, even if named differently.
  Examples that SHOULD merge:
  "lua scripting" and "scripting backend" — same feature, different
  perspective (user-facing vs internal name)
  "CubeOrange" and "Cube Orange" — same hardware, spacing difference

KEEP DISTINCT if:
- The mentions refer to related but genuinely different things.
  Examples that should NOT merge:
  "serial0" and "serial1" — different hardware ports
  "matek h743 mini" and "matek h743 slim" — different board variants
  "EKF2" and "EKF3" — different algorithm versions
  "GPS" and "GPS Yaw" — GPS is a subsystem, GPS Yaw is a specific
  feature within it
  "SITL" and "SITL Rover" — SITL is a simulator, SITL Rover is a
  specific vehicle configuration

Respond with a merge decision, a confidence score, and a one-sentence
reasoning. Be conservative — when in doubt, keep distinct. A missed
merge is safer than an incorrect merge.\
"""

_USER_PROMPT_TEMPLATE = """\
Entity type: {entity_type}

Entity A: "{surface_form_a}"  (normalized: "{normalized_form_a}")
Entity A context: "{context_sentence_a}"
Entity B: "{surface_form_b}"  (normalized: "{normalized_form_b}")
Entity B context: "{context_sentence_b}"
Similarity score: {score} / 100

Additional context:
Entity A appears in documents: {doc_ids_a}
Entity B appears in documents: {doc_ids_b}

Should these be merged into one canonical entity?
"""


@dataclass
class MergeCandidate:
    mention_a: EntityMention
    mention_b: EntityMention
    similarity_score: float


@dataclass
class MergeDecision:
    mention_id_a: str
    mention_id_b: str
    similarity_score: float
    decision: Literal["merge", "keep_distinct"]
    confidence: float
    reasoning: str
    strategy: Literal["llm_merge_decision"]


def get_merge_decision_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "record_merge_decision",
            "description": "Record the merge decision for this entity pair.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["merge", "keep_distinct"],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence explaining the decision.",
                    },
                },
                "required": ["decision", "confidence", "reasoning"],
            },
        },
    }


class MergeResolver:
    def resolve(self, candidates: list[MergeCandidate]) -> list[MergeDecision]:
        return [self._resolve_one(c) for c in candidates]

    def _resolve_one(self, candidate: MergeCandidate) -> MergeDecision:
        ma = candidate.mention_a
        mb = candidate.mention_b

        doc_ids_a = ", ".join(sorted({ma.doc_id})[:3])
        doc_ids_b = ", ".join(sorted({mb.doc_id})[:3])

        context_a = getattr(ma, "context_sentence", None) or "N/A"
        context_b = getattr(mb, "context_sentence", None) or "N/A"

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            entity_type=ma.entity_type.value,
            surface_form_a=ma.surface_form,
            normalized_form_a=ma.normalized_form,
            context_sentence_a=context_a,
            surface_form_b=mb.surface_form,
            normalized_form_b=mb.normalized_form,
            context_sentence_b=context_b,
            score=int(candidate.similarity_score),
            doc_ids_a=doc_ids_a,
            doc_ids_b=doc_ids_b,
        )

        try:
            response = litellm.completion(
                messages=[
                    {"role": "system", "content": MERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=200,
                tools=[get_merge_decision_tool()],
                tool_choice="required",
                **config.LLM_KWARGS,
            )

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("No tool calls in LLM response")

            args = json.loads(tool_calls[0].function.arguments)
            decision = args["decision"]
            confidence = float(args["confidence"])
            reasoning = args["reasoning"]

            if decision not in ("merge", "keep_distinct"):
                raise ValueError(f"Unexpected decision value: {decision!r}")

        except Exception as exc:
            logger.error(
                "LLM merge decision failed for %s / %s: %s",
                ma.mention_id,
                mb.mention_id,
                exc,
            )
            return MergeDecision(
                mention_id_a=ma.mention_id,
                mention_id_b=mb.mention_id,
                similarity_score=candidate.similarity_score,
                decision="keep_distinct",
                confidence=0.0,
                reasoning="llm_error",
                strategy="llm_merge_decision",
            )

        return MergeDecision(
            mention_id_a=ma.mention_id,
            mention_id_b=mb.mention_id,
            similarity_score=candidate.similarity_score,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            strategy="llm_merge_decision",
        )
