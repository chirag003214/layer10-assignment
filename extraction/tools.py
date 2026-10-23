"""LiteLLM tool definitions for the extraction layer."""


def get_entity_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "extract_entities",
            "description": (
                "Extract all named entities from the document. "
                "Call this tool with every entity you identify."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Entity"},
                    }
                },
                "required": ["entities"],
                "$defs": {
                    "PersonAttributes": {
                        "type": "object",
                        "properties": {
                            "github_handle": {"type": "string"},
                            "display_name": {"type": ["string", "null"]},
                        },
                        "required": ["github_handle"],
                    },
                    "ComponentAttributes": {
                        "type": "object",
                        "properties": {},
                    },
                    "ParameterAttributes": {
                        "type": "object",
                        "properties": {
                            "value": {"type": ["string", "null"]},
                        },
                    },
                    "FirmwareVersionAttributes": {
                        "type": "object",
                        "properties": {
                            "version_string": {"type": "string"},
                            "vehicle_type": {"type": ["string", "null"]},
                        },
                        "required": ["version_string"],
                    },
                    "HardwarePlatformAttributes": {
                        "type": "object",
                        "properties": {},
                    },
                    "Entity": {
                        "type": "object",
                        "properties": {
                            "entity_type": {
                                "type": "string",
                                "enum": [
                                    "PERSON",
                                    "COMPONENT",
                                    "PARAMETER",
                                    "FIRMWARE_VERSION",
                                    "HARDWARE_PLATFORM",
                                ],
                            },
                            "surface_form": {
                                "type": "string",
                                "description": "Exact text as it appears in the document",
                            },
                            "attributes": {
                                "type": "object",
                                "description": (
                                    "Type-specific attributes. "
                                    "PERSON: {github_handle, display_name}. "
                                    "COMPONENT: {}. "
                                    "PARAMETER: {value}. "
                                    "FIRMWARE_VERSION: {version_string, vehicle_type}. "
                                    "HARDWARE_PLATFORM: {}."
                                ),
                            },
                            "local_id": {
                                "type": "string",
                                "description": "Sequential ID like 'e1', 'e2', 'e3' assigned in encounter order",
                            },
                            "context_sentence": {
                                "type": ["string", "null"],
                                "description": (
                                    "The exact sentence from the document body "
                                    "containing this entity mention. Copy the full "
                                    "sentence verbatim."
                                ),
                            },
                        },
                        "required": ["entity_type", "surface_form", "attributes", "local_id"],
                    },
                },
            },
        },
    }


def get_evidence_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "extract_evidence",
            "description": (
                "Extract selective, verbatim, concrete facts only. "
                "Copy exact text from the document body — no paraphrasing. "
                "Only extract concrete facts: bugs, versions, parameters, root causes. "
                "Do NOT extract speculation, greetings, or restatements of the issue title."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "evidence": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Evidence"},
                    }
                },
                "required": ["evidence"],
                "$defs": {
                    "Evidence": {
                        "type": "object",
                        "properties": {
                            "excerpt": {
                                "type": "string",
                                "description": (
                                    "Verbatim text copied exactly from the document body. "
                                    "Must be a contiguous substring of the body as written, "
                                    "including any markdown syntax."
                                ),
                            },
                            "significance": {
                                "type": "string",
                                "description": "One sentence explaining why this excerpt is notable.",
                            },
                        },
                        "required": ["excerpt", "significance"],
                    },
                },
            },
        },
    }


def get_claim_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "extract_claims",
            "description": (
                "Extract all factual claims from the document. "
                "Reference only entity local_ids listed in the pre-extracted entity list "
                "and evidence IDs from the pre-extracted evidence list provided in the user message."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Claim"},
                    }
                },
                "required": ["claims"],
                "$defs": {
                    "Claim": {
                        "type": "object",
                        "properties": {
                            "predicate": {
                                "type": "string",
                                "enum": [
                                    "CAUSES",
                                    "AFFECTS",
                                    "FIXED_IN",
                                    "REGRESSED_IN",
                                    "HAS_PARAMETER",
                                    "DUPLICATE_OF",
                                    "ALIAS_OF",
                                ],
                            },
                            "subject_local_id": {
                                "type": "string",
                                "description": (
                                    "local_id of the subject entity (always required). "
                                    "Must be a valid ID from the entity list (e1, e2, ...). "
                                    "If no valid subject exists, omit the claim. Never use placeholders."
                                ),
                            },
                            "object_local_id": {
                                "type": ["string", "null"],
                                "description": (
                                    "local_id of the object entity. "
                                    "Used for FIXED_IN (version entity), "
                                    "REGRESSED_IN (version entity), "
                                    "HAS_PARAMETER (parameter entity)."
                                ),
                            },
                            "object_freetext": {
                                "type": ["string", "null"],
                                "description": (
                                    "Freetext object description. "
                                    "Used for CAUSES (symptom description), "
                                    "DUPLICATE_OF (issue number as string)."
                                ),
                            },
                            "evidence_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "description": (
                                    "Valid evidence IDs from the pre-extracted evidence list (ev_...). "
                                    "Non-empty. Do not invent new IDs."
                                ),
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                            },
                        },
                        "required": [
                            "predicate",
                            "subject_local_id",
                            "evidence_ids",
                            "confidence",
                        ],
                    },
                },
            },
        },
    }


def get_tools() -> list[dict]:
    return [get_entity_tool(), get_evidence_tool(), get_claim_tool()]
