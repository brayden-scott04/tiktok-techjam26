"""JSON Schema for the LLM response envelope, used with OpenAI structured
outputs so a malformed response is rejected by the API itself rather than
crashing our parser. See agent/llm.py.
"""

DIRECTION_ENUM = [
    "loss", "sequence", "multitask", "watchtime", "arch",
    "time_shift", "calibration", "ensemble", "other",
]

RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hypothesis", "rationale", "direction", "direction_note", "references",
        "changes", "expected_effect", "risks", "est_runtime_sec", "code",
    ],
    "properties": {
        "hypothesis": {
            "type": "string",
            "description": "<=3 sentences: what change, and the causal claim for why it should help.",
        },
        "rationale": {
            "type": "string",
            "description": "The reasoning, including which diagnostics evidence motivated it.",
        },
        "direction": {"type": "string", "enum": DIRECTION_ENUM},
        "direction_note": {
            "type": "string",
            "description": "Free text if direction == 'other'; empty string otherwise.",
        },
        "references": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Published methods being drawn on, e.g. 'BPR (Rendle 2009)'.",
        },
        "changes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet list of concrete edits vs. the parent solution.",
        },
        "expected_effect": {
            "type": "object",
            "additionalProperties": False,
            "required": ["metric", "delta", "confidence"],
            "properties": {
                "metric": {"type": "string", "enum": ["primary", "GAUC", "nDCG@5"]},
                "delta": {"type": "number"},
                "confidence": {"type": "number"},
            },
        },
        "risks": {
            "type": "string",
            "description": "What could make this fail or blow the time budget.",
        },
        "est_runtime_sec": {
            "type": "number",
            "description": "Your own estimate of full-run wall-clock seconds.",
        },
        "code": {
            "type": "string",
            "description": "The complete, standalone solution.py source implementing fit_predict(ctx).",
        },
    },
}

REPAIR_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["diagnosis", "code"],
    "properties": {
        "diagnosis": {"type": "string", "description": "One sentence: what broke and why."},
        "code": {"type": "string", "description": "The complete corrected solution.py source."},
    },
}
