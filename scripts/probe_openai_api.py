"""One-time, cheap probe: call gpt-5.6-sol and gpt-5.6-luna with a trivial
structured-output request via the Responses API, and print the ACTUAL shape
of the response and usage object -- grounding agent/llm.py in real API
behavior instead of documentation excerpts or pre-cutoff assumptions.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.env_loader import load_env

load_env()

from openai import OpenAI

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer"],
    "properties": {"answer": {"type": "string"}},
}


def probe(client, model, with_reasoning):
    kwargs = dict(
        model=model,
        input=[{"role": "user", "content": "Reply with the single word: pong"}],
        text={"format": {"type": "json_schema", "name": "reply", "schema": SCHEMA, "strict": True}},
    )
    if with_reasoning:
        kwargs["reasoning"] = {"effort": "low"}
    print(f"\n=== {model}  reasoning={with_reasoning} ===")
    try:
        resp = client.responses.create(**kwargs)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")
        return
    print("output_text:", getattr(resp, "output_text", None))
    usage = getattr(resp, "usage", None)
    if usage is not None:
        try:
            print("usage (model_dump):", json.dumps(usage.model_dump(), indent=2))
        except Exception:
            print("usage (raw):", usage)
    else:
        print("usage: None")


def main():
    client = OpenAI()
    probe(client, "gpt-5.6-luna", with_reasoning=False)
    probe(client, "gpt-5.6-luna", with_reasoning=True)
    probe(client, "gpt-5.6-sol", with_reasoning=True)


if __name__ == "__main__":
    main()
