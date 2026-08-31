"""Cheap real-call smoke test of agent/llm.py against the actual
RESPONSE_SCHEMA envelope, using the repair (cheap) tier. Not a pytest test --
this spends real (tiny) money, so it's run deliberately, not on every
`pytest` invocation.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.env_loader import load_env

load_env()

from agent.llm import LLMClient
from agent.schema import RESPONSE_SCHEMA


def main():
    client = LLMClient()
    messages = [
        {
            "role": "user",
            "content": (
                "You are drafting one iteration for a numpy-only factorization "
                "machine recommender. Propose a trivial, safe first change: "
                "increase the embedding dimension k from 16 to 24. Fill every "
                "field of the schema. For `code`, just write a minimal valid "
                "Python file with a single function `def fit_predict(ctx): "
                "return {'valid': ctx.splits['valid'].date * 0.0, 'test': "
                "ctx.splits['test'].date * 0.0}` (a placeholder; do not "
                "actually implement the FM, this is only a plumbing test)."
            ),
        }
    ]
    parsed, usage = client.call("repair", messages, RESPONSE_SCHEMA, "solution_response")
    print("=== parsed keys ===")
    print(list(parsed.keys()))
    print("\n=== hypothesis ===")
    print(parsed["hypothesis"])
    print("\n=== direction ===")
    print(parsed["direction"])
    print("\n=== code (first 300 chars) ===")
    print(parsed["code"][:300])
    print("\n=== usage record ===")
    print(json.dumps({k: v for k, v in usage.items() if k != "raw_usage"}, indent=2))


if __name__ == "__main__":
    main()
