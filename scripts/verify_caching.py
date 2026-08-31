"""Confirms OpenAI prompt caching actually triggers on our static prefix
(system.md + task_brief.md, ~2.5k+ tokens, sent first and byte-identical on
every call). Makes two back-to-back calls with the real draft-message
structure but on the CHEAP tier (luna), since the caching mechanism is
prefix-identity-based, not model-specific -- if it caches here, it caches the
same way for the brain (sol) calls in the real run. Not a pytest test:
makes real (tiny) API calls.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.env_loader import load_env

load_env()

from agent import memory
from agent.llm import LLMClient
from agent.schema import RESPONSE_SCHEMA


def main():
    client = LLMClient()
    journal_md = "(no prior iterations yet -- this is the first one)"
    diag_md = "(no scored node yet)"
    budget_md = "Nodes used: 0/50. Scored iterations: 0 (floor: 12). Wall-clock used: 0s / 21600s. Spend so far: $0.00 / $40.00."

    messages = memory.build_draft_messages(journal_md, diag_md, budget_md, used_directions=[])
    prefix_tokens_estimate = len(messages[0]["content"]) // 4
    print(f"static prefix length: {len(messages[0]['content'])} chars (~{prefix_tokens_estimate} tokens estimate)")

    print("\n=== call 1 (cold, no cache expected) ===")
    _, usage1 = client.call("repair", messages, RESPONSE_SCHEMA, "solution_response")
    print(json.dumps({k: v for k, v in usage1.items() if k != "raw_usage"}, indent=2))

    print("\n=== call 2 (same static prefix, cache should hit) ===")
    _, usage2 = client.call("repair", messages, RESPONSE_SCHEMA, "solution_response")
    print(json.dumps({k: v for k, v in usage2.items() if k != "raw_usage"}, indent=2))

    print(f"\ncached_tokens: call1={usage1['cached_tokens']}  call2={usage2['cached_tokens']}")
    if usage2["cached_tokens"] > 0:
        print("CACHING IS WORKING: call 2 hit the cache.")
    else:
        print("NO CACHE HIT on call 2 -- either the prefix is below the cache size "
              "threshold, the cache TTL is shorter than the gap between these calls, "
              "or caching needs a larger prefix than what we're sending pre-journal-growth.")


if __name__ == "__main__":
    main()
