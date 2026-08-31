"""One-time account check: confirms OPENAI_API_KEY works and lists available
models, so config/models.json can be filled with real, verified IDs instead
of assumed ones. Never prints the key itself.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.env_loader import load_env

load_env()

from openai import OpenAI


def main():
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise SystemExit("OPENAI_API_KEY not set (check .env)")
    print(f"OPENAI_API_KEY present, length={len(key)}, prefix={key[:8]}... (not printing the rest)")

    client = OpenAI(api_key=key)
    models = list(client.models.list())
    ids = sorted(m.id for m in models)
    print(f"\n{len(ids)} models visible to this account.\n")

    interesting_prefixes = ("gpt-5", "gpt-4.1", "gpt-4o", "o3", "o4", "gpt-4-turbo")
    interesting = [i for i in ids if i.startswith(interesting_prefixes)]
    print("Relevant chat/reasoning models:")
    for i in interesting:
        print(f"  {i}")

    print("\nAll model IDs (for reference):")
    for i in ids:
        print(f"  {i}")


if __name__ == "__main__":
    main()
