"""The ONLY way a manual intervention during a scored run gets recorded.
Routing every human touch through this script (rather than a free-text note
anywhere) is what makes the autonomy claim in the README auditable: the
intervention count is `wc -l artifacts/interventions.jsonl`, not a number
someone typed into a report.
"""
import argparse
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "artifacts", "interventions.jsonl")

CATEGORIES = ("restart_manual", "code_edit", "config_change", "data_fix", "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reason", required=True)
    ap.add_argument("--category", required=True, choices=CATEGORIES)
    a = ap.parse_args()

    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    with open(PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"timestamp": time.time(), "category": a.category, "reason": a.reason}) + "\n")

    with open(PATH, encoding="utf-8") as fh:
        n = sum(1 for _ in fh)
    print(f"Logged intervention #{n}: [{a.category}] {a.reason}")


if __name__ == "__main__":
    main()
