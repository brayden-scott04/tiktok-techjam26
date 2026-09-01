"""One-time backfill: adds a `diff_vs_parent` field to every existing
artifacts/run_log.jsonl record from before agent/loop.py computed this live.

This does not fabricate anything new -- it computes a standard unified diff
between the already-committed solutions/{parent}/solution.py and
solutions/{node}/solution.py files, using each record's own recorded
parent_node_id. Every other field in every record is left byte-for-byte
untouched. Idempotent: records that already have a non-null diff_vs_parent
are skipped.
"""
import difflib
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_LOG_PATH = os.path.join(ROOT, "artifacts", "run_log.jsonl")


def _solution_path(node_id):
    return os.path.join(ROOT, "solutions", node_id, "solution.py")


def _diff(parent_id, node_id):
    parent_path, child_path = _solution_path(parent_id), _solution_path(node_id)
    if not (os.path.exists(parent_path) and os.path.exists(child_path)):
        return None
    with open(parent_path, encoding="utf-8") as fh:
        parent_lines = fh.readlines()
    with open(child_path, encoding="utf-8") as fh:
        child_lines = fh.readlines()
    diff = difflib.unified_diff(
        parent_lines, child_lines,
        fromfile=f"{parent_id}/solution.py", tofile="solution.py",
    )
    return "".join(diff)


def main():
    with open(RUN_LOG_PATH, encoding="utf-8") as fh:
        lines = [line for line in fh.read().splitlines() if line.strip()]

    updated = 0
    out_lines = []
    for line in lines:
        record = json.loads(line)
        if "node_id" in record and record.get("diff_vs_parent") is None:
            parent_id = record.get("parent_node_id")
            node_id = record.get("node_id")
            if parent_id and node_id:
                record["diff_vs_parent"] = _diff(parent_id, node_id)
                if record["diff_vs_parent"] is not None:
                    updated += 1
            elif "diff_vs_parent" not in record:
                record["diff_vs_parent"] = None  # draft with no parent -- legitimately no diff
        out_lines.append(json.dumps(record))

    with open(RUN_LOG_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out_lines) + "\n")

    print(f"backfilled diff_vs_parent for {updated} records in {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
