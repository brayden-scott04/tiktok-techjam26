"""Cheap real dry run of the full agent loop: a handful of iterations with
ctx.smoke=True (subsampled data, few epochs) so both the LLM plumbing and the
sandbox plumbing get exercised end to end for well under $1 before spending a
full multi-hour, multi-dollar competition run. Not a pytest test.
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.env_loader import load_env

load_env()

from agent.loop import AgentLoop

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(n_iterations=3):
    with open(os.path.join(ROOT, "config", "run.json"), encoding="utf-8") as fh:
        run_config = json.load(fh)

    # Isolate this dry run's state/log/solutions from any real run.
    for p in ("artifacts/state.json", "artifacts/run_log.jsonl"):
        full = os.path.join(ROOT, p)
        if os.path.exists(full):
            os.rename(full, full + ".pre_smoke_backup")

    try:
        loop = AgentLoop(ROOT, run_config, smoke=True, max_iterations_override=n_iterations)
        trigger = loop.run()
        print(f"\nstopped: trigger={trigger}")
        print(f"total_iterations={loop.state['total_iterations']}  scored_iterations={loop.state['scored_iterations']}")
        print(f"best_node_id={loop.state['best_node_id']}  best_primary={loop.state['best_primary']:.4f}")
        print(f"cost_usd=${loop.state['cost_usd']:.4f}")

        print("\n=== run_log.jsonl ===")
        with open(loop.run_log_path, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                print(json.dumps(rec, indent=2)[:1500])
                print("---")
    finally:
        for p in ("artifacts/state.json", "artifacts/run_log.jsonl"):
            full = os.path.join(ROOT, p)
            backup = full + ".pre_smoke_backup"
            if os.path.exists(full):
                os.remove(full)
            if os.path.exists(backup):
                os.rename(backup, full)


if __name__ == "__main__":
    main()
