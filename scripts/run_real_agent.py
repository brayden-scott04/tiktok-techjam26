"""Phase 3: the real scored competition run. Up to 50 iterations, 6h
wall-clock, $40 cost circuit breaker (all from config/run.json), unattended.

Verifies kit/ and harness/ integrity before starting (aborts on tamper), and
resumes from artifacts/state.json if one already exists rather than
restarting -- so an interrupted run can be continued with
`python -X utf8 -m scripts.run_real_agent` again.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.env_loader import load_env

load_env()

from agent.loop import AgentLoop
from harness import integrity
from harness.report import render_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    print("Verifying kit/ and harness/ integrity before starting...")
    integrity.verify_all()
    print("  OK -- no tampering detected.\n")

    with open(os.path.join(ROOT, "config", "run.json"), encoding="utf-8") as fh:
        run_config = json.load(fh)

    resuming = os.path.exists(os.path.join(ROOT, "artifacts", "state.json"))
    print(f"{'Resuming existing' if resuming else 'Starting new'} run.")
    print(f"Caps: {run_config['caps']['max_total_iterations']} iterations, "
          f"{run_config['caps']['max_wall_seconds']/3600:.1f}h wall-clock, "
          f"${run_config['caps']['max_cost_usd']} budget.")
    print(f"Convergence: epsilon={run_config['convergence']['epsilon']}, "
          f"N={run_config['convergence']['N']}, "
          f"min_scored_iterations={run_config['convergence']['min_scored_iterations']}.\n")

    loop = AgentLoop(ROOT, run_config, smoke=False)
    t0 = time.time()
    trigger = loop.run()
    wall = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RUN ENDED: trigger={trigger}")
    print(f"total_iterations={loop.state['total_iterations']}  scored_iterations={loop.state['scored_iterations']}")
    print(f"best_node_id={loop.state['best_node_id']}  best_valid_primary={loop.state['best_primary']:.4f}")
    print(f"cost_usd=${loop.state['cost_usd']:.2f}")
    print(f"wall_seconds={wall:.0f}s ({wall/3600:.2f}h)")
    print(f"{'='*60}\n")

    report_path = os.path.join(ROOT, "artifacts", "report.md")
    table_path = os.path.join(ROOT, "artifacts", "results_table.md")
    best_id = loop.state["best_node_id"]
    best_metrics = loop.state["nodes"][best_id]["metrics"] if best_id else None
    render_report(loop.run_log_path, report_path, table_path, best_node_id=best_id, best_valid_metrics=best_metrics)
    print(f"Wrote {report_path} and {table_path} (test columns pending scripts.final_test_score)")


if __name__ == "__main__":
    main()
