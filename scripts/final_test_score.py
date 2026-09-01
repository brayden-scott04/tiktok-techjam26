"""Phase 4: sealed, one-shot hidden-test scoring. Run by a human, once, after
the agent run has converged. Reads the validation-best node's already-computed
test_scores.npy (produced during the run's normal seed-0 full run -- the loop
never looks at it), writes the submission CSV via the unmodified kit.submit
writer, independently re-validates with the unmodified kit.submit reader, and
calls harness.sealed.score_test exactly once.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.env_loader import load_env

load_env()

from harness import dataset as ds
from harness import task_spec as spec
from harness import sealed
from harness.report import render_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-score even if already scored once (logged, deliberate use only)")
    a = ap.parse_args()

    state_path = os.path.join(ROOT, "artifacts", "state.json")
    with open(state_path, encoding="utf-8") as fh:
        state = json.load(fh)

    best_node_id = state["best_node_id"]
    best_node = state["nodes"][best_node_id]
    print(f"Validation-best node: {best_node_id}  (valid primary {best_node['metrics']['primary']:.4f})")
    print(f"Selected using validation metrics only -- see artifacts/run_log.jsonl for the full history "
          f"(it contains no test metrics anywhere, by construction).")

    test_scores_path = best_node["test_scores_path"]
    test_scores = np.load(test_scores_path)

    cache_path = os.path.join(ROOT, "artifacts", "cache", "kuairand_pure")
    auth_cache, _ = ds.load_cache(cache_path, skip_raw_meta=True)
    test_c = auth_cache["test"]
    test_user_ids = list(test_c["user_id_raw"])
    test_video_ids = list(test_c["video_id_raw"])
    test_labels = [int(x) for x in test_c["long_view"]]

    rows_for_submission = list(zip(test_c["date"], test_user_ids, test_video_ids))

    submission_path = os.path.join(ROOT, "artifacts", "submission_test.csv")
    aligned_scores = sealed.write_and_verify_submission(submission_path, rows_for_submission, test_scores)
    print(f"Submission written and independently re-validated: {submission_path}")

    metrics = sealed.score_test(test_user_ids, test_labels, test_scores, force=a.force)
    print(f"\n=== SEALED TEST RESULT (computed once) ===")
    print(f"GAUC {metrics['GAUC']:.4f} | nDCG@5 {metrics['nDCG@5']:.4f} | primary {metrics['primary']:.4f}")
    print(f"Baseline:  GAUC {spec.BASELINE_TEST['GAUC']:.4f} | nDCG@5 {spec.BASELINE_TEST['nDCG@5']:.4f} | primary {spec.BASELINE_TEST_PRIMARY:.4f}")
    delta = metrics["primary"] - spec.BASELINE_TEST_PRIMARY
    print(f"Delta vs baseline (primary): {delta:+.4f}")

    final_result = {
        "best_node_id": best_node_id,
        "valid_metrics": best_node["metrics"],
        "test_metrics": metrics,
        "baseline_test": spec.BASELINE_TEST,
        "delta_vs_baseline_test_primary": delta,
    }
    with open(os.path.join(ROOT, "artifacts", "final_result.json"), "w", encoding="utf-8") as fh:
        json.dump(final_result, fh, indent=2)

    run_log_path = os.path.join(ROOT, "artifacts", "run_log.jsonl")
    report_path = os.path.join(ROOT, "artifacts", "report.md")
    table_path = os.path.join(ROOT, "artifacts", "results_table.md")
    render_report(run_log_path, report_path, table_path, test_metrics=metrics,
                  best_node_id=best_node_id, best_valid_metrics=best_node["metrics"])
    print(f"\nWrote {report_path} and {table_path}")


if __name__ == "__main__":
    main()
