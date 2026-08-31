"""Phase 2 exit criterion 1: run solutions/n0000 (the ported FM baseline)
through harness/runner.py exactly as any agent-generated node would run, and
confirm its validation primary matches the official baseline.

Note on tolerance: kit.data.encode() builds its 5th field's vocab (the
duration bucket) via first-appearance order over the training rows, same as
every other categorical field -- so "bucket 3" does not necessarily get local
embedding index 3 in the ORIGINAL kit implementation; it gets whatever index
it happened to reach first. Our ctx-based reimplementation assigns
dur_bucket's local index directly from the bucket number (a fixed, sane
mapping), which is a physically different but functionally equivalent
parametrization: same vocab size, same information, different embedding-row
labeling. That means the two implementations cannot be expected to produce
bit-identical floats (a different labeling changes which specific row of the
embedding table gets which random init and which gradient updates), even
though they are the same model. We therefore compare against the organizer's
PUBLISHED baseline number (0.6016, within seed-level noise), not against a
fresh kit.baseline.run_fm call's exact bits.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness import dataset as ds
from harness import task_spec as spec
from harness.runner import run_node


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cache_path = os.path.join(root, "artifacts", "cache", "kuairand_pure")
    sanitized_path = os.path.join(root, "artifacts", "cache", "kuairand_pure_sanitized")

    print("Loading authoritative cache (for eval_server's true valid labels only) ...")
    auth_cache, _ = ds.load_cache(cache_path, skip_raw_meta=True)
    valid_c = auth_cache["valid"]
    valid_user_ids = list(valid_c["user_id_raw"])
    valid_labels = [int(x) for x in valid_c["long_view"]]

    solution_path = os.path.join(root, "solutions", "n0000", "solution.py")
    out_dir = os.path.join(root, "artifacts", "nodes", "n0000_verify")

    print("Running node 0 through harness/runner.py (seed=0, full run, ~40s expected) ...")
    t0 = time.time()
    result = run_node(
        solution_path=solution_path,
        out_dir=out_dir,
        sanitized_cache_path=sanitized_path,
        authoritative_valid_user_ids=valid_user_ids,
        authoritative_valid_labels=valid_labels,
        seed=0,
        smoke=False,
        timeout_sec=300,
        repo_root=root,
    )
    wall = time.time() - t0
    print(f"status={result.status}  wall={wall:.1f}s  exit_code={result.exit_code}  peak_rss_mb={result.peak_rss_mb}")

    if result.status != "ok":
        print("STDOUT TAIL:\n" + result.stdout[-2000:])
        print("STDERR TAIL:\n" + result.stderr[-2000:])
        print("TRACEBACK:\n", result.traceback)
        raise SystemExit(f"node 0 did not complete OK (status={result.status})")

    print("STDOUT TAIL:\n" + result.stdout[-1500:])

    from kit.evaluate import evaluate as kit_evaluate

    valid_metrics = kit_evaluate(valid_user_ids, valid_labels, result.valid_scores)
    print(f"node0 valid: GAUC {valid_metrics['GAUC']:.4f} nDCG@5 {valid_metrics['nDCG@5']:.4f} primary {valid_metrics['primary']:.4f}")
    print(f"baseline valid primary: {spec.BASELINE_VALID_PRIMARY:.4f}")

    tol = 0.003  # ~4x the baseline's own 5-seed std (0.0008); see module docstring
    diff = abs(valid_metrics["primary"] - spec.BASELINE_VALID_PRIMARY)
    ok = diff <= tol
    print(f"diff={diff:.4f}  tolerance={tol}  -> {'PASS' if ok else 'FAIL'}")

    import json

    out_path = os.path.join(root, "artifacts", "phase2_verification.json")
    existing = {}
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing["verify_node0"] = {
        "status": result.status,
        "wall_seconds": wall,
        "peak_rss_mb": result.peak_rss_mb,
        "eval_calls": result.eval_calls,
        "valid_metrics": valid_metrics,
        "baseline_valid_primary": spec.BASELINE_VALID_PRIMARY,
        "diff": diff,
        "tolerance": tol,
        "ok": ok,
        "note": "Not bit-identical to kit.baseline.run_fm by design -- see module docstring. Compared against the organizer-published baseline number instead.",
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)

    if not ok:
        raise SystemExit("node 0 fidelity check FAILED")
    print("Phase 2 node0 fidelity: PASS")


if __name__ == "__main__":
    main()
