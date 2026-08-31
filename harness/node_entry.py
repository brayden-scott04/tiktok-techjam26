"""Entry point executed INSIDE the sandboxed child process (harness/runner.py
launches `python -X utf8 -I -m harness.node_entry` with a minimal, secret-free
environment). Loads the SANITIZED cache only, builds ctx, dynamically imports
the node's solution.py, calls fit_predict(ctx), validates the output, and
writes the two prediction arrays to disk. Never imports kit.evaluate and never
has a path to the authoritative (labeled) cache -- both are structural, not
just policy (see harness/context.py's module docstring).
"""
import importlib.util
import json
import os
import sys
import traceback

import numpy as np

from harness import dataset as ds
from harness.context import build_ctx, assert_sanitized


def _env(name, required=True, default=None):
    v = os.environ.get(name, default)
    if required and v is None:
        raise RuntimeError(f"missing required env var {name}")
    return v


def _load_solution(path):
    spec = importlib.util.spec_from_file_location("candidate_solution", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "fit_predict"):
        raise RuntimeError("solution.py does not define fit_predict(ctx)")
    return mod


def main():
    sanitized_path = _env("KR_SANITIZED_CACHE")
    solution_path = _env("KR_SOLUTION_PATH")
    out_dir = _env("KR_OUT_DIR")
    seed = int(_env("KR_SEED", default="0"))
    smoke = _env("KR_SMOKE", default="0") == "1"
    max_seconds = float(_env("KR_MAX_SECONDS", default="900"))
    eval_host = _env("KR_EVAL_HOST")
    eval_port = int(_env("KR_EVAL_PORT"))
    eval_token = _env("KR_EVAL_TOKEN")

    os.makedirs(out_dir, exist_ok=True)

    def log(msg):
        print(f"[node] {msg}", flush=True)

    log(f"loading sanitized cache from {sanitized_path}")
    cache, meta = ds.load_cache(sanitized_path, skip_raw_meta=True)

    log("running sanitizer self-check (must find only -1 in non-train outcome columns)")
    assert_sanitized(cache)

    ctx = build_ctx(
        sanitized_cache=cache, meta=meta, seed=seed, smoke=smoke, max_seconds=max_seconds,
        eval_host=eval_host, eval_port=eval_port, eval_token=eval_token, log_fn=log,
    )

    log(f"loading solution from {solution_path}")
    mod = _load_solution(solution_path)

    log("calling fit_predict(ctx)")
    result = mod.fit_predict(ctx)

    if not isinstance(result, dict) or "valid" not in result or "test" not in result:
        raise RuntimeError("fit_predict must return {'valid': array, 'test': array}")

    valid_scores = np.asarray(result["valid"], dtype=np.float64)
    test_scores = np.asarray(result["test"], dtype=np.float64)

    n_valid = len(ctx.splits["valid"])
    n_test = len(ctx.splits["test"])
    if len(valid_scores) != n_valid:
        raise RuntimeError(f"valid scores length {len(valid_scores)} != expected {n_valid}")
    if len(test_scores) != n_test:
        raise RuntimeError(f"test scores length {len(test_scores)} != expected {n_test}")
    if not np.all(np.isfinite(valid_scores)):
        raise RuntimeError("valid scores contain NaN/Inf")
    if not np.all(np.isfinite(test_scores)):
        raise RuntimeError("test scores contain NaN/Inf")

    np.save(os.path.join(out_dir, "valid_scores.npy"), valid_scores)
    np.save(os.path.join(out_dir, "test_scores.npy"), test_scores)

    with open(os.path.join(out_dir, "node_result.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "status": "ok",
                "eval_calls": ctx._eval_calls,
                "eval_seconds": ctx._eval_time,
                "elapsed_seconds": ctx.elapsed_seconds(),
            },
            fh,
        )
    log("done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        out_dir = os.environ.get("KR_OUT_DIR")
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "node_result.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {"status": "error", "error": str(e), "traceback": traceback.format_exc()},
                    fh,
                )
        print(f"[node] ERROR: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)
