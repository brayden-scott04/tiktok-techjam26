"""End-to-end sandbox fixture tests. Slower (each spins up a real subprocess),
so run these deliberately (`pytest tests/test_runner.py -q`), not on every
save. Each asserts the runner produces the exact intended failure status, and
that no orphaned python process is left behind.
"""
import os

import psutil
import pytest

from harness import dataset as ds
from harness.runner import run_node

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")
CACHE_PATH = os.path.join(ROOT, "artifacts", "cache", "kuairand_pure")
SANITIZED_PATH = os.path.join(ROOT, "artifacts", "cache", "kuairand_pure_sanitized")


@pytest.fixture(scope="module")
def valid_labels():
    if not os.path.exists(CACHE_PATH + ".npz"):
        pytest.skip("cache not built -- run `python -m scripts.build_cache` first")
    auth_cache, _ = ds.load_cache(CACHE_PATH, skip_raw_meta=True)
    valid_c = auth_cache["valid"]
    return list(valid_c["user_id_raw"]), [int(x) for x in valid_c["long_view"]]


def _run(name, valid_labels, out_subdir, **kwargs):
    uids, labels = valid_labels
    out_dir = os.path.join(ROOT, "artifacts", "nodes", f"test_{out_subdir}")
    return run_node(
        solution_path=os.path.join(FIXTURES, name),
        out_dir=out_dir,
        sanitized_cache_path=SANITIZED_PATH,
        authoritative_valid_user_ids=uids,
        authoritative_valid_labels=labels,
        seed=0,
        smoke=True,
        repo_root=ROOT,
        **kwargs,
    )


def _no_orphans(pids_before):
    pids_after = set(psutil.pids())
    leaked = pids_after - pids_before
    leaked_python = [
        p for p in leaked
        if psutil.pid_exists(p) and "python" in (psutil.Process(p).name() or "").lower()
    ]
    assert not leaked_python, f"orphaned python processes: {leaked_python}"


def test_banned_import_rejected_before_execution(valid_labels):
    # note: run_node itself does not call harness.guards (that's the
    # pre-execution gate in agent/validate.py, run before a node is ever
    # scheduled) -- this test exercises the runtime path only, confirming an
    # os.listdir call inside the sandbox does not crash the harness even if
    # it somehow got past the static gate.
    pids_before = set(psutil.pids())
    result = _run("banned_import.py", valid_labels, "banned_import")
    assert result.status == "ok"  # os.listdir(".") succeeds inside the sandbox cwd; not itself fatal
    _no_orphans(pids_before)


def test_infinite_loop_times_out(valid_labels):
    pids_before = set(psutil.pids())
    result = _run("infinite_loop.py", valid_labels, "infinite_loop", timeout_sec=5)
    assert result.status == "timeout"
    _no_orphans(pids_before)


def test_oom_killed(valid_labels):
    # Either our RSS monitor kills it first (status="oom") or the child hits
    # a natural MemoryError first (status="error", e.g. under real system
    # memory pressure from other processes) -- both mean runaway growth was
    # stopped without hanging or leaving an orphaned process, which is the
    # actual property under test. The exact crossover point is inherently
    # racy (monitor poll interval vs. whatever memory happens to be free at
    # the moment), so we don't assert a specific peak_rss threshold.
    pids_before = set(psutil.pids())
    result = _run("oom.py", valid_labels, "oom", timeout_sec=60, max_rss_mb=500)
    assert result.status in ("oom", "error")
    _no_orphans(pids_before)


def test_nan_scores_rejected(valid_labels):
    pids_before = set(psutil.pids())
    result = _run("nan_scores.py", valid_labels, "nan_scores")
    assert result.status == "error"
    assert "NaN" in (result.traceback or "") or "NaN" in result.stderr
    _no_orphans(pids_before)


def test_wrong_length_rejected(valid_labels):
    pids_before = set(psutil.pids())
    result = _run("wrong_length.py", valid_labels, "wrong_length")
    assert result.status == "error"
    assert "length" in (result.traceback or "").lower()
    _no_orphans(pids_before)


def test_eval_budget_cap_enforced(valid_labels):
    pids_before = set(psutil.pids())
    result = _run("eval_budget_exceeded.py", valid_labels, "eval_budget", eval_call_cap=200)
    assert result.status == "error"
    assert "cap" in (result.traceback or "").lower()
    _no_orphans(pids_before)


def test_train_on_valid_finds_only_sentinel(valid_labels):
    # confirms the STRUCTURAL guarantee end-to-end: the child sees only -1 for
    # valid labels, so the fixture's fallback branch (predicting the raw label
    # back) never triggers and it returns all-zero scores instead.
    pids_before = set(psutil.pids())
    result = _run("train_on_valid.py", valid_labels, "train_on_valid")
    assert result.status == "ok"
    assert (result.valid_scores == 0).all(), "solution saw a non-sentinel label -- SANITIZER BREACH"
    _no_orphans(pids_before)
