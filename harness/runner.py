"""Subprocess sandbox: launches one node's solution.py inside a locked-down
child process, enforces a wall-clock timeout and an RSS memory cap, tree-kills
on either, and captures stdout/stderr. This is the ONLY place a solution's
code actually executes.
"""
import os
import secrets
import subprocess
import sys
import threading
import time

import psutil

from harness.eval_server import EvalServer

STDOUT_CAP_BYTES = 10 * 1024 * 1024


class NodeResult:
    def __init__(self):
        self.status = None  # ok | error | timeout | oom | guard_reject | dedup_reject | syntax_error
        self.seconds = None
        self.peak_rss_mb = None
        self.stdout = ""
        self.stderr = ""
        self.traceback = None
        self.valid_scores = None
        self.test_scores = None
        self.eval_calls = None
        self.eval_seconds = None
        self.exit_code = None


def _tail(text, n=4000):
    return text[-n:] if len(text) > n else text


def _rss_monitor(pid, max_rss_mb, stop_event, peak_holder):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    while not stop_event.is_set():
        try:
            rss_mb = proc.memory_info().rss / (1024 * 1024)
            for child in proc.children(recursive=True):
                try:
                    rss_mb += child.memory_info().rss / (1024 * 1024)
                except psutil.NoSuchProcess:
                    pass
            peak_holder["peak_mb"] = max(peak_holder.get("peak_mb", 0), rss_mb)
            if rss_mb > max_rss_mb:
                peak_holder["oom"] = True
                _kill_tree(pid)
                return
        except psutil.NoSuchProcess:
            return
        stop_event.wait(0.3)


def _kill_tree(pid):
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = proc.children(recursive=True)
    for c in children:
        try:
            c.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        proc.kill()
    except psutil.NoSuchProcess:
        pass
    psutil.wait_procs(children + [proc], timeout=5)


def run_node(
    solution_path,
    out_dir,
    sanitized_cache_path,
    authoritative_valid_user_ids,
    authoritative_valid_labels,
    seed=0,
    smoke=False,
    timeout_sec=900,
    max_rss_mb=12000,
    eval_call_cap=200,
    repo_root=None,
):
    """Runs one solution.py inside the sandbox. Returns a NodeResult.
    `authoritative_valid_*` come from the AUTHORITATIVE cache and are held
    only in this parent process, inside the EvalServer -- never passed to the
    child's environment or filesystem.
    """
    repo_root = repo_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    result = NodeResult()
    os.makedirs(out_dir, exist_ok=True)

    token = secrets.token_hex(16)
    server = EvalServer(authoritative_valid_user_ids, authoritative_valid_labels, token, call_cap=eval_call_cap)
    port = server.start()

    env = {
        "PYTHONUTF8": "1",
        "PYTHONHASHSEED": "0",
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "KR_SANITIZED_CACHE": sanitized_cache_path,
        "KR_SOLUTION_PATH": solution_path,
        "KR_OUT_DIR": out_dir,
        "KR_SEED": str(seed),
        "KR_SMOKE": "1" if smoke else "0",
        "KR_MAX_SECONDS": str(timeout_sec),
        "KR_EVAL_HOST": "127.0.0.1",
        "KR_EVAL_PORT": str(port),
        "KR_EVAL_TOKEN": token,
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),  # needed by Windows Python itself
        "PATH": os.environ.get("PATH", ""),  # needed to find the python interpreter's DLLs on Windows
    }
    # Deliberately absent from `env`: OPENAI_API_KEY, KR_DATA_ROOT -- the child
    # cannot reach the OpenAI API or the raw CSVs even if its code tried to.

    # Not -I (isolated mode): on Python 3.11+, -I implies safe-path behavior
    # that excludes the cwd from sys.path even for `-m`, which breaks
    # `-m harness.node_entry` resolving relative to cwd=repo_root. Isolation
    # here comes from the explicit env allowlist below (no PYTHONPATH is set,
    # so there is nothing for -I to additionally protect against), not from
    # interpreter flags.
    cmd = [sys.executable, "-X", "utf8", "-m", "harness.node_entry"]

    t0 = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stop_event = threading.Event()
    peak_holder = {}
    monitor = threading.Thread(target=_rss_monitor, args=(proc.pid, max_rss_mb, stop_event, peak_holder), daemon=True)
    monitor.start()

    timed_out = False
    try:
        stdout, stderr = proc.communicate(timeout=timeout_sec + 60)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc.pid)
        stdout, stderr = proc.communicate()
    finally:
        stop_event.set()
        monitor.join(timeout=3)
        server.stop()

    result.seconds = time.time() - t0
    result.stdout = _tail(stdout or "", STDOUT_CAP_BYTES)
    result.stderr = _tail(stderr or "", STDOUT_CAP_BYTES)
    result.peak_rss_mb = peak_holder.get("peak_mb")
    result.exit_code = proc.returncode

    if timed_out:
        result.status = "timeout"
        return result
    if peak_holder.get("oom"):
        result.status = "oom"
        return result

    valid_path = os.path.join(out_dir, "valid_scores.npy")
    test_path = os.path.join(out_dir, "test_scores.npy")
    result_json_path = os.path.join(out_dir, "node_result.json")

    import json

    node_info = {}
    if os.path.exists(result_json_path):
        with open(result_json_path, encoding="utf-8") as fh:
            node_info = json.load(fh)

    if proc.returncode != 0 or node_info.get("status") == "error":
        result.status = "error"
        result.traceback = node_info.get("error") or _tail(result.stderr, 4000)
        return result

    if not (os.path.exists(valid_path) and os.path.exists(test_path)):
        result.status = "error"
        result.traceback = "child exited 0 but did not write valid_scores.npy/test_scores.npy"
        return result

    import numpy as np

    result.valid_scores = np.load(valid_path)
    result.test_scores = np.load(test_path)
    result.eval_calls = node_info.get("eval_calls")
    result.eval_seconds = node_info.get("eval_seconds")
    result.status = "ok"
    return result
