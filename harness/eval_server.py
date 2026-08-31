"""A tiny localhost TCP server holding the TRUE validation labels, so that
ctx.eval_valid() can score a solution's predictions without the sandboxed
child process ever holding a validation label itself.

Runs in a background thread inside the parent (runner) process — never inside
the sandboxed subprocess. The child only knows a port number and a per-node
token, both passed via environment variables that the AST-restricted solution
code cannot even see (they're read by harness/context.py, not by solution.py).

This module imports kit.evaluate and must therefore NEVER be imported by
harness/context.py, harness/node_entry.py, or anything reachable from the
sandboxed child -- that's why the client side lives in the separate
harness/eval_client.py, which has no such import. harness/runner.py (the
parent-process orchestrator) is the only importer of this module.
tests/test_isolation.py checks this by walking the actual import graph.

Wire protocol, over one connection per call:
  request:  4-byte big-endian length N, then N bytes of latin-1 token,
            then 4-byte big-endian length M, then M bytes of float32 LE array
            (the candidate's per-row valid scores, in cache row order)
  response: 4-byte big-endian length K, then K bytes of UTF-8 JSON:
            {"ok": true, "GAUC":.., "nDCG@5":.., "primary":..}
            or {"ok": false, "error": "..."}
"""
import json
import socket
import struct
import threading
import time

import numpy as np

from kit.evaluate import evaluate
from harness.eval_client import eval_valid_client  # noqa: F401  (re-exported for callers/tests that only import this module)


class EvalServer:
    def __init__(self, valid_user_ids, valid_labels, token, call_cap=200, host="127.0.0.1"):
        self.valid_user_ids = list(valid_user_ids)
        # Defensive: kit.evaluate.auc() computes npos*(npos+1), which overflows
        # a narrow numpy int dtype (e.g. int8) for any user with more than a
        # handful of positives under numpy 2.x's NEP 50 promotion rules. Force
        # plain Python ints regardless of what dtype the caller's cache used.
        self.valid_labels = [int(x) for x in valid_labels]
        self.token = token
        self.call_cap = call_cap
        self.host = host
        self.calls_made = 0
        self.cumulative_eval_seconds = 0.0
        self._sock = None
        self._thread = None
        self._stop = False
        self.port = None

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, 0))
        self._sock.listen(8)
        self._sock.settimeout(0.5)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve_loop, daemon=True)
        self._thread.start()
        return self.port

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)
        if self._sock:
            self._sock.close()

    def _serve_loop(self):
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._handle(conn)
            except Exception:
                pass
            finally:
                conn.close()

    def _recv_exact(self, conn, n):
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("peer closed early")
            buf += chunk
        return buf

    def _handle(self, conn):
        (tok_len,) = struct.unpack(">I", self._recv_exact(conn, 4))
        token = self._recv_exact(conn, tok_len).decode("latin-1")
        (arr_len,) = struct.unpack(">I", self._recv_exact(conn, 4))
        arr_bytes = self._recv_exact(conn, arr_len)

        if token != self.token:
            resp = {"ok": False, "error": "bad_token"}
        elif self.calls_made >= self.call_cap:
            resp = {"ok": False, "error": f"eval_valid call cap ({self.call_cap}) exceeded"}
        else:
            scores = np.frombuffer(arr_bytes, dtype=np.float32)
            if len(scores) != len(self.valid_labels):
                resp = {
                    "ok": False,
                    "error": f"expected {len(self.valid_labels)} scores, got {len(scores)}",
                }
            else:
                t0 = time.time()
                metrics = evaluate(self.valid_user_ids, self.valid_labels, scores)
                self.cumulative_eval_seconds += time.time() - t0
                self.calls_made += 1
                resp = {
                    "ok": True,
                    "GAUC": metrics["GAUC"],
                    "nDCG@5": metrics["nDCG@5"],
                    "primary": metrics["primary"],
                    "calls_remaining": self.call_cap - self.calls_made,
                }

        payload = json.dumps(resp).encode("utf-8")
        conn.sendall(struct.pack(">I", len(payload)) + payload)
