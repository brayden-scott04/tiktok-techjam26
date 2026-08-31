"""Client half of the eval_valid protocol -- the only half that ever runs
inside the sandboxed child process (via harness/context.py's ctx.eval_valid
closure). Split out from harness/eval_server.py specifically so this module
has no import path to kit.evaluate: the child can import this file's one
function without ever importing the scoring code itself, which is what makes
"node_entry.py never imports kit.evaluate" a fact about the import graph
(tests/test_isolation.py checks this), not just about what the code happens
to call at runtime. See harness/eval_server.py for the wire protocol this
implements the client side of.
"""
import json
import socket
import struct

import numpy as np


def eval_valid_client(host, port, token, scores):
    arr = np.asarray(scores, dtype=np.float32)
    tok_bytes = token.encode("latin-1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(30)
        s.connect((host, port))
        s.sendall(struct.pack(">I", len(tok_bytes)) + tok_bytes)
        s.sendall(struct.pack(">I", len(arr.tobytes())) + arr.tobytes())
        (resp_len,) = struct.unpack(">I", s.recv(4))
        buf = b""
        while len(buf) < resp_len:
            chunk = s.recv(resp_len - len(buf))
            if not chunk:
                raise ConnectionError("eval server closed early")
            buf += chunk
    resp = json.loads(buf.decode("utf-8"))
    if not resp.get("ok"):
        raise RuntimeError(f"eval_valid failed: {resp.get('error')}")
    return {"GAUC": resp["GAUC"], "nDCG@5": resp["nDCG@5"], "primary": resp["primary"]}
