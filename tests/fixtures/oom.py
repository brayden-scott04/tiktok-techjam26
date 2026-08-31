import time

import numpy as np


def fit_predict(ctx):
    # Small increments with a short sleep between them, so the harness's RSS
    # monitor gets several chances to intervene before a natural MemoryError
    # -- a large single allocation can blow past both the cap and the OS's
    # real limit within one polling interval, racing our own guard.
    hogs = []
    while True:
        hogs.append(np.zeros((5_000_000,), dtype=np.float64))  # ~40MB per chunk
        time.sleep(0.05)
    n_valid = len(ctx.splits["valid"])
    n_test = len(ctx.splits["test"])
    return {"valid": np.zeros(n_valid), "test": np.zeros(n_test)}
