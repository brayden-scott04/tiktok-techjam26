import numpy as np


def fit_predict(ctx):
    n_valid = len(ctx.splits["valid"])
    n_test = len(ctx.splits["test"])
    scores = np.zeros(n_valid, dtype=np.float32)
    for _ in range(300):  # exceeds the 200-call cap
        ctx.eval_valid(scores)
    return {"valid": scores, "test": np.zeros(n_test)}
