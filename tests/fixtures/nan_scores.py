import numpy as np


def fit_predict(ctx):
    n_valid = len(ctx.splits["valid"])
    n_test = len(ctx.splits["test"])
    v = np.zeros(n_valid, dtype=np.float64)
    v[0] = np.nan
    return {"valid": v, "test": np.zeros(n_test)}
