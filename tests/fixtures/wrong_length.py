import numpy as np


def fit_predict(ctx):
    n_test = len(ctx.splits["test"])
    return {"valid": np.zeros(10), "test": np.zeros(n_test)}
