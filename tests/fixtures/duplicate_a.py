import numpy as np


def fit_predict(ctx):
    # a trivial constant baseline
    n_valid = len(ctx.splits["valid"])
    n_test = len(ctx.splits["test"])
    return {"valid": np.zeros(n_valid), "test": np.zeros(n_test)}
