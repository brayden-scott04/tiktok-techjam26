import numpy as np



def fit_predict(ctx):
    # same logic as duplicate_a.py, just extra blank lines and a different comment
    n_valid = len(ctx.splits["valid"])
    n_test = len(ctx.splits["test"])
    return {"valid": np.zeros(n_valid), "test": np.zeros(n_test)}
