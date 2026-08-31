import os
import numpy as np


def fit_predict(ctx):
    os.listdir(".")
    n_valid = len(ctx.splits["valid"])
    n_test = len(ctx.splits["test"])
    return {"valid": np.zeros(n_valid), "test": np.zeros(n_test)}
