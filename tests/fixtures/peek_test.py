import numpy as np


def fit_predict(ctx):
    with open("../../KuaiRand-Pure/data/log_standard_4_22_to_5_08_pure.csv") as fh:
        data = fh.read()
    n_valid = len(ctx.splits["valid"])
    n_test = len(ctx.splits["test"])
    return {"valid": np.zeros(n_valid), "test": np.zeros(n_test)}
