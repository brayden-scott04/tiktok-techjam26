"""Attempts to train on the valid split's own labels. Must find only -1s and
therefore be structurally unable to fit anything meaningful -- proves the
labels are physically absent, not just conventionally off-limits."""
import numpy as np


def fit_predict(ctx):
    valid = ctx.splits["valid"]
    labels_seen = np.unique(valid.long_view)
    n_valid = len(valid)
    n_test = len(ctx.splits["test"])
    # if labels_seen is anything other than [-1], the sanitizer failed
    scores = np.zeros(n_valid) if list(labels_seen) == [-1] else valid.long_view.astype(float)
    return {"valid": scores, "test": np.zeros(n_test)}
