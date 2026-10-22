"""Shared probability transforms used by the final ridge pipeline."""
import numpy as np

def logit(values):
    """Transform clipped proportions to the unbounded modelling scale."""
    clipped = np.clip(np.asarray(values, dtype=float), 0.001, 0.999)
    return np.log(clipped / (1 - clipped))


def probability(values):
    """Return bounded probabilities without overflowing the exponential."""
    return 1 / (1 + np.exp(-np.clip(np.asarray(values), -700, 700)))
