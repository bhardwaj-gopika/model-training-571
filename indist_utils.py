"""In-distribution scoring for model inputs.

Measures how well-represented a query setting is in the training data using a
k-nearest-neighbour distance in z-scored feature space. Lower distance = the
setting sits in a densely-sampled region (well in-distribution); high distance
= sparse / out-of-distribution.

Used by end_to_end_demo_haai.py (to label + select overlay samples) and
scan_beam_response.py (to pick the most-represented anchor setting).
"""
from __future__ import annotations

import numpy as np


def zscore_stats(train_X: np.ndarray):
    mu = train_X.mean(axis=0)
    sd = train_X.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    return mu, sd


def knn_distance(query_X: np.ndarray, train_X: np.ndarray, k: int = 10,
                 mu=None, sd=None):
    """Mean distance to the k nearest training points in z-scored space.

    Returns (query_dist, ref_dist):
      * query_dist[i] = mean kNN distance of query row i to the training set
      * ref_dist[j]   = same metric for training row j vs the rest of training
                        (self excluded), for percentile calibration.
    """
    if mu is None or sd is None:
        mu, sd = zscore_stats(train_X)
    Xt = (train_X - mu) / sd
    Xq = (query_X - mu) / sd

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(Xt)
        dq, _ = tree.query(Xq, k=k)
        dq = dq.mean(axis=1) if dq.ndim > 1 else dq
        dref, _ = tree.query(Xt, k=k + 1)      # +1: first neighbour is self
        dref = dref[:, 1:].mean(axis=1)
    except ImportError:
        def mean_knn(A, B, kk, skip_self=False):
            out = np.empty(len(A))
            for i, a in enumerate(A):
                d = np.sqrt(((B - a) ** 2).sum(axis=1))
                d.sort()
                start = 1 if skip_self else 0
                out[i] = d[start:start + kk].mean()
            return out
        dq = mean_knn(Xq, Xt, k)
        dref = mean_knn(Xt, Xt, k, skip_self=True)
    return dq, dref


def indist_percentile(dist: np.ndarray, ref_dist: np.ndarray) -> np.ndarray:
    """Percentile of each `dist` within `ref_dist`.

    ~0   -> denser than almost all training points (very in-distribution)
    ~100 -> farther than almost all training points (out-of-distribution)
    """
    ref_sorted = np.sort(ref_dist)
    idx = np.searchsorted(ref_sorted, dist)
    return 100.0 * idx / max(len(ref_sorted), 1)
