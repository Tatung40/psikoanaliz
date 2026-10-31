"""Monte Carlo precision for pooled coverage, clustering by independent dataset.

Item intervals from one calibration are dependent. For unequal numbers of eligible
intervals, use the delta-method SE of the ratio of covered/eligible totals, with
the independent simulation replication as the cluster. This is not a CI for an
individual model parameter or a new scientific acceptance criterion.
"""
import numpy as np


def coverage_precision(parts, replications, *, eligible_column=None):
    """Each part contains one fitted dataset; its last column is coverage (0/1).

    Optional eligible_column contains withheld flags (0=eligible, 1=withheld).
    Failed/missing fits contribute zero counts; coverage remains conditional on
    eligible intervals. Failure rates must still be reported separately.
    """
    if replications < len(parts) or replications < 1:
        raise ValueError("Invalid independent replication count")
    covered = np.zeros(replications)
    eligible = np.zeros(replications)
    for i, part in enumerate(parts):
        arr = np.asarray(part)
        if arr.ndim != 2 or not np.isin(arr[:, -1], [0, 1]).all():
            raise ValueError("Coverage indicators must be binary")
        keep = np.ones(len(arr), dtype=bool)
        if eligible_column is not None:
            flags = arr[:, eligible_column]
            if not np.isin(flags, [0, 1]).all():
                raise ValueError("Withheld indicators must be binary")
            keep = flags == 0
        eligible[i] = keep.sum()
        covered[i] = arr[keep, -1].sum()
    total = eligible.sum()
    if total == 0 or replications < 2:
        se = None
    else:
        rate = covered.sum() / total
        influence = covered - rate * eligible
        se = float(np.sqrt(replications / (replications - 1) * (influence @ influence)) / total)
    return {"coverage_mc_se": se,
            "coverage_mc_se_method": "replication-cluster delta method (pooled ratio)",
            "independent_replications": replications,
            "eligible_replications": int(np.count_nonzero(eligible)),
            "coverage_cluster_counts": {"covered": covered.astype(int).tolist(),
                                        "eligible": eligible.astype(int).tolist()}}
