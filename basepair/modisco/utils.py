"""Small helper-functions for used by modisco classes
"""
import pandas as pd
import numpy as np
from kipoi.readers import HDF5Reader
from basepair.functions import mean
import warnings


def bootstrap_mean(x, n=100):
    """Bootstrap the mean computation"""
    out = []

    for i in range(n):
        idx = pd.Series(np.arange(len(x))).sample(frac=1.0, replace=True).values
        out.append(x[idx].mean(0))
    outm = np.stack(out)
    return outm.mean(0), outm.std(0)


def nan_like(a, dtype=float):
    a = np.empty(a.shape, dtype)
    a.fill(np.nan)
    return a


def ic_scale(x):
    from modisco.visualization import viz_sequence
    background = np.array([0.27, 0.23, 0.23, 0.27])
    return viz_sequence.ic_scale(x, background=background)


def shorten_pattern(pattern):
    """metacluster_0/pattern_1 -> m1_p1
    """
    return pattern.replace("metacluster_", "m").replace("/", "_").replace("pattern_", "p")


def longer_pattern(shortpattern):
    """m1_p1 -> metacluster_0/pattern_1
    """
    return shortpattern.replace("_", "/").replace("m", "metacluster_").replace("p", "pattern_")


def extract_name_short(ps):
    m, p = ps.split("_")
    return {"metacluster": int(m.replace("m", "")), "pattern": int(p.replace("p", ""))}


def extract_name_long(ps):
    m, p = ps.split("/")
    return {"metacluster": int(m.replace("metacluster_", "")), "pattern": int(p.replace("pattern_", ""))}


def load_imp_scores(h5_file, tasks=None, importance='weighted', incl=None):
    """Load the importance scores from the hdf5 file
    """
    warnings.warn("`load_imp_scores` is deprecated. Use ImpScoreFile directly!")
    
    imp = ImpScoreFile(h5_file, include_samples=incl, default_imp_score=importance)
    seq, contrib, hyp_contrib, profile, ranges = imp.get_all()
    
    if tasks is None:
        tasks = imp.get_tasks()
    
    contrib = {t: contrib[t] for t in tasks}
    hyp_contrib = {t: hyp_contrib[t] for t in tasks}
    profile = {"t/" + t: hyp_contrib[t] for t in tasks}  # to be compatible with old code

    imp.close()
    return seq, contrib, hyp_contrib, profile, ranges