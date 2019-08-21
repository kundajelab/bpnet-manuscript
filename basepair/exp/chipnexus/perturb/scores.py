"""
Perturbation scores
"""
import pandas as pd
import numpy as np
from tqdm import tqdm


def profile_count(narrow, wide, profile_slice=None, **kwargs):
    if profile_slice is None:
        profile_slice = np.arange(wide['ref']['pred'].shape[1])
    return (wide['dother']['pred'][:, profile_slice].mean(axis=(1, 2)),
            wide['ref']['pred'][:, profile_slice].mean(axis=(1, 2)))

def log_profile_count(narrow, wide, profile_slice=None, **kwargs):
    if profile_slice is None:
        profile_slice = np.arange(wide['ref']['pred'].shape[1])
    return (np.log(1 + wide['dother']['pred'][:, profile_slice].mean(axis=(1, 2))),
            np.log(1 + wide['ref']['pred'][:, profile_slice].mean(axis=(1, 2))))


def profile_count_norm(narrow, wide, profile_slice=None, **kwargs):
    if profile_slice is None:
        profile_slice = np.arange(wide['ref']['pred'].shape[1])

    return (wide['dother']['pred'][:, profile_slice].mean(axis=(1, 2)) - wide['dboth']['pred'][:, profile_slice].mean(axis=(1, 2)),
            wide['ref']['pred'][:, profile_slice].mean(axis=(1, 2)) - wide['dthis']['pred'][:, profile_slice].mean(axis=(1, 2)))


def log_profile_count_norm(narrow, wide, profile_slice=None, **kwargs):
    if profile_slice is None:
        profile_slice = np.arange(wide['ref']['pred'].shape[1])

    return (np.log(1 + wide['dother']['pred'][:, profile_slice].mean(axis=(1, 2))) - np.log(1 + wide['dboth']['pred'][:, profile_slice].mean(axis=(1, 2))),
            np.log(1 + wide['ref']['pred'][:, profile_slice].mean(axis=(1, 2))) - np.log(1 + wide['dthis']['pred'][:, profile_slice].mean(axis=(1, 2))))


def max_profile_count(narrow, wide, max_position=None, profile_slice=None, **kwargs):
    if profile_slice is None:
        profile_slice = np.arange(wide['ref']['pred'].shape[1])

    if max_position is None:
        max_position = np.argmax(wide['ref']['pred'][:, profile_slice].mean(axis=0), axis=0)

    return (wide['dother']['pred'][:, profile_slice][:, max_position, [0, 1]].mean(axis=-1),
            wide['ref']['pred'][:, profile_slice][:, max_position, [0, 1]].mean(axis=-1))


def max_profile_count_bt(narrow, wide, max_position=None, profile_slice=None, **kwargs):
    """Max profile count bleed-through corrected
    """
    if profile_slice is None:
        profile_slice = np.arange(wide['ref']['pred'].shape[1])

    if max_position is None:
        max_position = np.argmax(wide['ref']['pred'][:, profile_slice].mean(axis=0), axis=0)

    return (wide['dother']['pred'][:, profile_slice][:, max_position, [0, 1]].mean(axis=-1),
            wide['ref']['pred'][:, profile_slice][:, max_position, [0, 1]].mean(axis=-1) -
            wide['dthis']['pred'][:, profile_slice][:, max_position, [0, 1]].mean(axis=-1) +
            wide['dboth']['pred'][:, profile_slice][:, max_position, [0, 1]].mean(axis=-1)
            )


def log_max_profile_count(narrow, wide, max_position=None, **kwargs):
    if max_position is None:
        max_position = np.argmax(wide['ref']['pred'].mean(axis=0), axis=0)

    return (np.log(1 + wide['dother']['pred'][:, max_position, [0, 1]].mean(axis=-1)),
            np.log(1 + wide['ref']['pred'][:, max_position, [0, 1]].mean(axis=-1)))


def max_profile_count_norm(narrow, wide, max_position=None, **kwargs):
    if max_position is None:
        max_position = np.argmax(wide['ref']['pred'].mean(axis=0), axis=0)

    return (wide['dother']['pred'][:, max_position, [0, 1]].mean(axis=-1) - wide['dboth']['pred'][:, max_position, [0, 1]].mean(axis=-1),
            wide['ref']['pred'][:, max_position, [0, 1]].mean(axis=-1) - wide['dthis']['pred'][:, max_position, [0, 1]].mean(axis=-1))


def log_max_profile_count_norm(narrow, wide, max_position=None, **kwargs):
    if max_position is None:
        max_position = np.argmax(wide['ref']['pred'].mean(axis=0), axis=0)

    return (np.log(1 + wide['dother']['pred'][:, max_position, [0, 1]].mean(axis=-1)) - np.log(1 + wide['dboth']['pred'][:, max_position, [0, 1]].mean(axis=-1)),
            np.log(1 + wide['ref']['pred'][:, max_position, [0, 1]].mean(axis=-1)) - np.log(1 + wide['dthis']['pred'][:, max_position, [0, 1]].mean(axis=-1)))


def imp_profile(narrow, wide, **kwargs):
    return (narrow['dother']['imp']['profile'].max(axis=-1).mean(axis=-1),
            narrow['ref']['imp']['profile'].max(axis=-1).mean(axis=-1))


def imp_count(narrow, wide, **kwargs):
    return (narrow['dother']['imp']['count'].max(axis=-1).mean(axis=-1),
            narrow['ref']['imp']['count'].max(axis=-1).mean(axis=-1))


SCORES = [profile_count,
          profile_count_norm,
          max_profile_count,
          max_profile_count_bt,
          max_profile_count_norm,
          imp_profile,
          imp_count]

LOGSCORES = [log_profile_count,
             log_profile_count_norm,
             log_max_profile_count,
             log_max_profile_count_norm,
             imp_profile,
             imp_count]


def compute_features(narrow, wide, SCORES, **kwargs):
    for score in SCORES:
        return {score.__name__: score(narrow, wide, **kwargs)}


def compute_features_tidy(motif_pair_lpdata, tasks,
                          plot_features=SCORES,
                          pseudo_count_quantile=0,
                          profile_slice=slice(82, 118),
                          variable=None,
                          pval=False):
    nf = len(plot_features)

    out = []
    for motif_pair_name, lpdata in tqdm(motif_pair_lpdata.items()):
        motif_pair = list(motif_pair_name.split("<>"))
        dfab_sma = lpdata['dfab'].copy()

        # loop through all possible combinations
        xvals = list(motif_pair_lpdata[motif_pair_name]['x'])

        for task in tasks:
            # compute features
            for score in plot_features:
                dfab_sm = dfab_sma.copy()
                dfab_sm['task'] = task
                dfab_sm['score'] = score.__name__
                for xy in ['x', 'y']:
                    wide = {k: motif_pair_lpdata[motif_pair_name][xy][k]['wide'][task]
                            for k in motif_pair_lpdata[motif_pair_name][xy]}
                    narrow = {k: motif_pair_lpdata[motif_pair_name][xy][k]['narrow'][task]
                              for k in motif_pair_lpdata[motif_pair_name][xy]}
                    dfab_sm[xy + '_alt'], dfab_sm[xy + '_ref'] = score(narrow, wide, profile_slice=profile_slice)
                    dfab_sm[xy + '_alt_ref'] = dfab_sm[xy + '_alt'] / dfab_sm[xy + '_ref']
                    pc = np.percentile(dfab_sm[xy + '_ref'], pseudo_count_quantile * 100)
                    dfab_sm[xy + '_alt_pc'], dfab_sm[xy + '_ref_pc'] = (dfab_sm[xy + '_alt'] + pc,
                                                                        dfab_sm[xy + '_ref'] + pc)
                    dfab_sm[xy + '_alt_ref_pc'] = dfab_sm[xy + '_alt_pc'] / dfab_sm[xy + '_ref_pc']

                out.append(dfab_sm)
    return pd.concat(out, axis=0)


# TODO - figure out the average profile
def average_profile(motif_pair_lpdata, tasks):
    out = []
    for motif_pair_name, lpdata in tqdm(motif_pair_lpdata.items()):
        motif_pair = list(motif_pair_name.split("<>"))
        dfab_sma = lpdata['dfab'].copy()

        # loop through all possible combinations
        xvals = list(motif_pair_lpdata[motif_pair_name]['x'])

        for task in tasks:
            # compute features
            dfab_sm['task'] = task
            for xy in ['x', 'y']:
                wide_ref = motif_pair_lpdata[motif_pair_name][xy]['ref']['wide'][task]
    return pd.concat(out, axis=0)


def ism_compute_features_tidy(motif_pair_lpdata, tasks,):
    out = []
    for motif_pair_name, lpdata in tqdm(motif_pair_lpdata.items()):
        motif_pair = list(motif_pair_name.split("<>"))
        for task in tasks:
            dfab_sm = lpdata['dfab'].copy()
            dfab_sm['task'] = task
            whole = {k: motif_pair_lpdata[motif_pair_name]['x'][k]['whole'][task]
                     for k in motif_pair_lpdata[motif_pair_name]['x']}
            dfab_sm['Wt_obs'] = whole['ref']['obs'].sum(axis=(1, 2))
            dfab_sm['Wt'] = whole['ref']['pred'].sum(axis=(1, 2))
            dfab_sm['dA'] = whole['dthis']['pred'].sum(axis=(1, 2))
            dfab_sm['dB'] = whole['dother']['pred'].sum(axis=(1, 2))
            dfab_sm['dAB'] = whole['dboth']['pred'].sum(axis=(1, 2))
            out.append(dfab_sm)
    return pd.concat(out, axis=0)
