"""Useful functions for plotting locus specific importance scores
"""
import pandas as pd
import numpy as np
from collections import OrderedDict
from basepair.modisco.results import Seqlet
from basepair.preproc import resize_interval, parse_interval
from basepair.seqmodel import SeqModel
from basepair.utils import unflatten
from genomelake.extractors import FastaExtractor
import pybedtools
from basepair.preproc import moving_average
from basepair.modisco.utils import shorten_pattern
from basepair.utils import flatten_list

dfi_cols = ['pattern', 'pattern_center', 'imp_weighted_p', 'match_weighted_p']


def shorten_te_pattern(s):
    tf, p = s.split("/", 1)
    return tf + "/" + shorten_pattern(p)


def str_interval(interval, xlim=None):
    """Convert interval to string
    """
    if xlim is not None:
        start = interval.start + xlim[0]
        end = interval.start + xlim[1]
    else:
        start = interval.start
        end = interval.end

    return f"{interval.chrom}:{start}-{end}, {interval.name}"


def to_neg(track):
    """Use the negative sign for reads on the reverse strand
    """
    track = track.copy()
    track[:, 1] = - track[:, 1]
    return track


def trim_seq(seq_width, peak_width):
    if seq_width > peak_width:
        # Trim
        # make sure we can nicely trim the peak
        assert (seq_width - peak_width) % 2 == 0
        trim_start = (seq_width - peak_width) // 2
        trim_end = seq_width - trim_start
        assert trim_end - trim_start == peak_width
    elif seq_width == peak_width:
        trim_start = 0
        trim_end = peak_width
    else:
        raise ValueError("seq_width < peak_width")
    return trim_start, trim_end


def interval_predict(bpnet, dataspec, interval, tasks, smooth_obs_n=0, neg_rev=True, incl_pred=False):
    input_seqlen = 1000 - bpnet.body.get_len_change() - bpnet.heads[0].net.get_len_change()

    int_len = interval.end - interval.start
    if int_len != input_seqlen:
        print(f"resizing the interval of length {int_len} to {input_seqlen}")
        interval = resize_interval(interval, input_seqlen)

    # fetch the sequence
    fe = FastaExtractor(dataspec.fasta_file)
    seq = fe([interval])
    # Fetch read counts
    obs = {task: dataspec.task_specs[task].load_counts([interval])[0] for task in tasks}
    if smooth_obs_n > 0:
        obs = {k: moving_average(v, n=smooth_obs_n) for k,v in obs.items()}
        
    # TODO  have the function to get the right trimming

    trim_i, trim_j = trim_seq(input_seqlen, 1000)

    # Compute importance scores
    imp_scores = bpnet.imp_score_all(seq, preact_only=True)

    # Make predictions
    # x = bpnet.neutral_bias_inputs(1000, 1000)
    # x['seq'] = seq
    # preds = bpnet.predict(x)

    # Compile everything into a single ordered dict
    if incl_pred:
        preds = bpnet.predict(seq)

        def proc_pred(preds, task, neg_rev):
            out =  preds[f"{task}/profile"][0] * np.exp(preds[f"{task}/counts"][0])
            if neg_rev:
                return to_neg(out)
            else:
                return out
    
        viz_dict = OrderedDict(flatten_list([[
            (f"{task} Obs", to_neg(obs[task]) if neg_rev else obs[task]),
            (f"{task} Pred", proc_pred(preds, task, neg_rev)),
            # (f"{task} Imp counts", sum(pred['grads'][task_idx]['counts'].values()) / 2 * seq),
        ] + [(f"{task} Imp profile", (v * seq)[0])
             for imp_score, v in unflatten(imp_scores, "/")[task]['profile'].items() if imp_score == 'wn'
             ]
            for task_idx, task in enumerate(tasks)]))
    else:
        viz_dict = OrderedDict(flatten_list([[
            # (f"{task} Pred", to_neg(preds[f"{task}/profile"][0])),
            (f"{task} Obs", to_neg(obs[task]) if neg_rev else obs[task]),
            # (f"{task} Imp counts", sum(pred['grads'][task_idx]['counts'].values()) / 2 * seq),
        ] + [(f"{task} Imp profile", (v * seq)[0])
             for imp_score, v in unflatten(imp_scores, "/")[task]['profile'].items() if imp_score == 'wn'
             ]
            for task_idx, task in enumerate(tasks)]))
    return viz_dict, seq, imp_scores


def get_ylim(viz_dict, tasks, profile_per_tf=False, neg_rev=True):
    # ylim
    features = {k.split(" ", 1)[1] for k in viz_dict}
    fmax = {feature: max([np.abs(viz_dict[f"{task} {feature}"]).max() for task in tasks])
            for feature in features}  # 'Pred',
    fmin = {feature: min([viz_dict[f"{task} {feature}"].min() for task in tasks])
            for feature in features}  # 'Pred',
    ylim = []
    for k in viz_dict:
        task, f = k.split(" ", 1)
        if "Imp" in f:
            ylim.append((fmin[f], fmax[f]))
        else:
            if profile_per_tf:
                m = np.abs(viz_dict[k]).max()
            else:
                m = fmax[f]
            if neg_rev:
                ylim.append((-m, m))
            else:
                ylim.append((0, m))
    return ylim


def get_instances_single(pattern, seq, imp_scores, imp_score, centroid_seqlet_matches, tasks, n_jobs=1):
    task = pattern.name.split("/")[0]
    pname_short = "/".join(pattern.name.split("/")[1:])

    hyp_contrib = {task: imp_scores[f'{task}/{imp_score}'] for task in tasks}
    contrib = {k: v * seq for k, v in hyp_contrib.items()}

    match, importance = pattern.scan_importance(contrib, hyp_contrib, [task],
                                                n_jobs=n_jobs, verbose=False)
    seq_match = pattern.scan_seq(seq, n_jobs=n_jobs, verbose=False)
    norm_df = centroid_seqlet_matches[task].query(f"pattern == '{pname_short}'")
    dfm = pattern.get_instances([task], match, importance, seq_match,
                                norm_df=norm_df,
                                verbose=False, plot=False)
    return dfm


def get_instances(patterns, seq, imp_scores, imp_score, centroid_seqlet_matches, motifs, tasks):
    motifs_inv = {v: k for k, v in motifs.items()}
    dfim = pd.concat([(get_instances_single(p, seq, imp_scores, imp_score, centroid_seqlet_matches, tasks)
                       .assign(motif=motifs_inv[shorten_te_pattern(p.name)]))
                      for p in patterns], axis=0)
    dfim.example_idx = dfim.imp_max_task + ' Imp profile'
    return dfim.sort_values('match_weighted_p', ascending=False)


def dfi_row2seqlet(row, motifs_inv):
    seqlet = Seqlet(row.example_idx,
                    row.pattern_start,
                    row.pattern_end,
                    name=motifs_inv[shorten_te_pattern(row.pattern)],
                    strand=row.strand)
    seqlet.alpha = row.match_weighted_p
    return seqlet


def dfi2seqlets(dfi, motifs_inv):
    """Convert the data-frame produced by pattern.get_instances()
    to a list of Seqlets

    Args:
      dfi: pd.DataFrame returned by pattern.get_instances()
      short_name: if True, short pattern name will be used for the seqlet name

    Returns:
      Seqlet list
    """
    return [dfi_row2seqlet(row, motifs_inv) for i, row in dfi.iterrows()]
