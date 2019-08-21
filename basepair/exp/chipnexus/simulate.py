from collections import OrderedDict
from basepair.plot.tracks import filter_tracks
from basepair.BPNet import BPNet
from basepair.utils import flatten, unflatten
import numpy as np
from copy import deepcopy
import pandas as pd
from scipy.stats import entropy
import random
from concise.preprocessing import encodeDNA
from basepair.plot.tracks import plot_tracks
from basepair.plot.csi import SeqMutationTree, SeqNode
from kipoi.data_utils import get_dataset_item, numpy_collate_concat
from basepair.functions import mean, softmax
from tqdm import tqdm
from kipoi.utils import unique_list
import matplotlib.pyplot as plt
from basepair.modisco.results import Seqlet

# ---------------------
# TODO - put these functions into BPNet model


def model2tasks(model):
    """Get tasks from the model
    """
    return unique_list([l.name.split("/")[1] for l in model.outputs])


def model2seqlen(model):
    """Get the sequence length from the model
    """
    return model.layers[0].output.shape[1].value

# ---------------------


def motif_coords(motif, position):
    start = position - len(motif) // 2
    end = start + len(motif)
    return start, end


def insert_motif(seq, motif, position):
    assert position < len(seq)
    start, end = motif_coords(motif, position)
    new_seq = seq[:start] + motif + seq[end:]
    assert len(new_seq) == len(seq)
    return new_seq


def random_seq(seqlen):
    return ''.join(random.choices("ACGT", k=seqlen))


def generate_seq(central_motif, side_motif=None, side_distances=[], seqlen=1000):
    random_seq = ''.join(random.choices("ACGT", k=seqlen))
    # mlen = len(central_motif)

    injected_seq = insert_motif(random_seq, central_motif, seqlen // 2)
    for d in side_distances:
        injected_seq = insert_motif(injected_seq, side_motif, d)
    return injected_seq


def pred2scale_strands(preds, tasks):
    """Compute the scaling factor for the profile in order to
    obtain the absolute counts
    """
    return {task: np.exp(preds['counts'][i]) - 1
            for i, task in enumerate(tasks)}


def postproc(preds, tasks):
    ntasks = len(tasks)
    preds[:len(tasks)] = [softmax(p) for p in preds[:ntasks]]
    preds_dict = dict(profile=preds[:ntasks],
                      counts=preds[len(tasks):])
    scales = pred2scale_strands(preds_dict, tasks)
    return{task: preds_dict['profile'][i] * scales[task][:, np.newaxis]
           for i, task in enumerate(tasks)}


def average_profiles(p):
    return {k: v.mean(0) for k, v in p.items()}


def simmetric_kl(ref, alt):
    return (entropy(ref, alt) + entropy(alt, ref)) / 2


def profile_sim_metrics(ref, alt, pc=0):
    d = {}
    d['simmetric_kl'] = simmetric_kl(ref, alt).mean() - simmetric_kl(ref, ref).mean()
    d['counts'] = alt.sum()
    d['counts_frac'] = (alt.sum() + pc) / (ref.sum() + pc)
    d['max'] = alt.max()
    d['max_frac'] = (alt.max() + pc) / (ref.max() + pc)

    max_idx = np.argmax(ref, axis=0)
    d['counts_max_ref'] = alt[max_idx, [0, 1]].sum()
    d['counts_max_ref_frac'] = (d['counts_max_ref'] + pc) / (ref[max_idx, [0, 1]].sum() + pc)
    return d


def imp_sim_metrics(ref, alt, motif, seqlen):
    start, end = motif_coords(motif, seqlen // 2)
    return alt[start:end].sum(), alt[start:end].sum() / ref[start:end].sum()


def get_scores(ref_pred, alt_pred, tasks, motif, seqlen, center_coords):
    d = {}
    cstart, cend = center_coords
    for task in tasks:
        # profile - use the filtered tracks
        d[task] = flatten({"profile": profile_sim_metrics(ref_pred['profile'][task][cstart:cend],
                                                          alt_pred['profile'][task][cstart:cend])}, "/")

        # importance scores - use the central motif region
        if 'imp' in ref_pred:
            for imp_score in ref_pred["imp"][task]:
                imp, imp_frac = imp_sim_metrics(ref_pred["imp"][task][imp_score],
                                                alt_pred["imp"][task][imp_score], motif, seqlen)
                d[task] = {f"imp/{imp_score}": imp, f"imp/{imp_score}_frac": imp_frac, **d[task]}
    return d


def single_motif_sim(bpnet, motif, max_hamming_dist=1,
                     center_coords=[450, 550],
                     repeat=128, importance=['count', 'weighted']):
    """Explore the space of different motif mutations

    Args:
      bpnet: BPNet
      motif: which motif to use
      max_hamming_dist: maximum hamming distance to compute
      repeat: how many simulations to run for
      importance: list of importance scores to compute

    Returns:
      dictionary with key=hamming distance from `motif`, values= list of all possible
        sequences `hamming distance` away from `motif`
    """
    seq_by_depth = SeqMutationTree.create(SeqNode(motif),
                                          max_hamming_distance=max_hamming_dist).get_seq_by_depth()
    out = {}
    tasks = bpnet.tasks
    seqlen = bpnet.input_seqlen()
    motif = seq_by_depth[0][0]

    # get the reference value
    ref_pred = unflatten(bpnet.sim_pred(seq_by_depth[0][0], repeat=repeat, importance=importance), "/")
    out[0] = [{"seq": motif, "scores": get_scores(ref_pred, ref_pred,
                                                  tasks, motif, seqlen, center_coords)}]

    for hamming_dist in range(1, len(seq_by_depth)):

        # prepare the output list
        out[hamming_dist] = [None] * len(seq_by_depth[hamming_dist])

        # loop through all the sequences
        for i, alt_seq in enumerate(tqdm(seq_by_depth[hamming_dist])):
            # get the predicitons for the current sequence
            alt_pred = unflatten(bpnet.sim_pred(alt_seq, repeat=repeat, importance=importance), "/")

            # compare ref and alt predictions
            out[hamming_dist][i] = {"seq": alt_seq,
                                    "scores": get_scores(ref_pred, alt_pred, tasks, alt_seq, seqlen,
                                                         center_coords)}
    return out


def motif_sims2df(motif_sims):
    """Convert a dictionary of single_motif_sim outputs into a pd.DataFrame
    """

    return pd.DataFrame([dict(motif=motif,
                              hamming_distance=hd,
                              i=i,
                              seq=iv['seq'],
                              score=score,
                              task=task,
                              value=scorev,
                              )
                         for motif, mv in motif_sims.items()
                         for hd, hv in mv.items()
                         for i, iv in enumerate(hv)
                         for task, taskv in iv['scores'].items()
                         for score, scorev in taskv.items()
                         ])


def generate_sim(bpnet, central_motif, side_motif, side_distances,
                 center_coords=[450, 550], repeat=128, importance=['count', 'weighted'], correct=False):
    outl = []
    tasks = bpnet.tasks
    seqlen = bpnet.input_seqlen()
    # ref_preds = sim_pred(model, central_motif)
    ref_preds = unflatten(bpnet.sim_pred(central_motif,
                                         repeat=repeat,
                                         importance=importance), "/")
    none_preds = unflatten(bpnet.sim_pred('', '', [],
                                          repeat=repeat, importance=importance), "/")

    alt_profiles = []
    for dist in tqdm(side_distances):
        # alt_preds = sim_pred(model, central_motif, side_motif, [dist])

        # Note: bpnet.sim_pred already averages the predictions
        alt_preds = unflatten(bpnet.sim_pred(central_motif, side_motif, [dist],
                                             repeat=repeat, importance=importance), "/")
        if correct:
            # Correct for the 'shoulder' effect
            #
            # this performs: AB - (B - 0)
            # Where:
            # - AB: contains both, central and side_motif
            # - B : contains only side_motif
            # - 0 : doesn't contain any motif
            edge_only_preds = unflatten(bpnet.sim_pred('', side_motif, [dist],
                                                       repeat=repeat, importance=importance), "/")

            alt_preds_f = flatten(alt_preds, '/')
            # ref_preds_f = flatten(ref_preds, '/')
            none_preds_f = flatten(none_preds, "/")
            # substract the other counts
            alt_preds = unflatten({k: alt_preds_f[k] - v + none_preds_f[k]
                                   for k, v in flatten(edge_only_preds, "/").items()}, "/")
            # ref_preds = unflatten({k: ref_preds_f[k] - v  for k,v in flatten(none_preds, "/").items()}, "/")
        alt_profiles.append((dist, alt_preds))

        # This normalizes the score by `A` finally yielding:
        # (AB - B + 0) / A
        scores = get_scores(ref_preds, alt_preds, tasks, central_motif, seqlen, center_coords)

        # compute the distance metrics
        for task in bpnet.tasks:
            d = scores[task]

            # book-keeping
            d['task'] = task
            d['central_motif'] = central_motif
            d['side_motif'] = side_motif
            d['position'] = dist
            d['distance'] = dist - seqlen // 2

            outl.append(d)

    return pd.DataFrame(outl), alt_profiles


def plot_sim(dfm, tasks, variables, motifs=None, subfigsize=(4, 2), alpha=0.5):
    fig, axes = plt.subplots(len(variables), len(tasks),
                             figsize=(subfigsize[0] * len(tasks), subfigsize[1] * len(variables)),
                             sharex=True, sharey='row')
    for i, variable in enumerate(variables):
        for j, task in enumerate(tasks):
            ax = axes[i, j]

            if motifs is not None:
                for motif in motifs:
                    dfms = dfm[(dfm.task == task) & (dfm.motif == motif)]
                    ax.plot(dfms.distance, dfms[variable], label=motif, alpha=alpha)
            else:
                dfms = dfm[dfm.task == task]
                ax.plot(dfms.distance, dfms[variable], label=motif)

            if i == 0:
                ax.set_title(task)
            if i == len(variables) - 1:
                ax.set_xlabel("Distance")
            if j == 0:
                ax.set_ylabel(variable)

            # hard-code
            if variable == 'profile/simmetric_kl':
                ax.axhline(0, linestyle='dashed', color='black', alpha=0.3)
            else:
                ax.axhline(1, linestyle='dashed', color='black', alpha=0.3)

    fig.subplots_adjust(wspace=0, hspace=0)
    if motifs is not None:
        fig.legend(motifs, title="Side motifs")


def plot_sim_motif_col(dfm, tasks, variables, motifs, subfigsize=(4, 2), alpha=0.5):

    # TODO - motifs can be rc
    non_rc_motifs = [m for m in motifs if "/rc" not in m]
    fig, axes = plt.subplots(len(variables), len(non_rc_motifs),
                             figsize=(subfigsize[0] * len(tasks), subfigsize[1] * len(variables)),
                             sharex=True, sharey='row')
    cmap = plt.get_cmap("tab10")

    for i, variable in enumerate(variables):
        for j, motif in enumerate(non_rc_motifs):
            ax = axes[i, j]

            for ti, task in enumerate(tasks):
                dfms = dfm[(dfm.task == task) & (dfm.motif == motif)]
                ax.plot(dfms.distance, dfms[variable],
                        color=cmap(ti),
                        label=task, alpha=alpha)
                if dfm.motif.str.contains(motif + "/rc").any():
                    # add a dotted line for the reverse-complement version of the motif
                    dfms_rc = dfm[(dfm.task == task) & (dfm.motif == motif + "/rc")]
                    ax.plot(dfms_rc.distance, dfms_rc[variable],
                            color=cmap(ti),
                            ls='dotted',
                            label='_nolegend_',
                            alpha=alpha)

            if i == 0:
                ax.set_title(motif)
            if i == len(variables) - 1:
                ax.set_xlabel("Distance")
            if j == 0:
                ax.set_ylabel(variable)

            # hard-code
            if variable == 'profile/simmetric_kl':
                ax.axhline(0, linestyle='dashed', color='black', alpha=0.3)
            else:
                if "frac" in variable:
                    ax.axhline(1, linestyle='dashed', color='black', alpha=0.3)

    fig.subplots_adjust(wspace=0, hspace=0)
    if motifs is not None:
        fig.legend(tasks, title="Tasks")


def interactive_tracks(profiles, central_motif, side_motif, imp_score='weighted'):
    def interactive_tracks_build_fn(profiles, central_motif, side_motif, imp_score):
        p = {k: v['profile'] for k, v in profiles}
        imp = {k: {task: v['imp'][task][imp_score].max(axis=-1) for task in v['imp']}
               for k, v in profiles}
        ymax = max([x.max() for t, v in profiles for x in v['profile'].values()])
        ymax_imp = max([v.max() for x in imp.values() for v in x.values()])
        cstart, cend = motif_coords(central_motif, 500)

        def fn(dist):
            position = dist + 500
            sstart, send = motif_coords(side_motif, position)
            seqlets = [Seqlet(None, cstart, cend, "center", ""),
                       Seqlet(None, sstart, send, "side", "")]
            # TODO - add also importance scores
            du = {"p": p[position], "imp": imp[position]}

            # TODO - order them correctly
            d = OrderedDict([(f"{prefix}/{task}", du[prefix][task])
                             for task in p[position]
                             for prefix in ['p', 'imp']])

            ylims = []
            for k in d:
                if k.startswith("p"):
                    ylims.append((0, ymax))
                else:
                    ylims.append((0, ymax_imp))
            plot_tracks(d,
                        seqlets,
                        title=dist, ylim=ylims)
        return fn
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
    positions, track = zip(*profiles)
    dist = [p - 500 for p in positions]
    return interactive(interactive_tracks_build_fn(profiles, central_motif,
                                                   side_motif, imp_score=imp_score),
                       dist=widgets.IntSlider(min=min(dist),
                                              max=max(dist),
                                              step=dist[1] - dist[0],
                                              value=max(dist)))


def plot_motif_table(mr, motif_consensus):
    """Plot motif table
    """
    from vdom import p, div, img
    from basepair.plot.vdom import fig2vdom, vdom_pssm
    from basepair.modisco.table import longer_pattern
    return div([fig2vdom(mr.plot_pssm(*longer_pattern(pattern).split("/"), trim_frac=0.08, title=f"{motif} ({pattern})"), height=80)
                for motif, (pattern, motif_seq) in motif_consensus.items()])
