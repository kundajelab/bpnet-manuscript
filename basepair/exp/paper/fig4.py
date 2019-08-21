from basepair.exp.chipnexus.motif_clustering import similarity_matrix
from basepair.modisco.motif_clustering import create_pattern_table, align_clustered_patterns
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, cut_tree, leaves_list
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from basepair.exp.
from basepair.plot.config import get_figsize
from basepair.exp.paper.config import tf_colors
from basepair.plot.tracks import plot_track, plot_tracks
from concise.utils.plot import seqlogo
from basepair.plot.utils import strip_axis


def get_patterns(mr_dict, footprints, tasks, min_n_seqlets=300):
    patterns = []
    for task, mr in mr_dict:
        for pattern_name in mr.patterns():
            n_seqlets = mr.n_seqlets(*pattern_name.split("/"))
            if n_seqlets < min_n_seqlets:
                # ignore patterns with few seqlets
                continue
            p = mr.get_pattern(pattern_name)
            p.attrs = {"TF": task, 'n_seqlets': n_seqlets,
                       'features': {'n seqlets': n_seqlets}}
            for t in tasks:
                if t in p.contrib:
                    continue
                else:
                    p.contrib[t] = p.contrib[task]
                    p.hyp_contrib[t] = p.hyp_contrib[task]
            p._tasks = tasks
            p.profile = footprints[task][p.name]
            patterns.append(p.copy().rename(p.name))
    return patterns


def cluster_align_patterns(patterns, n_clusters=9, trials=20, max_shift=15):
    # get the similarity matrix
    sim_seq = similarity_matrix(patterns, track='seq_ic')

    # cluster
    iu = np.triu_indices(len(sim_seq), 1)
    lm_seq = linkage(1 - sim_seq[iu], 'ward', optimal_ordering=True)

    # determine clusters and the cluster order
    cluster = cut_tree(lm_seq, n_clusters=n_clusters)[:, 0]
    cluster_order = np.argsort(leaves_list(lm_seq))

    # align patterns
    patterns_clustered = align_clustered_patterns(patterns, cluster_order, cluster,
                                                  align_track='seq_ic',
                                                  metric='continousjaccard',
                                                  # don't shit the major patterns
                                                  # by more than 15 when aligning
                                                  trials=trials,
                                                  max_shift=max_shift)
    return patterns_clustered


def get_df_info(patterns, tasks):
    pinfo = []
    for p in patterns:
        d = {}
        d['name'] = p.name
        if 'n_seqlets' in p.attrs:
            d['n_seqlets'] = p.attrs['n_seqlets']
        else:
            d['n_seqlets'] = p.attrs['features']['n seqlets']
        d['TF'] = p.attrs['TF']

        d_footprint = {t + "_max": p.profile[t].max()
                       for t in tasks}
        d = {**d, **d_footprint}
        pinfo.append(d)
    df_info = pd.DataFrame(pinfo)
    return df_info


def blank_ax(ax):
    strip_axis(ax)
    ax.axison = False


def plot_patterns(patterns, tasks, pattern_trim=(24, 41), fp_slice=slice(10, 190), n_blank=1):

    df_info = get_df_info(patterns, tasks)
    max_vals = {t: df_info.max()[t + "_max"] for t in tasks}

    fig, axes = plt.subplots(len(patterns), 3 + len(tasks), figsize=get_figsize(1, aspect=1.2))
    fig.subplots_adjust(hspace=0, wspace=0)

    # Get the ylim for each TF
    contrib_ylim = {tf: (min([p.contrib[p.attrs['TF']].min() for p in patterns if p.attrs['TF'] == tf]),
                         max([p.contrib[p.attrs['TF']].max() for p in patterns if p.attrs['TF'] == tf]))
                    for tf in tasks}

    max_digits = max([len(str(p.attrs["n_seqlets"])) for p in patterns])
    for i, p in enumerate(patterns):
        # Motif logo
        ax = axes[i, 0]
        p = p.trim(*pattern_trim).rc()

        seqlogo(p.contrib[p.attrs['TF']], ax=ax)
        ax.set_ylim(contrib_ylim[p.attrs['TF']])  # all plots have the same shape
        strip_axis(ax)
        ax.axison = False
        pos1 = ax.get_position()  # get the original position
        extra_x = pos1.width * 0.2
        pos2 = [pos1.x0, pos1.y0 + pos1.height * 0.4, pos1.width + extra_x, pos1.height * .5]
        ax.set_position(pos2)  # set a new position
        if i == 0:
            ax.set_title("Importance\nscore")

        # Text columns before
        if "/" in p.name:
            pid = p.name.split("_")[-1]
        else:
            pid = p.name.replace("m0_p", "")

        # Oct4/1 150
        # str_n_seqlets = "%*d" % (max_digits, p.attrs["n_seqlets"])
        ax.text(-9, 0, p.attrs["TF"] + "/" + pid, fontsize=8, horizontalalignment='right')
        ax.text(-1, 0, str(p.attrs['n_seqlets']), fontsize=8, horizontalalignment='right')

        ax = axes[i, 1]
        seqlogo(p.get_seq_ic(), ax=ax)
        ax.set_ylim([0, 2])  # all plots have the same shape
        strip_axis(ax)
        ax.axison = False
        pos1 = ax.get_position()  # get the original position
        pos2 = [pos1.x0 + extra_x, pos1.y0 + pos1.height * 0.4, pos1.width + extra_x, pos1.height * .5]
        ax.set_position(pos2)  # set a new position
        if i == 0:
            ax.set_title("Sequence\ninfo. content")

        # ax.text(22, 1, i_to_motif_names[i], fontsize=8, horizontalalignment='center')

        ax = axes[i, 2]
        blank_ax(ax)
        pos1 = ax.get_position()  # get the original position
        pos2 = [pos1.x0 + extra_x * 2, pos1.y0 + pos1.height * 0.4, pos1.width - 2 * extra_x, pos1.height * .5]
        ax.set_position(pos2)  # set a new position
        # if i == 0:
        #    ax.set_title("Assumed\nmotif")

        # Profile columns
        for j, task in enumerate(tasks):
            ax = axes[i, 3 + j]
            fp = p.profile[task]

            ax.plot(fp[fp_slice, 0], color=tf_colors[task])
            ax.plot(-fp[fp_slice, 1], color=tf_colors[task], alpha=0.5)  # negative strand

            ax.set_ylim([-max_vals[task], max_vals[task]])  # all plots have the same shape
            strip_axis(ax)
            ax.axison = False

            if i == 0:
                ax.set_title(task)
    return fig
