import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from basepair.plot.config import get_figsize
import plotnine
from plotnine import *
from vdom.helpers import (h1, p, li, img, div, b, br, ul,
                          details, summary,
                          table, thead, th, tr, tbody, td, ol)
from basepair.plot.vdom import fig2vdom
from basepair.plot.heatmaps import RowQuantileNormalizer, QuantileTruncateNormalizer
from basepair.data import NumpyDataset
from basepair.plot.profiles import plot_stranded_profile, multiple_plot_stranded_profile
from basepair.plot.heatmaps import (heatmap_stranded_profile, multiple_heatmap_stranded_profile,
                                    multiple_heatmap_importance_profile,
                                    multiple_heatmaps, heatmap_sequence)


def plot_spacing_hist(dfab, max_dist=200):
    plotnine.options.figure_size = get_figsize(1, aspect=2 / 10 * 3)
    fig = (ggplot(aes(x='center_diff', fill='strand_combination'), dfab[(dfab.center_diff <= max_dist)]) +
           geom_vline(xintercept=10, alpha=0.1) +
           geom_vline(xintercept=20, alpha=0.1) +
           geom_vline(xintercept=30, alpha=0.1) +
           geom_vline(xintercept=40, alpha=0.1) +
           geom_histogram(bins=max_dist) + facet_grid("strand_combination~.") +
           theme_classic() +
           theme(strip_text=element_text(rotation=0), legend_position='top') +
           xlim([0, max_dist]) +
           xlab("Pairwise distance") +
           # ggtitle("Nanog<>Nanog") +
           scale_fill_brewer(type='qual', palette=3))
    return fig


def prepare_profiles(lpdata, site, measured, motif_pair):
    xy_hash = {motif_pair[0]: 'x', motif_pair[1]: 'y'}
    this = site
    other = motif_pair[0] if motif_pair[0] != this else motif_pair[1]
    return {"Ref observed": lpdata[xy_hash[site]]['ref']['wide'][measured]['obs'],
            "Ref predicted": lpdata[xy_hash[site]]['ref']['wide'][measured]['pred'],
            f"Other {other} removed": lpdata[xy_hash[site]]['dother']['wide'][measured]['pred'],
            f"This {this} removed": lpdata[xy_hash[site]]['dthis']['wide'][measured]['pred'],
            "Both removed": lpdata[xy_hash[site]]['dboth']['wide'][measured]['pred']
            }


def prepare_imp(lpdata, site, measured, motif_pair, which='profile'):
    xy_hash = {motif_pair[0]: 'x', motif_pair[1]: 'y'}
    this = site
    other = motif_pair[0] if motif_pair[0] != this else motif_pair[1]
    return {"Ref": lpdata[xy_hash[site]]['ref']['wide'][measured]['imp'][which].max(axis=-1),
            f"Other {other} removed": lpdata[xy_hash[site]]['dother']['wide'][measured]['imp'][which].max(axis=-1),
            f"This {this} removed": lpdata[xy_hash[site]]['dthis']['wide'][measured]['imp'][which].max(axis=-1),
            "Both removed": lpdata[xy_hash[site]]['dboth']['wide'][measured]['imp'][which].max(axis=-1)
            }


def imp2seq(x):
    o = np.where(x == 0, x, np.ones_like(x))
    assert np.all(o.max(axis=-1) == 1)
    return o


def prepare_seq(lpdata, site, measured, motif_pair):
    """Get the DNA sequences used in this analysis
    """
    xy_hash = {motif_pair[0]: 'x', motif_pair[1]: 'y'}
    which = 'profile'
    this = site
    other = motif_pair[0] if motif_pair[0] != this else motif_pair[1]
    values = {"Ref": lpdata[xy_hash[site]]['ref']['wide'][measured]['imp'][which],
              f"Other {other} removed": lpdata[xy_hash[site]]['dother']['wide'][measured]['imp'][which],
              f"This {this} removed": lpdata[xy_hash[site]]['dthis']['wide'][measured]['imp'][which],
              "Both removed": lpdata[xy_hash[site]]['dboth']['wide'][measured]['imp'][which]
              }
    return {k: imp2seq(v) for k, v in values.items()}


def cache_fig(fig, output_dir, url_dir, k, width=840, cache=True):
    import plotnine
    output_file = os.path.join(output_dir, k + ".png")
    output_file_url = os.path.join(url_dir, k + ".png")
    if not (os.path.exists(output_file) and cache):
        if isinstance(fig, plotnine.ggplot):
            fig.save(output_file)
        else:
            fig.savefig(output_file)
    return img(src=output_file_url, width=width)


def motif_pair_figs(main_motif, measure, motif_pair, lpdata, dfab,
                    normalizer=RowQuantileNormalizer(pmin=50, pmax=99),
                    subset_profile_agg=True,
                    top_n=1500, max_dist=150, figsize=(15, 15)):
    """Plot the vdom for the main motif

    - Profile this task
    - Importance (profile) this task
    - Importance (count) this task
    - Profile other task
    - Importance (profile) other task
    - Importance (count) other task
    """
    from basepair.exp.chipnexus.spacing import plot_spacing
    profiles = prepare_profiles(lpdata, main_motif, measure, motif_pair)
    counts = profiles['Ref observed'].sum(axis=(1, 2))
    assert len(counts) == len(dfab)  # make sure dfab length matches counts
    sort_idx_top_counts = np.argsort(-counts)[:top_n]
    distances = dfab.center_diff.values
    top_counts_sort_idx = sort_idx_top_counts[np.argsort(distances[sort_idx_top_counts])]
    # Make sure the maximum distance is < max_dist
    # top_counts_sort_idx = top_counts_sort_idx[distances[top_counts_sort_idx] < max_dist]

    # pairwise distance
    fig_distance = plt.figure(figsize=get_figsize(.5))
    plt.plot(dfab.center_diff.iloc[top_counts_sort_idx].values)
    plt.ylabel("Pairwise distance")
    plt.xlabel("Seqlet index")

    fig_spacing_hist = plot_spacing_hist(dfab)
    fig_spacing_hist2 = plot_spacing(dfab, alpha_scatter=0.05, y_feature='imp_weighted', 
                                     figsize=get_figsize(.4, aspect=2))
    plt.close()
    
    if subset_profile_agg:
        fig_profile_agg = multiple_plot_stranded_profile(NumpyDataset(profiles)[top_counts_sort_idx],
                                                         figsize_tmpl=(2.55, 2))
    else:
        fig_profile_agg = multiple_plot_stranded_profile(profiles,
                                                         figsize_tmpl=(2.55, 2))
    plt.close()
    fig_profile_this = multiple_heatmap_stranded_profile(profiles, sort_idx=top_counts_sort_idx,
                                                         figsize=figsize, normalizer=normalizer)
    plt.close()
    imp = prepare_imp(lpdata, main_motif, measure, motif_pair)
    plt.close()
    fig_imp_profile_this = multiple_heatmap_importance_profile(imp, sort_idx=top_counts_sort_idx,
                                                               figsize=figsize)
    plt.close()
    imp = prepare_imp(lpdata, main_motif, measure, motif_pair, 'count')
    fig_imp_count_this = multiple_heatmap_importance_profile(imp, sort_idx=top_counts_sort_idx,
                                                             figsize=figsize)
    plt.close()

    # sequence logo
    seq = prepare_seq(lpdata, main_motif, measure, motif_pair)
    fig_seq = multiple_heatmaps(seq, heatmap_sequence, sort_idx=top_counts_sort_idx,
                                figsize=(15, 10), aspect='auto')
    plt.close()

    dist_seq = [('Distance', [fig_distance, fig_spacing_hist, fig_spacing_hist2]),
                (f'Sequences', [fig_seq])]
    return [(f'Profile (task={measure})', [fig_profile_agg, fig_profile_this]),
            (f'Importance profile (task={measure})', [fig_imp_profile_this]),
            (f'Importance count (task={measure})', [fig_imp_count_this])], dist_seq
            


def template_vdom_motif_pair(main_motif, motif_pair, lpdata, dfab, profile_mapping,
                             figures_dir, figures_url, cache=False, **kwargs):
    
    motif_figs = []
    other = motif_pair[0] if motif_pair[0] != main_motif else motif_pair[1]
    # HACK - pass the tasks
    for task in ['Oct4', 'Sox2', 'Nanog', 'Klf4']:
        figs, dist_seq = motif_pair_figs(main_motif, task, motif_pair, lpdata, dfab, **kwargs)
        motif_figs += figs
        if task == profile_mapping[main_motif]:
            dist_seq_fig = dist_seq
            
    return details(summary(b(main_motif), f" (perturb {other})"),
                   *[details(summary(k), *[cache_fig(x, figures_dir, figures_url, str(i) + "-" + main_motif + "<>" + other + "-" + k,
                                                     width=840 if k is not "Distance" else 400, cache=cache) for i, x in enumerate(v)])
                     for k, v in dist_seq_fig + motif_figs]
                   )


def vdom_motif_pair(motif_pair_lpdata, dfab, profile_mapping, figures_dir, figures_url,
                    profile_width=200, cache=False, **kwargs):
    from basepair.exp.chipnexus.spacing import remove_edge_instances
    out = []
    os.makedirs(figures_dir, exist_ok=True)
    for motif_pair_name, lpdata in tqdm(motif_pair_lpdata.items()):

        motif_pair = list(motif_pair_name.split("<>"))
        dfab_subset = remove_edge_instances(dfab[dfab.motif_pair == motif_pair_name],
                                            profile_width=profile_width)
        out.append(template_vdom_motif_pair(motif_pair[0], motif_pair,
                                            lpdata, dfab_subset, profile_mapping,
                                            figures_dir=figures_dir, figures_url=figures_url, cache=cache, **kwargs))
        if motif_pair[0] != motif_pair[1]:
            out.append(template_vdom_motif_pair(motif_pair[1], motif_pair,
                                                lpdata, dfab_subset, profile_mapping,
                                                figures_dir=figures_dir, figures_url=figures_url, cache=True, **kwargs))
        # break
    return div(ul([li(elem) for elem in out], start=0))
