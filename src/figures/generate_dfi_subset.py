"""


# Read the output file with:
dfi_subset2 = pd.read_parquet(f"{model_dir}/deeplift/dfi_subset.parq", engine='fastparquet')
"""
# !/usr/bin/env python
# coding: utf-8

from basepair.exp.chipnexus.simulate import (insert_motif, generate_sim, plot_sim, generate_seq,
                                             model2tasks, motif_coords, interactive_tracks, plot_motif_table,
                                             plot_sim_motif_col)
from basepair.modisco.pattern_instances import (multiple_load_instances, load_instances, filter_nonoverlapping_intervals,
                                                plot_coocurence_matrix, align_instance_center, dfi2seqlets, annotate_profile)
from genomelake.extractors import FastaExtractor
from basepair.exp.paper.config import fasta_file
from basepair.extractors import Interval
from basepair.exp.chipnexus.spacing import co_occurence_matrix
from basepair.exp.chipnexus.spacing import coocurrence_plot
from basepair.exp.paper.fig4 import cluster_align_patterns
from basepair.modisco.pattern_instances import annotate_profile
from basepair.exp.paper.config import tasks
import warnings
import plotnine
from plotnine import *
from scipy.fftpack import fft, ifft
from copy import deepcopy
from basepair.preproc import rc_seq, dfint_no_intersection
from basepair.cli.imp_score import ImpScoreFile
from basepair.cli.modisco import load_profiles
from basepair.exp.paper.config import *
from basepair.exp.chipnexus.spacing import remove_edge_instances, get_motif_pairs, motif_pair_dfi
from basepair.exp.chipnexus.perturb.vdom import vdom_motif_pair, plot_spacing_hist
from basepair.modisco.results import MultipleModiscoResult, Seqlet, resize_seqlets
from basepair.plot.tracks import plot_tracks, filter_tracks
from basepair.plot.profiles import plot_stranded_profile, multiple_plot_stranded_profile, extract_signal
from basepair.plot.heatmaps import heatmap_stranded_profile, multiple_heatmap_stranded_profile
from basepair.math import softmax
from basepair.imports import *
from collections import OrderedDict
from basepair.utils import kwargs_str2kwargs
import argparse


if __name__ == "__main__":
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())
    # ---------------------------------
    parser = argparse.ArgumentParser(description='Generate dfi_subset')
    parser.add_argument('exp', help='All required parameters for the job')
    parser.add_argument('--imp-score', default='profile/wn', help='Which imp score to use')
    parser.add_argument('--append-profile', action='store_true', help='Append profile features if enabled')
    parser.add_argument('--motifs', help='Motifs as a json file')

    args = parser.parse_args()

    exp = args.exp
    imp_score = args.imp_score
    motifs = kwargs_str2kwargs(args.motifs)
    print("exp:", exp)
    print("imp_score:", imp_score)
    print("motifs:", motifs)
    task_map = {"O": "Oct4", "S": "Sox2", "N": "Nanog", "K": "Klf4"}
    # Figure out the tasks from the name
    try:
        tasks = [task_map[s] for s in exp.split(",")[2]]
    except:
        # HACK to run it also for other exp names
        tasks = [task_map[s] for s in exp.split(",")[3]]
    print("tasks:", tasks)
    # exp = 'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE'
    # imp_score = 'profile/wn'

    # motifs = OrderedDict([
    #     ("Oct4-Sox2", 'Oct4/m0_p0'),
    #     ("Oct4", 'Oct4/m0_p1'),
    #     # ("Strange-sym-motif", 'Oct4/m0_p5'),
    #     ("Sox2", 'Sox2/m0_p1'),
    #     ("Nanog", 'Nanog/m0_p1'),
    #     ("Zic3", 'Nanog/m0_p2'),
    #     ("Nanog-partner", 'Nanog/m0_p4'),
    #     ("Klf4", 'Klf4/m0_p0'),
    # ])

    # Imports
    warnings.filterwarnings("ignore")

    # interval columns in dfi
    interval_cols = ['example_chrom', 'pattern_start_abs', 'pattern_end_abs']

    # figures dir
    model_dir = models_dir / exp
    fdir = Path(f'{ddir}/figures/modisco/{exp}/spacing/')
    fdir_individual = fdir / 'individual'
    fdir_individual_sim = fdir / 'individual-simulation'

    # Generate motif pairs
    pairs = get_motif_pairs(motifs)
    pair_names = ["<>".join(x) for x in pairs]

    # define the global set of distances
    dist_subsets = ['center_diff<=35',
                    '(center_diff>35)&(center_diff<=70)',
                    '(center_diff>70)&(center_diff<=150)',
                    'center_diff>150']
    dist_subset_labels = ['dist < 35',
                          '35 < dist <= 70',
                          '70 < dist <= 150',
                          '150 < dist',
                          ]
    # ---------------------------------

    mr = MultipleModiscoResult({t: model_dir / f'deeplift/{t}/out/{imp_score}/modisco.h5'
                                for t in tasks})

    main_motifs = [mr.get_pattern(pattern_name).rename(name)
                   for name, pattern_name in motifs.items()]

    # Load instances
    logger.info("Load instances")
    instance_parq_paths = {t: model_dir / f'deeplift/{t}/out/{imp_score}/instances.parq'
                           for t in tasks}
    dfi = multiple_load_instances(instance_parq_paths, motifs)

    # Subset the motifs
    dfi_subset = dfi.query('match_weighted_p > .2').query('imp_weighted_p > .01')

    # ### Append profile features
    logger.info("Append profile features")
    if args.append_profile:
        isf = ImpScoreFile(model_dir / 'deeplift.imp_score.h5', default_imp_score=imp_score)
        ranges = isf.get_ranges()
        profiles = isf.get_profiles()
        dfi_subset = annotate_profile(dfi_subset, mr, profiles, profile_width=70,
                                      trim_frac=0.08,
                                      profiles_mr={task: {k: profile[ranges.interval_from_task.values == task]
                                                          for k, profile in profiles.items()}
                                                   for task in profiles})

    # #### Exclude TE's
    logger.info("Exclude TEs")

    def shorten_te_pattern(s):
        tf, p = s.split("/", 1)
        return tf + "/" + shorten_pattern(p)

    motifs_te = [p.name
                 for p in mr.get_all_patterns()
                 if p.seq_info_content > 30 and mr.n_seqlets(p.name) > 100]
    motifs_te_d = OrderedDict([(shorten_te_pattern(x), shorten_te_pattern(x)) for x in motifs_te])

    # # get transposable element locations
    dfi_te = multiple_load_instances(instance_parq_paths, motifs_te_d)
    dfi_te = dfi_te[(dfi_te.match_weighted_p > 0.1) & (dfi_te.seq_match > 20)]

    # Get rows without intersecting transposable elements
    non_te_idx = dfint_no_intersection(dfi_subset[interval_cols], dfi_te[interval_cols])

    # Append non_te_idx
    dfi_subset['is_te'] = True
    dfi_subset['is_te'][non_te_idx] = False

    print(f"Not overlapping te's: {non_te_idx.mean()}")

    # Add clustered motifs locations
    logger.info("Append motif_center_aln")
    main_motifs = [mr.get_pattern(pattern_name).add_attr('features', {'n seqlets': mr.n_seqlets(pattern_name)})
                   for name, pattern_name in motifs.items()]
    main_motifs = [p.rename(longer_pattern(p.name)) for p in main_motifs]

    try:
        main_motifs_clustered = cluster_align_patterns(main_motifs, n_clusters=1)
        dfi_subset = align_instance_center(dfi_subset, main_motifs, main_motifs_clustered, trim_frac=0.08)
    except:
        logger.error('Unable to run align_instance_center')

    logger.info("Store the file")
    dfi_subset.to_parquet(f"{model_dir}/deeplift/dfi_subset.parq", index=False, engine='fastparquet')


# def norm_matrix(s):
#     """Create the normalization matrix

#     Example:
#     print(norm_matrix(pd.Series([1,3,5])).to_string())
#        0  1  2
#     0  1  1  1
#     1  3  3  3
#     2  5  5  5

#     Args:
#       s: pandas series
#     """
#     tnc = s.values[:, np.newaxis]
#     vals_by_row = tnc * np.ones_like(tnc).T
#     # np.fill_diagonal(vals_by_row,  1)
#     return pd.DataFrame(vals_by_row, index=s.index, columns=s.index)

# # normalization: minimum number of counts
# total_number = dfi_subset.groupby(['pattern']).size()
# norm_counts = norm_matrix(total_number)

# # cross-product
# dfi_filt_crossp = pd.merge(dfi_subset[['example_idx', 'pattern', 'pattern_center_aln', 'pattern_strand_aln', 'pattern_center']],
#                            dfi_subset[['example_idx', 'pattern', 'pattern_center_aln', 'pattern_strand_aln', 'pattern_center']],
#                            how='outer', left_on='example_idx', right_on='example_idx').reset_index()
# # remove self-matches
# dfi_filt_crossp = dfi_filt_crossp.query('~((pattern_x == pattern_y) & (pattern_center_aln_x == pattern_center_aln_y) & (pattern_strand_aln_x == pattern_strand_aln_x))')
# # dfi_filt_crossp = dfi_filt_crossp.query('(pattern_x != pattern_y)')

# # order the matrix by names
# idx_order = [p.name for p in main_motifs_clustered if p.name in dfi_subset.pattern.unique()]

# dfi_sp = dfi_filt_crossp.query('(abs(pattern_center_aln_x- pattern_center_aln_y) == 0) & (pattern_strand_aln_x == pattern_strand_aln_x)')
# match_sizes = dfi_sp.groupby(['pattern_x', 'pattern_y']).size()
# count_matrix = match_sizes.unstack(fill_value=0)

# norm_count_matrix = count_matrix / norm_counts# .truediv(min_counts, axis='columns').truediv(total_number, axis='index')
# norm_count_matrix = norm_count_matrix.fillna(0)  # these examples didn't have any paired pattern

# %opts HeatMap [xrotation=90] (cmap='Blues')
# norm_count_matrix[idx_order].loc[idx_order].stack().reset_index().hvplot.heatmap(x='level_0', y='level_1', C="0", width=600, height=600, colorbar=True)  # TODO add vmax
