#!/usr/bin/env python
"""
Run spacing

"""
import os
import pandas as pd
import types
import numpy as np
from basepair.seqmodel import SeqModel
from collections import OrderedDict
from basepair.preproc import rc_seq
from basepair.config import test_chr, get_data_dir, create_tf_session
from basepair.BPNet import BPNetSeqModel
from basepair.exp.paper.config import models_dir
from basepair.utils import read_pkl
import argparse
from basepair.modisco.pattern_instances import multiple_load_instances, filter_nonoverlapping_intervals, plot_coocurence_matrix, align_instance_center
from basepair.exp.paper.config import profile_mapping
from basepair.utils import flatten
from kipoi.writers import HDF5BatchWriter
from basepair.exp.chipnexus.perturb.gen import generate_data, generate_motif_data, get_reference_profile
from basepair.exp.chipnexus.spacing import motif_pair_dfi, plot_spacing, get_motif_pairs
from basepair.cli.imp_score import ImpScoreFile
from basepair.exp.chipnexus.simulate import profile_sim_metrics
from basepair.plot.profiles import plot_stranded_profile, multiple_plot_stranded_profile
from basepair.plot.heatmaps import heatmap_stranded_profile, multiple_heatmap_stranded_profile, heatmap_stranded_profile
from basepair.modisco.results import ModiscoResult
from basepair.imports import *
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


motifs = OrderedDict([
    ("Oct4-Sox2", 'Oct4/m0_p0'),
    ("Oct4", "Oct4/m0_p1"),
    ("Oct4-Oct4", "Oct4/m0_p6"),
    ("Sox2", "Sox2/m0_p1"),
    ("Nanog", "Nanog/m0_p1"),
    ("Nanog-alt", "Nanog/m0_p4"),
    ("Klf4", "Klf4/m0_p0"),
    ("Klf4-long", "Klf4/m0_p5"),
    ("B-box", "Oct4/m0_p5"),
    ("Zic3", "Nanog/m0_p2"),
    ("Essrb", "Oct4/m0_p16"),
])

# Use a fixed profile slice
PROFILE_SLICE = slice(65, 135)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run motif simulation')
    parser.add_argument('exp', help='which experiment to run the simulation for')
    parser.add_argument('--gpu', default=0, type=int, help='Which GPU to use')
    args = parser.parse_args()

    exp = args.exp
    # imp_score = args.imp_score
    ddir = get_data_dir()

    # load the model
    logger.info("Loading model")
    if args.gpu is not None:
        create_tf_session(args.gpu)
    model_dir = models_dir / exp
    bpnet = BPNetSeqModel.from_mdir(model_dir)

    output_dir = model_dir / 'perturbation-analysis'

    pairs = get_motif_pairs(motifs)

    dfi = multiple_load_instances({task: model_dir / f'deeplift/{task}/out/profile/wn/instances.parq'
                                   for task in bpnet.tasks},
                                  motifs=motifs)
    dfi = filter_nonoverlapping_intervals(dfi)
    # dfi = dfi.iloc[:1000]

    tasks = bpnet.tasks

    # Load imp scores and profiles
    imp_scores = ImpScoreFile(model_dir / 'deeplift.imp_score.h5')
    seqs = imp_scores.get_seq()
    profiles = imp_scores.get_profiles()

    # generate_data also generates some csv files
    dfab, ref, single_mut, double_mut = generate_data(bpnet, dfi, seqs, profiles, pairs, tasks, output_dir)

    logger.info("Running generate_motif_data")
    dfabf_ism, dfabf = generate_motif_data(dfab, ref, single_mut, double_mut, pairs, output_dir,
                                           tasks=tasks, profile_width=200, save=False,
                                           profile_slice=PROFILE_SLICE)
    dfs = dfabf_ism[['Wt_obs', 'Wt', 'dA', 'dB', 'dAB', 'motif_pair',
                     'task', 'center_diff', 'strand_combination']]
    # Store the output
    dfabf_ism.to_csv(output_dir / 'dfabf_ism.csv.gz', compression='gzip')
    dfabf.to_csv(output_dir / 'dfabf.csv.gz', compression='gzip')
    dfs.to_csv(output_dir / 'dfs.csv.gz', compression='gzip')
