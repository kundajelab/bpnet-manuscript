#!/usr/bin/env python
# coding: utf-8
from basepair.modisco.periodicity import smooth
import pandas as pd
import numpy as np
from basepair.exp.chipnexus.spacing import remove_edge_instances, get_motif_pairs, motif_pair_dfi
from basepair.config import test_chr
from pybedtools import BedTool
from basepair.cli.schemas import DataSpec, TaskSpec
from basepair.modisco.pattern_instances import (load_instances, filter_nonoverlapping_intervals,
                                                plot_coocurence_matrix, dfi_filter_valid, dfi_add_ranges, annotate_profile)
# from MPRA.config import model_exps
import sys
from basepair.imports import *
from basepair.exp.paper.config import *
from basepair.utils import flatten
from basepair.modisco.core import Pattern
from plotnine import *
import plotnine
from config import experiments
from basepair.utils import pd_first_cols, flatten
from basepair.exp.chipnexus.comparison import read_peakxus_dfi, read_chexmix_dfi, read_fimo_dfi, read_meme_motifs, read_transfac
from basepair.plot.utils import plt9_tilt_xlab
from basepair.exp.paper.config import models_dir, get_tasks
from basepair.cli.imp_score import ImpScoreFile
from basepair.modisco.results import MultipleModiscoResult

import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '..')


if __name__ == '__main__':
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.NullHandler())

    models = {
        'nexus/profile': 'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE',
        'nexus/binary': 'nexus,gw,OSNK,1,0,0,FALSE,same,0.5,64,25,0.001,9,FALSE',
        'seq/profile': 'seq,peaks,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE,TRUE',
        'seq/binary': 'seq,gw,OSN,1,0,0,FALSE,same,0.5,64,50,0.001,9,FALSE',
        'nexus/profile.peaks-union': 'nexus,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE,TRUE,0',
        'seq/profile.peaks-union': 'seq,nexus-seq-union,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE,TRUE,0'
    }
    models_inv = {v: k for k, v in models.items()}

    dfi_list = []

    def load_data(model_name, imp_score, exp):
        logger.info(f"Model: {model_name}")
        isf = ImpScoreFile(models_dir / exp / 'deeplift.imp_score.h5', default_imp_score=imp_score)
        dfi_subset = pd.read_parquet(models_dir / exp / "deeplift/dfi_subset.parq", engine='fastparquet').assign(model=model_name).assign(exp=exp)
        mr = MultipleModiscoResult({t: models_dir / exp / f'deeplift/{t}/out/{imp_score}/modisco.h5'
                                    for t in get_tasks(exp)})
        return isf, dfi_subset, mr

    # load DataSpec
    # old config
    rdir = get_repo_root()
    dataspec_file = rdir / "src/chipnexus/train/seqmodel/ChIP-nexus.dataspec.yml"
    ds = DataSpec.load(dataspec_file)

    # Load all files into the page cache
    logger.info("Touch all files")
    ds.touch_all_files()

    # --------------------------------------------
    # ### nexus/profile
    model_name = 'nexus/profile'
    imp_score = 'profile/wn'
    exp = models[model_name]
    isf, dfi_subset, mr = load_data(model_name, imp_score, exp)
    ranges_profile = isf.get_ranges()
    profiles = isf.get_profiles()
    # TODO - fix seqlets
    dfi_list.append(annotate_profile(dfi_subset, mr, profiles, profile_width=70, trim_frac=0.08,
                                     profiles_mr={task: {k: profile[ranges_profile.interval_from_task.values == task]
                                                         for k, profile in profiles.items()}
                                                  for task in profiles}))

    # ### nexus/binary
    model_name = 'nexus/binary'
    imp_score = 'class/pre-act'
    exp = models[model_name]
    isf, dfi_subset, mr = load_data(model_name, imp_score, exp)
    ranges_binary = isf.get_ranges()
    assert np.all(ranges_profile == ranges_binary)  # we can directly use the nexus/profile ranges
    dfi_list.append(annotate_profile(dfi_subset, mr, profiles, profile_width=70, trim_frac=0.08,
                                     profiles_mr={task: {k: profile[ranges_binary.interval_from_task.values == task]
                                                         for k, profile in profiles.items()}
                                                  for task in profiles}))

    # ### seq/profile
    model_name = 'seq/profile'
    imp_score = 'profile/wn'
    exp = models[model_name]
    isf, dfi_subset, mr = load_data(model_name, imp_score, exp)
    # Get intervals
    ranges = isf.get_ranges()
    all_intervals = list(BedTool.from_dataframe(ranges[['chrom', 'start', 'end']]))
    # load ChIP-nexus counts
    profiles = ds.load_counts(all_intervals, progbar=True)
    dfi_list.append(annotate_profile(dfi_subset, mr, profiles, profile_width=70, trim_frac=0.08,
                                     profiles_mr={task: {k: profile[ranges.interval_from_task.values == task]
                                                         for k, profile in profiles.items()}
                                                  for task in profiles}))

    # ### seq/binary
    model_name = 'seq/binary'
    imp_score = 'class/pre-act'
    exp = models[model_name]
    isf, dfi_subset, mr = load_data(model_name, imp_score, exp)
    ranges_binary = isf.get_ranges()
    assert np.all(ranges_binary == ranges)  # we can use `profiles` from seq/profile
    dfi_list.append(annotate_profile(dfi_subset, mr, profiles, profile_width=70, trim_frac=0.08,
                                     profiles_mr={task: {k: profile[ranges_binary.interval_from_task.values == task]
                                                         for k, profile in profiles.items()}
                                                  for task in profiles}))

    # ### nexus/profile.peaks-union
    model_name = 'nexus/profile.peaks-union'
    imp_score = 'profile/wn'
    exp = models[model_name]
    isf, dfi_subset, mr = load_data(model_name, imp_score, exp)

    ranges = isf.get_ranges()  # Get intervals
    all_intervals = list(BedTool.from_dataframe(ranges[['chrom', 'start', 'end']]))
    profiles = ds.load_counts(all_intervals, progbar=True)  # load ChIP-nexus counts
    dfi_list.append(annotate_profile(dfi_subset, mr, profiles, profile_width=70, trim_frac=0.08,
                                     profiles_mr={task: {k: profile[ranges.interval_from_task.values == task]
                                                         for k, profile in profiles.items()}
                                                  for task in profiles}))

    # ### seq/profile.peaks-union
    model_name = 'seq/profile.peaks-union'
    imp_score = 'profile/wn'
    exp = models[model_name]
    isf, dfi_subset, mr = load_data(model_name, imp_score, exp)

    # Get intervals
    ranges_seq = isf.get_ranges()

    assert np.all(ranges_seq == ranges)  # we can use `profiles` from nexus/profile.peaks-union
    dfi_list.append(annotate_profile(dfi_subset, mr, profiles, profile_width=70, trim_frac=0.08,
                                     profiles_mr={task: {k: profile[ranges_seq.interval_from_task.values == task]
                                                         for k, profile in profiles.items()}
                                                  for task in profiles}))

    outfile = models_dir / 'dfi_subset.annotated.parq'
    logger.info(f"Concat and write to {outfile}")
    dfi = pd.concat(dfi_list)
    dfi.to_parquet(models_dir / 'dfi_subset.annotated.v3.parq', index=False, engine='fastparquet')

    logger.info("Done!")
