#!/usr/bin/env python
"""Generate all the reqired data
"""
import os
import pandas as pd
import types
import numpy as np
from basepair.seqmodel import SeqModel
from basepair.BPNet import BPNetSeqModel
from collections import OrderedDict
from basepair.preproc import rc_seq
from basepair.exp.chipnexus.simulate import (insert_motif, generate_sim, plot_sim, generate_seq,
                                             model2tasks, motif_coords, interactive_tracks, plot_motif_table,
                                             plot_sim_motif_col)
from concise.preprocessing import encodeDNA
from basepair.exp.paper.config import models_dir
from basepair.config import get_data_dir, create_tf_session
from basepair.utils import write_pkl
import argparse
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# --------------------------------------------
# Motif configuration
# motif_seqs = OrderedDict([
#     ("Oct4-Sox2", "TTTGCATAACAA"),
#     ("Sox2", "GAACAATGG"),
#     ("Nanog", "AGCCATCA"),
#     ("Klf4", "CCACGCCC"),
# ])
motif_seqs = OrderedDict([
    ("Oct4-Sox2", 'TTTGCATAACAA'),
    ("Oct4", "TATGCAAAT"),
    ("Oct4-Oct4", "TATGCATATGCATA"),
    ("Sox2", "GAACAATGG"),
    ("Nanog", "AGCCATCA"),
    ("Nanog-alt", "GATGGCCCATTTCCT"),
    ("Klf4", "CCACGCCC"),
    ("Klf4-long", "GCCCCGCCCCGCCC"),
    ("B-box", "CCGGGGTTCGAACCCGGG"),
    ("Zic3", "TCTCAGCAGGTAGCA"),
    ("Essrb", "TGACCTTGACCTT")
])

# get also the rc motifs
rc_motif_seqs = OrderedDict([
    (m + "/rc", rc_seq(v))
    for m, v in motif_seqs.items()
])
all_motif_seqs = OrderedDict(list(motif_seqs.items()) + list(rc_motif_seqs.items()))

# center_coords = [485, 520]
center_coords = [465, 535]
# --------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run motif simulation')
    parser.add_argument('exp', help='which experiment to run the simulation for')
    # parser.add_argument('--profile_imp_score', default='profile/wn', help='Profile importance score')
    # parser.add_argument('--count_imp_score', default='counts/pre-act', help='Count importance score')
    parser.add_argument('--repeat', default=128, type=int, help='How many sequences to generate for each example')
    parser.add_argument('--gpu', default=0, type=int, help='Which GPU to use')
    parser.add_argument('--correct', action='store_true', help='Correct for the bleed-through effect')
    args = parser.parse_args()

    exp = args.exp
    # imp_score = args.imp_score
    ddir = get_data_dir()
    repeat = args.repeat

    if args.gpu is not None:
        create_tf_session(args.gpu)
    # create the output path
    cache_path = f"{models_dir}/{exp}/motif-simulation/spacing;correct={args.correct}.pkl"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    # load the model
    logger.info("Loading model")
    model_dir = models_dir / exp
    bpnet = BPNetSeqModel.from_mdir(model_dir)

    logger.info("Creating the output directory")

    df_d = {}
    res_dict_d = {}
    for central_motif_name, central_motif in all_motif_seqs.items():
        logger.info(f"Runnig script for {central_motif_name}")
        # get the motifs
        res_dict = OrderedDict([(motif, generate_sim(bpnet, central_motif, side_motif, list(range(511, 511 + 150, 1)),
                                                     center_coords=center_coords,
                                                     repeat=repeat,
                                                     correct=args.correct,
                                                     importance=[]))  # 'counts/pre-act', 'profile/wn']))
                                for motif, side_motif in all_motif_seqs.items()])
        df = pd.concat([v[0].assign(motif=k) for k, v in res_dict.items()])  # stack the dataframes
        df_d[central_motif_name] = df
        res_dict_d[central_motif_name] = res_dict

    # Store all the results
    write_pkl((df_d, res_dict_d), cache_path)
