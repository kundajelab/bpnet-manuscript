"""Paper config file
"""
from collections import OrderedDict
from pathlib import Path
from basepair.config import get_data_dir
from basepair.metrics import PeakPredictionProfileMetric
ddir = get_data_dir()

# Main tasks
tasks = ['Oct4', 'Sox2', 'Nanog', 'Klf4']

def get_tasks(exp):
    """Get tasks from experiment ID
    """
    task_map = {"O": "Oct4", "S": "Sox2", "N": "Nanog", "K": "Klf4"}
    tasks = [task_map[s] for s in exp.split(",")[2]]
    return tasks


# Common paths
models_dir = Path('/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/output/')
# Main ChIP-nexus run
exp_chipnexus_id = 'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE,[1,50],TRUE'
# exp_chipnexus_id = 'nexus,peaks,OSNK,0,10,1,FALSE,same,0.5,64,25,0.004,9,FALSE-2'
exp_chipnexus_model_dir = models_dir / exp_chipnexus_id
exp_chipnexus_imp_score_file = exp_chipnexus_model_dir / "deeplift.imp_score.h5"
exp_chipnexus_modisco_dirs = {t: exp_chipnexus_model_dir / f'deeplift/{t}/out/profile/wn'
                              for t in tasks}


exp_chipseq_id = 'seq,peaks,OSN,0,10,1,FALSE,same,0.5,64,50,0.004,9,FALSE,[1,50],TRUE/'
exp_chipseq_model_dir = models_dir / exp_chipseq_id
exp_chipseq_imp_score_file = exp_chipseq_model_dir / "deeplift.imp_score.h5"
exp_chipseq_modisco_dirs = {t: exp_chipseq_model_dir / f'deeplift/{t}/out/profile/wn'
                            for t in tasks}

# TODO add other paths as well
fasta_file = '/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/data/mm10_no_alt_analysis_set_ENCODE.fasta'


# specify motifs to use in the analysis
# motifs = OrderedDict([
#     ("Oct4-Sox2", "m0_p0"),
#     # ("Oct4-Sox2-deg", "m6_p8"),
#     # ("Oct4", "m0_p18"),
#     ("Sox2", "m0_p1"),
#     # ("Essrb", "m0_p2"),
#     ("Nanog", "m2_p0"),
#     # ("Nanog-periodic", "m0_p9"),
#     ("Klf4", "m1_p0"),
# ])

side_motifs = OrderedDict([
    ("Oct4-Sox2", ("m0_p0", "TTTGCATAACAAAGG")),
    ("Sox2", ("m0_p1", "AGAACAATGG")),
    # ("Essrb", ("m0_p2", "TCAAGGTCA")),
    ("Nanog", ("m2_p0", "AGCCATCAA")),
    ("Klf4", ("m1_p0", "TGGGTGTGGC")),
])


profile_mapping = {
    "Oct4-Sox2": "Oct4",
    "Oct4": "Oct4",
    "Sox2": "Sox2",
    "Nanog": "Nanog",
    "Klf4": "Klf4",
    "Essrb": "Oct4"
}

peak_pred_metric = PeakPredictionProfileMetric(pos_min_threshold=0.015,
                                               neg_max_threshold=0.005,
                                               required_min_pos_counts=2.5,
                                               binsizes=[1, 10])


tf_colors = {
    "Klf4": "#357C42",
    "Oct4": "#9F1D20",
    "Sox2": "#3A3C97",
    "Nanog": "#9F8A31",
    "Esrrb": "#30BDC4"
}

# Bam files
base_dir = Path('/oak/stanford/groups/akundaje/avsec/basepair/data/processed/comparison/data/')

# chip_nexus_filt_bam = {tf: base_dir / f"filt_bam/{tf}.bam" for tf in tasks}


def data_sheet():
    import pandas as pd
    return pd.read_csv("https://docs.google.com/spreadsheets/d/1PvHGy0P9_Yq0tZFw807bjadxaZHAYECE4RytlI9rdeQ/export?gid=0&format=csv")
