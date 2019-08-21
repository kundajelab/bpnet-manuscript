"""Find-instances
"""
from collections import OrderedDict, defaultdict
from copy import deepcopy
import yaml
from basepair.functions import mean
import h5py
from concise.utils.helper import write_json, read_json
from basepair.modisco.utils import bootstrap_mean, nan_like, ic_scale
from basepair.modisco.results import *
from kipoi.readers import HDF5Reader
import numpy as np
from pybedtools import BedTool, Interval
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import basepair.plot.vdom as vdom_modisco
from tqdm import tqdm


def find_instances(mr,
                   tasks,
                   contrib_scores,
                   hypothetical_contribs,
                   one_hot,
                   seqlet_len=25,
                   n_cores=1,
                   method="rank",
                   trim_pattern=False):
    """Given the importance scores and modisco results, find the motif instances

    Args:
      mr: TfModiscoResults object containing original track_set
      tasks: list of task names
      contrib_scores: dictionary of importance scores (one-hot masked)
      hypothetical_contribs: dictionary of hypothetical_scores
      one_hot: one-hot encoded DNA sequence
    """
    import modisco
    assert method in ["rank"]
    if method == "rank":
        _, _, metacluster_idx_to_scorer = get_modisco_rank_scorer(mr,
                                                                  seqlet_len,
                                                                  n_cores,
                                                                  trim_pattern=trim_pattern)
    else:
        raise ValueError("method needs to be in {'rank'}")

    # get data from seqlets
    per_position_contrib_scores = OrderedDict([(x, [np.sum(s, axis=1) for s in contrib_scores[x]])
                                               for x in tasks])

    track_set = modisco.tfmodisco_workflow.workflow.prep_track_set(
        task_names=tasks,
        contrib_scores=contrib_scores,
        hypothetical_contribs=hypothetical_contribs,
        one_hot=one_hot)

    seqlets = mr.multitask_seqlet_creation_results.multitask_seqlet_creator(
        task_name_to_score_track=per_position_contrib_scores,
        track_set=track_set,
        task_name_to_thresholding_results=mr.multitask_seqlet_creation_results.task_name_to_thresholding_results).final_seqlets

    print("get metacluster_indices")
    metacluster_indices = mr.metaclustering_results.metaclusterer.transform(seqlets).metacluster_indices

    metacluster_idx_to_seqlets = defaultdict(list)
    for a_seqlet, metacluster_idx in zip(seqlets, metacluster_indices):
        metacluster_idx_to_seqlets[metacluster_idx].append(a_seqlet)

    # score only sequences where the scorer exist
    seqlets_out = []
    print("score seqlets")
    for metacluster_idx in tqdm(metacluster_idx_to_scorer):
        score_seqlets = metacluster_idx_to_seqlets[metacluster_idx]
        pattern_results = metacluster_idx_to_scorer[metacluster_idx](score_seqlets)
        for seqlet, pattern_result in zip(score_seqlets, pattern_results):
            seqlet.metacluster = metacluster_idx
            seqlet.score_result = pattern_result
            seqlet.pattern = seqlet.score_result.pattern_idx
        seqlets_out += score_seqlets

    return seqlets_out


def labelled_seqlets2df(seqlets):
    """Convert a list of sequences to a dataframe

    Args:
      seqlets: list of seqlets returned by find_instances

    Returns:
      pandas.DataFrame with one row per seqlet
    """
    def seqlet2row(seqlet):
        """Convert a single seqlete to a pandas array
        """
        return OrderedDict([
            ("example_idx", seqlet.coor.example_idx),
            ("seqlet_start", seqlet.coor.start),
            ("seqlet_end", seqlet.coor.end),
            ("seqlet_is_revcomp", seqlet.coor.is_revcomp),
            ("seqlet_score", seqlet.coor.score),
            ("metacluster", seqlet.metacluster),
            ("pattern", seqlet.pattern),
            ("percnormed_score", seqlet.score_result.percnormed_score),
            ("score", seqlet.score_result.score),
            ("offset", seqlet.score_result.offset),
            ("revcomp", seqlet.score_result.revcomp),
        ])

    return pd.DataFrame([seqlet2row(seqlet) for seqlet in seqlets])


def get_modisco_rank_scorer(loaded_tfmodisco_results, seqlet_size_to_score_with=25, n_cores=1, trim_pattern=False):
    """

    Args:
      loaded_tfmodisco_results: tf-modisco result containing the track_seq
      seqlet_size_to_score_with: width of the scored seqlet
      n_cores: number of cores to use for nearest-neighbour computation

    Returns:
      cross_metacluster_scorer, all_pattern_names, metacluster_idx_to_scorer
    """
    import modisco
    from modisco import affinitymat
    from modisco import hit_scoring
    from modisco import aggregator
    task_names = loaded_tfmodisco_results.task_names
    metacluster_idx_to_scorer = OrderedDict()
    all_pattern_scorers = []
    all_pattern_names = []

    # loop through the metaclusters
    for metacluster_name in sorted(loaded_tfmodisco_results.metacluster_idx_to_submetacluster_results.keys()):

        submetacluster_results = (loaded_tfmodisco_results.metacluster_idx_to_submetacluster_results[metacluster_name])

        activity_pattern = submetacluster_results.activity_pattern

        relevant_task_names = [task_name for (task_name, x) in
                               zip(task_names, activity_pattern) if np.abs(x) != 0]

        if trim_pattern:
            trim_sizes = {}
            trimmed_patterns = []
            for pattern_idx, pattern in\
                enumerate(submetacluster_results.
                          seqlets_to_patterns_result.patterns):
                pssm = ic_scale(pattern["sequence"].fwd)
                t1, t2 = trim_pssm_idx(pssm)
                trim_sizes[pattern_idx] = t2 - t1
                trimmer = aggregator.TrimToBestWindow(
                    window_size=trim_sizes[pattern_idx],
                    track_names=([x + "_contrib_scores" for x in relevant_task_names]
                                 + [x + "_hypothetical_contribs" for x in relevant_task_names]))
                trimmed_patterns.extend(trimmer([pattern]))

            submetacluster_results.seqlets_to_patterns_result.patterns = trimmed_patterns

        pattern_comparison_settings = affinitymat.core.PatternComparisonSettings(
            track_names=([x + "_contrib_scores" for x in relevant_task_names] +
                         [x + "_hypothetical_contribs" for x in relevant_task_names]),  # only compare across relevant tasks
            track_transformer=affinitymat.L1Normalizer(),
            min_overlap=0.7)

        pattern_to_seqlets_sim_computer = hit_scoring.PatternsToSeqletsSimComputer(
            pattern_comparison_settings=pattern_comparison_settings,
            cross_metric_computer=affinitymat.core.ParallelCpuCrossMetricOnNNpairs(
                n_cores=n_cores,
                cross_metric_single_region=affinitymat.core.CrossContinJaccardSingleRegionWithArgmax(),
                verbose=False),
            seqlet_trimmer=modisco.hit_scoring.SeqletTrimToBestWindow(
                window_size=seqlet_size_to_score_with,
                track_names=[x + "_contrib_scores" for x
                             in relevant_task_names])
        )

        # Get a list of scorers for all the patterns in the metacluster
        metacluster_pattern_scorers = []
        if submetacluster_results.seqlets_to_patterns_result.patterns is None or \
           len(submetacluster_results.seqlets_to_patterns_result.patterns) == 0:
            # metacluster has no patterns
            # don't append anything
            continue

        for pattern_idx, pattern in\
            enumerate(submetacluster_results.
                      seqlets_to_patterns_result.patterns):
            metacluster_idx = int(metacluster_name.split("_")[1])
            all_pattern_names.append("metacluster_" + str(metacluster_idx)
                                     + ",pattern_" + str(pattern_idx))
            if trim_pattern:
                pattern_to_seqlets_sim_computer = hit_scoring.PatternsToSeqletsSimComputer(
                    pattern_comparison_settings=pattern_comparison_settings,
                    cross_metric_computer=affinitymat.core.ParallelCpuCrossMetricOnNNpairs(
                        n_cores=n_cores,
                        cross_metric_single_region=affinitymat.core.CrossContinJaccardSingleRegionWithArgmax(),
                        verbose=False),
                    seqlet_trimmer=modisco.hit_scoring.SeqletTrimToBestWindow(
                        window_size=min(seqlet_size_to_score_with, trim_sizes[pattern_idx]),
                        track_names=[x + "_contrib_scores" for x
                                     in relevant_task_names])
                )
            pattern_scorer = hit_scoring.RankBasedPatternScorer(
                aggseqlets=pattern,
                patterns_to_seqlets_sim_computer=pattern_to_seqlets_sim_computer)
            metacluster_pattern_scorers.append(pattern_scorer)
            all_pattern_scorers.append(pattern_scorer)
        # This is the final scorer for the metacluster;
        # it takes the maximum score produced by all the
        # individual scorers
        max_rank_based_pattern_scorer = hit_scoring.MaxRankBasedPatternScorer(pattern_scorers=metacluster_pattern_scorers)

        metacluster_idx_to_scorer[metacluster_idx] = max_rank_based_pattern_scorer

    cross_metacluster_scorer = hit_scoring.MaxRankBasedPatternScorer(pattern_scorers=all_pattern_scorers)

    return cross_metacluster_scorer, all_pattern_names, metacluster_idx_to_scorer


# loading the instance locations

def append_pattern_loc(df, pattern_pssms, trim_frac=0.08):
    """Figure out the pattern location

    Args:
      df: pandas.DataFrame containing (pattern_id, seqlet_start, seqlet_end, offset, revcomp)
      pattern_pssm (dict): Key = pattern_id, value: pssm matricex (obtained by mr.get_pssm()). 
      trim_fact: how much to trim the pssm of the pattern using `trim_pssm_idx()` function

    Side effect: appends `pattern_start` and `pattern_end` to `df`

    Returns:
      None
    """
    d_trim_i = {}
    d_trim_j = {}
    d_pattern_width = {}
    for pattern in pattern_pssms:
        pssm = pattern_pssms[pattern]
        trim_i, trim_j = trim_pssm_idx(pssm, 0.08)
        d_trim_i[pattern] = trim_i
        d_trim_j[pattern] = trim_j
        d_pattern_width[pattern] = pssm.shape[0]
        del trim_i, trim_j

    # prepare the required vectors
    trim_i_vec = df.pattern_id.map(d_trim_i)
    trim_j_vec = df.pattern_id.map(d_trim_j)
    pattern_width_vec = df.pattern_id.map(d_pattern_width)
    rc_vec = df.revcomp == 1
    rc_offset = df.revcomp  # shift by one if revcomp

    # setup the reverse complement shift
    trim_i_rc = pattern_width_vec - trim_j_vec - 1
    trim_j_rc = pattern_width_vec - trim_i_vec - 1
    trim_i_vec = np.where(rc_vec, trim_i_rc, trim_i_vec)
    trim_j_vec = np.where(rc_vec, trim_j_rc, trim_j_vec)

    # where revcomp
    pattern_start = df.seqlet_start - df.offset + rc_offset
    df['pattern_start'] = (pattern_start + trim_i_vec).astype(int)
    df['pattern_end'] = (pattern_start + trim_j_vec).astype(int)
