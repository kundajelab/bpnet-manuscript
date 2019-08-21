import numpy as np
from tqdm import tqdm
import pandas as pd
from basepair.modisco.utils import shorten_pattern


def similarity_matrix(patterns, **kwargs):
    """Compute pattern-pattern similarity matrix
    """
    m = np.zeros((len(patterns), len(patterns)))
    for i in tqdm(range(len(patterns))):
        for j in range(len(patterns)):
            m[i, j] = max(patterns[i].similarity(patterns[j], **kwargs),
                          patterns[i].rc().similarity(patterns[j], **kwargs))
    return (m + m.T) / 2


def append_logo_cluster(pattern_table, patterns, cluster_order, cluster,
                        align_track='contrib/mean', logo_len=30, **kwargs):
    # setup patterns
    pattern_names = np.array([shorten_pattern(p.name) for p in patterns])
    patterns_nte_dict = {shorten_pattern(p.name): p for p in patterns}  # organize as a dict

    pattern_table = pattern_table.set_index('pattern')
    pattern_table_nte = pattern_table.loc[pattern_names]
    pattern_table = pattern_table.reset_index()
    pattern_table_nte['cluster'] = cluster
    pattern_table_nte['cluster_order'] = cluster_order

    # pattern_table_nte = pattern_table_nte.iloc[cluster_order]  # sort the whole table
    out = []
    for cluster_id in tqdm(pattern_table_nte.cluster.unique()):
        dfg = pattern_table_nte[pattern_table_nte.cluster == cluster_id]

        # identify the major pattern
        max_seqlets = dfg['n seqlets'].argmax()
        major_pattern = patterns_nte_dict[max_seqlets]

        # align w.r.t. the thing used for clustering
        logo_imp = [patterns_nte_dict[p].align(major_pattern, track=align_track).resize(logo_len).vdom_plot('contrib', as_html=True, **kwargs)
                    for p in dfg.index]
        logo_seq = [patterns_nte_dict[p].align(major_pattern, track=align_track).resize(logo_len).vdom_plot('seq', as_html=True, **kwargs)
                    for p in dfg.index]

        dfg['logo_imp'] = logo_imp
        dfg['logo_seq'] = logo_seq
        out.append(dfg)

    return pd.concat(out, axis=0).reset_index()
