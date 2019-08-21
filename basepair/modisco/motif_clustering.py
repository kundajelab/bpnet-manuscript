import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale, StandardScaler

NUMERIC_DTYPES = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def rename(df, task):
    """Subset all columns starting with `task`
    and remove `task` from it
    """
    dfs = df.iloc[:, df.columns.str.match(task)]
    dfs.columns = dfs.columns.str.replace(task + " ", "")
    dfs['task'] = task
    return dfs


def scale(df):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(df.values), columns=df.columns, index=df.index)


def pd_scale(s):
    return scale(pd.DataFrame(dict(a=s)))['a']


def pd_minmax_scale(s):
    """minmax scale pandas series (preserves the index)
    """
    return pd.Series(minmax_scale(np.array(s.values)), index=s.index)


def df_minmax_scale(df):
    """Min-max scale all columns in a pandas.DataFrame
    """
    df = df.copy()
    for c in df.columns:
        if str(df[c].dtype) in NUMERIC_DTYPES:
            df[c] = pd_minmax_scale(df[c])
    return df


def log_cols(df, tasks):
    for task in tasks:
        df[f'{task} imp counts'] = np.log(1 + df[f'{task} imp counts'])
        df[f'{task} imp profile'] = np.log(1 + df[f'{task} imp profile'])
        df[f'{task} footprint max'] = np.log(1 + df[f'{task} footprint max'])
        df[f'{task} region counts'] = np.log(1 + df[f'{task} region counts'])
    return df


def to_colors(df, cat_cmap='RdBu_r', num_cmap='Reds'):
    """Convert pandas data-table to colors. Can be directly used with
    seaborn.clustermap(row_colors= ...)

    Args:
      df: pandas.DataFrame
      cmap: Seaborn color pallete to use. Example: RdBu_r, Reds

    Returns:
      pandas data frame with the same columns names but
      values converted to colors
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    cols = {}
    for c in df.columns:
        if df[c].dtype.name in ['object', 'category']:
            categories = list(df[c].unique())
            cmap = sns.color_palette(cat_cmap, len(categories))
            cmap_dict = dict(zip(categories, cmap))
            cols[c] = df[c].map(cmap_dict)
        else:
            cols[c] = df[c].map(plt.cm.get_cmap(num_cmap))
    return pd.DataFrame(cols)


def append_features(df):
    df = df.copy()
    df['metacluster'] = df.pattern.str.split("_", expand=True)[0].str.replace("m", "").astype(int)
    df['metacluster'] = pd.Categorical(df.metacluster, ordered=True)
    df['log n seqlets'] = np.log10(df['n seqlets'])
    df['motif len'] = df.consensus.str.len()
    df['motif ic'] = df['motif len'] * df['ic pwm mean']  # add also motif ic
    return df


def preproc_motif_table(df, tasks, unclust_feat=['log n seqlets'], drop_columns=[]):
    """Preprocess motif table
    """
    df = df.copy()
    # annotate row / columns indexes
    df = df.set_index('pattern')
    df.columns.name = 'Feature'

    # setup the final matrix for the heatmap - dfx
    # drop some columns
    drop = ['idx', 'consensus', 'metacluster', 'n seqlets', 'index'] + unclust_feat + drop_columns
    drop += [c for c in df.columns if df[c].dtype.name not in ['int64', 'float64']]
    drop += [f"{t} pos unimodal" for t in tasks]
    dfx = df[[c for c in df.columns if c not in drop]]

    # np.log(1+x) some of the columns
    dfx = log_cols(dfx, tasks)

    # merge '{task} pos std' columns into one by averaging
    pos_std = list(dfx.columns[dfx.columns.str.contains('pos std')])
    dfx['pos std'] = dfx[pos_std].mean(axis=1)
    for p in pos_std:
        del dfx[p]

    # remove '{task} pos meandiff'
    for c in list(dfx.columns[dfx.columns.str.contains('pos meandiff')]):
        del dfx[c]

    # setup row and columns columns (annotation)
    row_df = df_minmax_scale(df[unclust_feat])  # TODO - make it general in case the columns are categories

    # column
    col_df = pd.DataFrame(dict(task=pd.Series(dfx.columns, index=dfx.columns).str.split(" ", n=1, expand=True)[0],
                               feature=pd.Series(dfx.columns, index=dfx.columns).str.split(" ", n=1, expand=True)[1]
                               ))
    return dfx, row_df, col_df


def preproc_df(df, min_n_seqlets=100):
    df['metacluster'] = df.pattern.str.split("_", expand=True)[0].str.replace("m", "").astype(int)
    df['metacluster'] = pd.Categorical(df.metacluster, ordered=True)
    df['log n seqlets'] = np.log10(df['n seqlets'])

    df['motif len'] = df.consensus.str.len()
    df['motif ic'] = df['motif len'] * df['ic pwm mean']  # add also motif ic

    if 'cluster' in df:
        df['cluster'] = pd.Categorical(df['cluster'])

    # filter
    if min_n_seqlets is not None:
        df = df[df['n seqlets'] >= min_n_seqlets]
    return df


def motif_table_long(df, tasks, index_cols=[]):
    """Convert the motif table into long format
    """
    drop = ['idx', 'consensus', 'n seqlets', 'metacluster']

    dfi = df.set_index(drop + ['pattern', 'motif len', 'ic pwm mean'] + index_cols)
    dfl = pd.concat([rename(dfi, task) for task in tasks]).reset_index()

    dfl['log n seqlets'] = np.log10(dfl['n seqlets'])

    # omit some columns

    dfl['imp counts'] = np.log(1 + dfl['imp counts'])
    dfl['imp profile'] = np.log(1 + dfl['imp profile'])
    dfl['footprint max'] = np.log(dfl['footprint max'])
    dfl['region counts'] = np.log(1 + dfl['region counts'])
    dfl = dfl.reset_index()
    del dfl['index']
    return dfl


def hirearchically_reorder_table(df, tasks):
    """Re-orders table using hirearchical clustering

    Args:
      df: pd.DataFrame returned by basepair.modisco.table.modisco_table
      tasks: list of tasks
    """
    from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list

    dfx, row_df, col_df = preproc_motif_table(append_features(df), tasks)
    x = scale(dfx)

    assert np.all(dfx.index == df.pattern)

    # cluster
    x_imp = x.fillna(x.mean()).values  # fill missing values with mean values
    row_linkage = linkage(x_imp, method='weighted', optimal_ordering=True)
    col_linkage = linkage(x_imp.T, method='weighted', optimal_ordering=True)

    # get the indices
    rows_idx = list(leaves_list(row_linkage))
    col_idx = list(dfx.columns[leaves_list(col_linkage)])

    # pos std
    columns = [c for c in df.columns
               if c not in dfx.columns] + [c for c in col_idx
                                           if c in df.columns]

    df_out = df.iloc[rows_idx][columns]
    df_out.doc = df.doc
    return df_out



# --------------------------------------------------
from kipoi.utils import unique_list
from joblib import Parallel, delayed
from basepair.utils import dict_suffix_key
from basepair.modisco.table import pattern_url
from collections import OrderedDict
from tqdm import tqdm
from basepair.plot.vdom import df2html, df2html_old, render_datatable, vdom_footprint
from basepair.stats import perc


def get_major_patterns(patterns, cluster):
    """Get major patterns for each cluster

    Args:
      patterns: Pattern list
      cluster: np.array with cluster names for each pattern

    Returns: 
      list: (cluster, pattern)
    """
    try:
        df = pd.DataFrame({"n_seqlets": [p.attrs['features']['n seqlets'] for p in patterns],
                           "cluster": cluster})
    except:
        print("Number of seqlets not provided, using random number of seqlets")
        df = pd.DataFrame({"n_seqlets": np.random.randn(len(cluster)),
                           "cluster": cluster})
    return OrderedDict([(k, patterns[v]) for k, v in df.groupby('cluster').n_seqlets.idxmax().items()])


def align_patterns(patterns, order, max_shift=30, metric='continousjaccard', align_track='contrib/mean', new_align_strategy='prev'):
    """Align multiple patterns. Alignment will procede one after the other
    """
    out = []
    prev = None
    total_sim = 0
    for i in order:
        p = patterns[i]
        if prev is None:
            # keep the first pattern as is
            p_aligned = p
        else:
            # use the previous pattern as the template
            if new_align_strategy == 'prev':
                total_sim += p.similarity(prev, track=align_track, metric=metric) 
                p_aligned = p.align(prev, track=align_track, max_shift=max_shift, metric=metric)
            elif new_align_strategy == 'best_prev':
                similarities = [p.similarity(p_prev, track=align_track, metric=metric)
                                for p_prev in out]
                align_to = np.argmax(similarities)
                total_sim += similarities[align_to]
                print(f"{patterns[i].name} aligned to {out[align_to].name}")
                p_aligned = p.align(out[align_to], track=align_track, 
                                    metric=metric,
                                    max_shift=max_shift)
            elif new_align_strategy == 'all_prev':
                total_sim += p.similarity(out, track=align_track, 
                                          metric=metric) 
                p_aligned = p.align(out, track=align_track, 
                                    metric=metric,
                                    max_shift=max_shift)
            else:
                raise ValueError(f"invalid new_align_strategy={new_align_strategy}. Use 'prev' or 'all_prev'")

        out.append(p_aligned)
        prev = p_aligned
    return OrderedDict(zip(order, out)), total_sim


def optimal_align_patterns(patterns, order, trials=10, max_shift=30, 
                           metric='continousjaccard', align_track='contrib/mean'):
    from basepair.utils import shuffle_list
    
    alignments = [align_patterns(patterns,
                                shuffle_list(order),
                                max_shift=max_shift,
                                metric=metric,
                                new_align_strategy="all_prev",
                                align_track=align_track)
                  for i in range(trials)]
    best_idx = np.argmax(list(zip(*alignments))[1])
    major_patterns, sim = alignments[best_idx]
    return OrderedDict([(k, major_patterns[k]) for k in order])


def align_clustered_patterns(patterns, cluster_order, cluster,
                             align_track='contrib/mean',
                             metric='continousjaccard', 
                             max_shift=30,
                             trials=10):
    """Align clustered patterns

    In addition to normal features under p.attrs['features'] it adds
    logo_imp, logo_seq, profile scores and the directness score

    Args:
      patterns: list of patterns
      cluster_order: order rank for each pattern id. e.g. cluster_order[1] = 5
        means that patterns[1] needs to be at position 5 at the end
      cluster: cluster identity for each pattern
      align_track: which track to use for alignment
      max_shift: maximum allowed shift of the motif
      trials: how many times to run the alignment algorithm with different ordering
      
    Returns: list of motifs of the same length as patterns. They are ordered
      as they should be plotted in the table. Also 'cluster' field is added
    """
    # Order of the clusters
    cluster_order_ind = unique_list(cluster[np.argsort(cluster_order)])

    # 1. Get major patterns and align them
    major_patterns = optimal_align_patterns(get_major_patterns(patterns, cluster),
                                            cluster_order_ind,
                                            max_shift=max_shift,
                                            metric=metric,
                                            trials=trials,
                                            align_track=align_track)    
    
    # align patterns to major patterns in the cluster and add the cluster/group information
    return [patterns[i].align(major_patterns[cluster[i]], 
                              metric=metric, 
                              track=align_track).add_attr('cluster', cluster[i])
            for i in tqdm(np.argsort(cluster_order))]



def cluster_patterns(patterns, n_clusters=9, cluster_track='seq_ic'):
    """Cluster patterns
    """
    # Whole pipeline from this notebook
    sim = similarity_matrix(patterns, track=cluster_track)
    
    # cluster
    lm_nte_seq = linkage(1-sim_nte_seq[iu1], 'ward', optimal_ordering=True)
    cluster = cut_tree(lm_nte_seq, n_clusters=n_clusters)[:,0]
    
    cluster_order = np.argsort(leaves_list(lm_nte_seq))
    pattern_table_nte_seq = create_pattern_table(patterns_nte, cluster_order, cluster, 
                                                 align_track='contrib/mean',
                                                 logo_len=70, 
                                                 seqlogo_kwargs=dict(width=320), 
                                                 footprint_width=320,
                                                 footprint_kwargs=dict())
    return sim, lm_nte_seq, cluster, cluster_order, pattern_table_nte_seq

        
    
def create_pattern_table(patterns,
                         logo_len=30,
                         footprint_width=320,
                         footprint_kwargs=None,
                         seqlogo_kwargs=None,
                         # report_url='results.html'
                         n_jobs=10):
    """Creates the pattern table given a list of patterns

    In addition to normal features under p.attrs['features'] it adds
    logo_imp, logo_seq, profile scores and the directness score

    Args:
      patterns: list of patterns with 'profile' and attrs['features'] and attrs['motif_freq'] features
      cluster_order: n

    """
    # get profile medians for each task across all patterns
    tasks = patterns[0].tasks()
    profile_max_median = {t: np.median([p.profile[t].max() for p in patterns])
                          for t in tasks}

    def extract_info(pattern_i):
        """Function ran for each cluster order
        """
        p = patterns[pattern_i]
        # setup the dictionary
        d = p.attrs['features']
        # Add motif frequencies
        d = {**d, 
             **dict_suffix_key(p.attrs.get('motif_odds', dict()), '/odds'), 
             **dict_suffix_key(p.attrs.get('motif_odds_p', dict()), '/odds_p')}
        # d['pattern'] = pattern_url(d['pattern'], report_url)
        d['cluster'] = p.attrs['cluster']

        # Add seqlogos
        d['logo_imp'] = p.resize(logo_len).vdom_plot('contrib', as_html=True, **seqlogo_kwargs)
        d['logo_seq'] = p.resize(logo_len).vdom_plot('seq', as_html=True, **seqlogo_kwargs)

        for t in p.tasks():
            # add profile sparklines
            d[t + '/f'] = (vdom_footprint(p.profile[t],
                                          r_height=profile_max_median[t],
                                          text=None,
                                          **footprint_kwargs)
                           .to_html()
                           .replace("<img", f"<img width={footprint_width}")
                           )

            # Add directness score
            d[t + "/d"] = 2 * d[t + " imp profile"] - d[t + " imp counts"]
        return d

    l = Parallel(n_jobs=n_jobs)(delayed(extract_info)(i) for i in tqdm(range(len(patterns))))
    df = pd.DataFrame(l)

    # add normalized features
    for t in tasks:
        df[t + '/d_p'] = perc(df[t + '/d'])
        
    # use float for n seqlets
    df['n seqlets'] = df['n seqlets'].astype(float)
    return df
