"""Code for analyzing motif spacing
"""
# Code for motif combinations
import matplotlib.pyplot as plt
# setup config
from basepair.modisco.pattern_instances import construct_motif_pairs
import numpy as np
import pandas as pd


def get_motif_pairs(motifs):
    """Generate motif pairs
    """
    pairs = []
    for i in range(len(motifs)):
        for j in range(i, len(motifs)):
            pairs.append([list(motifs)[i], list(motifs)[j], ])
    return pairs


comp_strand_compbination = {
    "++": "--",
    "--": "++",
    "-+": "+-",
    "+-": "-+"
}

strand_combinations = ["++", "--", "+-", "-+"]


def motif_pair_dfi(dfi_filtered, motif_pair):
    """Construct the matrix of motif pairs

    Args:
      dfi_filtered: dfi filtered to the desired property
      motif_pair: tuple of two pattern_name's
    Returns:
      pd.DataFrame with columns from dfi_filtered with _x and _y suffix
    """
    dfa = dfi_filtered[dfi_filtered.pattern_name == motif_pair[0]]
    dfb = dfi_filtered[dfi_filtered.pattern_name == motif_pair[1]]

    dfab = pd.merge(dfa, dfb, on='example_idx', how='outer')
    dfab = dfab[~dfab[['pattern_center_x', 'pattern_center_y']].isnull().any(1)]

    dfab['center_diff'] = dfab.pattern_center_y - dfab.pattern_center_x
    if "pattern_center_aln_x" in dfab:
        dfab['center_diff_aln'] = dfab.pattern_center_aln_y - dfab.pattern_center_aln_x
    dfab['strand_combination'] = dfab.strand_x + dfab.strand_y
    # assure the right strand combination
    dfab.loc[dfab.center_diff < 0, 'strand_combination'] = dfab[dfab.center_diff < 0]['strand_combination'].map(comp_strand_compbination).values

    if motif_pair[0] == motif_pair[1]:
        dfab.loc[dfab['strand_combination'] == "--", 'strand_combination'] = "++"
        dfab = dfab[dfab.center_diff > 0]
    else:
        dfab.center_diff = np.abs(dfab.center_diff)
        if "center_diff_aln" in dfab:
            dfab.center_diff_aln = np.abs(dfab.center_diff_aln)
    if "center_diff_aln" in dfab:
        dfab = dfab[dfab.center_diff_aln != 0]  # exclude perfect matches
    return dfab


def remove_edge_instances(dfab, profile_width=70, total_width=1000):
    half = profile_width // 2 + profile_width % 2
    return dfab[(dfab.pattern_center_x - half > 0) & (dfab.pattern_center_x + half < total_width) &
                (dfab.pattern_center_y - half > 0) & (dfab.pattern_center_y + half < total_width)]


def plot_spacing(dfab,
                 alpha_scatter=0.01,
                 y_feature='profile_counts',
                 center_diff_variable='center_diff',
                 figsize=(3.42519, 6.85038)):
    from basepair.exp.paper.config import profile_mapping
    from basepair.stats import smooth_window_agg, smooth_lowess, smooth_gam

    motif_pair = (dfab.iloc[0].pattern_name_x, dfab.iloc[0].pattern_name_y)
    strand_combinations = dfab.strand_combination.unique()
    fig_profile, axes = plt.subplots(2 * len(strand_combinations), 1, figsize=figsize, sharex=True, sharey='row')

    motif_pair_c = motif_pair
    axes[0].set_title("<>".join(motif_pair), fontsize=7)

    j = 0  # first column

    dftw_filt = dfab[(dfab.center_diff < 150)]  # & (dfab.imp_weighted_p.max(1) > 0.3)]
    for i, sc in enumerate(strand_combinations):
        if y_feature == 'profile_counts':
            y1 = np.log10(1 + dftw_filt[dftw_filt.strand_combination == sc][profile_mapping[motif_pair_c[0]] + "/profile_counts_x"])
            y2 = np.log10(1 + dftw_filt[dftw_filt.strand_combination == sc][profile_mapping[motif_pair_c[1]] + "/profile_counts_y"])
        elif y_feature == 'imp_weighted':
            y1 = np.log10(1 + dftw_filt[dftw_filt.strand_combination == sc]['imp_weighted_x'])
            y2 = np.log10(1 + dftw_filt[dftw_filt.strand_combination == sc]['imp_weighted_y'])
        else:
            raise ValueError(f"Unkown y_feature: {y_feature}")

        # y1 = dftw_filt[dftw_filt.strand_combination==sc]['imp_weighted'][motif_pair[0]]
        # y2 = dftw_filt[dftw_filt.strand_combination==sc]['imp_weighted'][motif_pair[1]]
        x = dftw_filt[dftw_filt.strand_combination == sc][center_diff_variable]

        # dm,ym,confi = average_distance(x,y, window=5)
        dm1, ym1, confi1 = smooth_lowess(x, y1, frac=0.15)
        dm2, ym2, confi2 = smooth_lowess(x, y2, frac=0.15)
        # dm,ym, confi = smooth_gam(x,y, 140, 20)

        ax = axes[2 * i]
        ax.hist(dftw_filt[dftw_filt.strand_combination == sc][center_diff_variable], np.arange(10, 150, 1))
        if j == 0:
            ax.set_ylabel(sc)

        # second plot
        ax.set_xlim([0, 150])
        ax = axes[2 * i + 1]
        ax.scatter(x, y1, alpha=alpha_scatter, s=8)
        if confi1 is not None:
            ax.fill_between(dm1, confi1[:, 0], confi1[:, 1], alpha=0.2)
        ax.plot(dm1, ym1, linewidth=2, alpha=0.8)
        ax.scatter(x, y2, alpha=alpha_scatter, s=8)
        if confi2 is not None:
            ax.fill_between(dm2, confi2[:, 0], confi2[:, 1], alpha=0.2)
        ax.plot(dm2, ym2, linewidth=2, alpha=0.8)
        if j == 0:
            ax.set_ylabel(sc)
        ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
        ax.xaxis.set_major_locator(plt.MultipleLocator(20))
        if j == 0:
            ax.set_ylabel(sc)
        if i == len(strand_combinations) - 1:
            ax.set_xlabel("Distance between motifs")
    fig_profile.subplots_adjust(wspace=0, hspace=0)
    return fig_profile


# --------------------------------------------
# Co-occurence


def co_occurence_matrix(dfi_subset, query_string=""):
    """Returns the fraction of times pattern x (row) overlaps pattern y (column)
    """
    from basepair.stats import norm_matrix
    total_number = dfi_subset.groupby(['pattern']).size()
    norm_counts = norm_matrix(total_number)

    # normalization: minimum number of counts
    total_number = dfi_subset.groupby(['pattern_name']).size()
    norm_counts = norm_matrix(total_number)

    # cross-product
    dfi_filt_crossp = pd.merge(dfi_subset[['pattern_name', 'pattern_center_aln',
                                           'pattern_strand_aln', 'pattern_center', 'example_idx']].set_index('example_idx'),
                               dfi_subset[['pattern_name', 'pattern_center_aln',
                                           'pattern_strand_aln', 'pattern_center', 'example_idx']].set_index('example_idx'),
                               how='outer', left_index=True, right_index=True).reset_index()
    # remove self-matches
    dfi_filt_crossp = dfi_filt_crossp.query('~((pattern_name_x == pattern_name_y) & '
                                            '(pattern_center_aln_x == pattern_center_aln_y) & '
                                            '(pattern_strand_aln_x == pattern_strand_aln_x))')
    dfi_filt_crossp['center_diff'] = dfi_filt_crossp.eval("abs(pattern_center_x- pattern_center_y)")
    dfi_filt_crossp['center_diff_aln'] = dfi_filt_crossp.eval("abs(pattern_center_aln_x- pattern_center_aln_y)")

    if query_string:
        dfi_filt_crossp = dfi_filt_crossp.query(query_string)
    match_sizes = dfi_filt_crossp.groupby(['pattern_name_x', 'pattern_name_y']).size()
    count_matrix = match_sizes.unstack(fill_value=0)

    norm_count_matrix = count_matrix / norm_counts  # .truediv(min_counts, axis='columns').truediv(total_number, axis='index')
    norm_count_matrix = norm_count_matrix.fillna(0)  # these examples didn't have any paired pattern

    return count_matrix, norm_count_matrix, norm_counts


def chi2_test_coc(random_coocurrence_counts, random_coocurrence,
                  random_coocurrence_norm, coocurrence_counts, coocurrence, coocurrence_norm):
    from scipy.stats import chi2_contingency
    cols = list(coocurrence_norm.columns)
    n = len(random_coocurrence_counts)
    o = np.zeros((n, n))
    op = np.zeros((n, n))
    for i in range(n):
        for j in range(n):

            # [[# not randomly found together , # not found together],
            #  [# randomly found together     , # found together]]
            ct = [[random_coocurrence_norm.iloc[i, j] - random_coocurrence_counts.iloc[i, j],
                   coocurrence_norm.iloc[i, j] - coocurrence_counts.iloc[i, j]],
                  [random_coocurrence_counts.iloc[i, j],
                   coocurrence_counts.iloc[i, j]]]
            ct = np.array(ct)
            # TODO - make this an actual fisher exact test from
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html
            # or the chi-square contingency table:
            # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html#scipy.stats.chi2_contingency
            chi2, p, dof, ex = chi2_contingency(ct, correction=False)
            # t22 = sm.stats.contingency_tables.Table2x2(np.array(ct))

            o[i, j] = (ct[1, 1] / ct[0, 1]) / (ct[1, 0] / ct[0, 0])
            op[i, j] = p
    return pd.DataFrame(o, columns=cols, index=cols), pd.DataFrame(op, columns=cols, index=cols)


def coocurrence_plot(dfi_subset, motif_list, query_string="(abs(pattern_center_aln_x- pattern_center_aln_y) <= 150)",
                     signif_threshold=1e-5, ax=None, **kwargs):
    """Test for co-occurence

    Args:
      dfi_subset: desired subset of dfi
      motif_list: list of motifs used to order the heatmap
      query_string: string used with df_cross.query() to detering the valid motif pairs
      signif_threshold: significance threshold for Fisher's exact test
    """
    import seaborn as sns
    if ax is None:
        ax = plt.gca()
    c_counts, c, c_norm = co_occurence_matrix(dfi_subset, query_string=query_string)

    # Generate the NULL
    dfi_subset_random = dfi_subset.copy()
    np.random.seed(42)
    dfi_subset_random['example_idx'] = dfi_subset_random['example_idx'].sample(frac=1).values
    rc_counts, rc, rc_norm = co_occurence_matrix(dfi_subset_random, query_string=query_string)

    # test for significance
    o, op = chi2_test_coc(rc_counts, rc, rc_norm, c_counts, c, c_norm)

    # re-order
    o = o[motif_list].loc[motif_list]
    op = op[motif_list].loc[motif_list]

    signif = op < signif_threshold
    a = np.zeros_like(signif).astype(str)
    a[signif] = "*"
    a[~signif] = ""

    sns.heatmap(o, annot=a, fmt="", vmin=0, vmax=2,
                cmap='RdBu_r', ax=ax, **kwargs)
    ax.set_title(f"odds-ratio (proximal / non-proximal) (*: p<{signif_threshold})")
