import numpy as np
import pandas as pd
from scipy.stats import entropy


def fdr_threshold_norm_right(x, skip_percentile=99, fdr=0.1):
    """Get the outlier threshold (right-sided test)
    by fitting a normal distribution to the core (skipping 1% of outliers)
    estimating the p-value and using Benjamini-Hochberg FDR correction

    Args:
      x: np.array with values of interest
    """
    from statsmodels.stats.multitest import fdrcorrection
    from scipy.stats import norm

    if fdr == 1:
        return x.min()
    upper = np.percentile(x, skip_percentile)
    loc, scale = norm.fit(x[(x < upper)])
    # p-values
    p = norm.cdf(x, loc, scale)
    keep, padj = fdrcorrection(1 - p, alpha=fdr)
    return x[~keep].max()


def quantile_norm(x, norm_x, step=0.01):
    """Get the quantile values w.r.t. the other empirical distribution

    Args:
      x: values for which to compute the percentiles
      norm_x: empirical distribution w.r.t. which to compute the percentiles

    Returns:
      pd.Series of the same length as x
    """
    x = np.minimum(x, norm_x.max())  # truncate to the max range of norm_x

    # skip inf and nan for norm
    norm_x = norm_x[~np.isinf(norm_x)]
    norm_x = norm_x[~np.isnan(norm_x)]

    vc = pd.Series(np.array(norm_x)).value_counts().reset_index().sort_values("index")

    vc['rank'] = vc[0].cumsum()

    quantiles = vc['index']
    labels = vc['rank'].iloc[1:] / len(norm_x)

    # quantiles = np.sort(pd.Series(norm_x).quantile(np.arange(0, 1+step, step)).unique())

    out = pd.cut(x,
                 quantiles,
                 labels=labels,
                 include_lowest=True, right=True)
    return out


def low_medium_high(p):
    """Stratify the vector of probabilities into
    low, medium and high categories
    """
    return pd.cut(p, [0, 0.33, 0.66, 1],
                  labels=['low', 'medium', 'high'],
                  include_lowest=True, right=True)


def perc(x):
    """get the percentile values (ECDF * 100)

    >>> perc(np.arange(10))

    array([ 10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90., 100.])
    """
    from statsmodels.distributions.empirical_distribution import ECDF
    return ECDF(x)(x) * 100


def ols_formula(df, dependent_var, *excluded_cols):
    '''
    Generates the R style formula for statsmodels (patsy) given
    the dataframe, dependent variable and optional excluded columns
    as strings
    '''
    df_columns = list(df.columns.values)
    df_columns.remove(dependent_var)
    for col in excluded_cols:
        df_columns.remove(col)
    return dependent_var + ' ~ ' + ' + '.join(df_columns)


def tidy_ols(results):
    smry = results.summary()
    coef = smry.tables[1]
    df = pd.DataFrame(coef.data[1:], columns=coef.data[0])
    return df.set_index('').astype(float)

# ----------------------------------------
# different smoothing functions


def smooth_window_agg(dist, y, window=10, agg_fn=np.mean):
    xmin, xmax = dist.min(), dist.max()
    dist_vec = np.arange(xmin, xmax - window + 1, step=1)
    y_vec = np.zeros_like(dist_vec)
    for i, d in enumerate(dist_vec):
        select = (dist >= d) & (dist < d + window)
        y_vec[i] = agg_fn(y[select])
    return dist_vec, y_vec, None


def smooth_lowess(x, y, **kwargs):
    from statsmodels.nonparametric.smoothers_lowess import lowess
    a = pd.DataFrame(lowess(y, x, **kwargs)).drop_duplicates()
    return a[0], a[1], None


def smooth_gam(x, y, n_splines=100, lam=10):
    from pygam import ExpectileGAM, LinearGAM, s, f
    gam = LinearGAM(s(0, n_splines=n_splines), lam=lam).fit(x, y)
    # gam = ExpectileGAM(s(0, n_splines=n_splines), expectile=0.5, lam=lam).gridsearch(x.values.reshape((-1,1)), y)
    XX = gam.generate_X_grid(term=0)
    confi = gam.confidence_intervals(XX)
    # confi = gam.prediction_intervals(XX)
    ym = gam.predict_mu(XX)
    return XX[:, 0], ym, confi


def symmetric_kl(ref, alt):
    from scipy.stats import entropy
    return (entropy(ref, alt) + entropy(alt, ref)) / 2


# Old typo
simmetric_kl = symmetric_kl


all_metrics = ['symmetric_kl', 'kl', 'wasserstein']


def get_metric(metric):
    from scipy.stats import entropy, wasserstein_distance

    if isinstance(metric, str):
        if metric == 'simmetric_kl':
            metric = simmetric_kl
        elif metric == 'symmetric_kl':
            metric = symmetric_kl
        elif metric == 'kl':
            metric = entropy
        elif metric == 'wasserstein':
            metric = wasserstein_distance
        else:
            # Get the metric as a string
            raise NotImplemented(f"unknown metric: {metric}")
    return metric


def norm_matrix(s):
    """Create the normalization matrix

    Example:
    print(norm_matrix(pd.Series([1,3,5])).to_string())
       0  1  2
    0  1  1  1
    1  3  3  3
    2  5  5  5

    Args:
      s: pandas series
    """
    tnc = s.values[:, np.newaxis]
    vals_by_row = tnc * np.ones_like(tnc).T
    # np.fill_diagonal(vals_by_row,  1)
    return pd.DataFrame(vals_by_row, index=s.index, columns=s.index)
