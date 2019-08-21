import pandas as pd
import numpy as np


def dfint_overlap_idx(dfa, dfb):
    """Overlap dfa with dfb

    Returns:
      np.array with length `len(dfa)` of the matching row indices in dfab

    Note:
      if multiple rows in dfab overlap a row in dfa,
      then the first mathing row in dfb is returned
    """
    from pybedtools import BedTool
    assert len(dfa.columns) == 3
    assert len(dfb.columns) == 3
    dfa = dfa.copy()
    dfa['id'] = np.arange(len(dfa))
    dfb = dfb.copy()
    dfb['id'] = np.arange(len(dfb))
    bta = BedTool.from_dataframe(dfa)
    btb = BedTool.from_dataframe(dfb)
    dfi = bta.intersect(btb, wa=True, loj=True).to_dataframe()
    keep = ~dfi[['chrom', 'start', 'end', 'name']].duplicated()
    out = dfi[keep].iloc[:, -1]  # final column
    out[out == '.'] = '-1'
    return out.astype(int).values


def dfi_append_example_idx(dfa, ranges):
    """
    Args:
      dfa [pd.DataFrame]: containes columns: chrom, pattern_center_abs
      ranges [pd.DataFrame]: containes columns: `example_chrom`, `example_start`, `example_end`

    Note: This assumes that dfa will contain the following 
    """
    dfa = dfa.copy()
    example_idx = dfint_overlap_idx(dfa[['chrom',
                                         'pattern_center_abs',
                                         'pattern_center_abs']],
                                    ranges[['example_chrom',
                                            'example_start',
                                            'example_end']])

    dfa['example_idx'] = example_idx
    dfa = dfa[dfa['example_idx'] != -1]
    dfa = pd.merge(dfa, ranges, on='example_idx', how='left')

    # Add the pattern center
    dfa['pattern_center'] = (dfa['pattern_center_abs'] - dfa['example_start']).astype(int)
    return dfa
