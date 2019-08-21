"""Scan the sequences for matches

Alternative to basepair.modisco.score


Workflow:
- get all the importance scores
  - Use either the TrackSet container from tf-Modisco or a plain dictionary of importance scores
    - which functionality do we need for this?

Two ingredients:
- tracks
- patterns

Pattern - define new class in modisco
ImportanceTracks - wraps a list of importance scores
PatternScanResults
"""

import numpy as np
from basepair.stats import fdr_threshold_norm_right
from basepair.utils import halve
from collections import OrderedDict
from pandas import pd


# TODO - merge later with Pattern?
