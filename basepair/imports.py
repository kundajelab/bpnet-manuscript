# Common imports
# suppress warnings
import warnings
try:
    warnings.filterwarnings("ignore")
except:
    pass

# general python stuff
import os
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
import pandas as pd
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed

# plotting
import matplotlib.pyplot as plt
import seaborn as sns
from plotnine import *
import plotnine
from vdom.helpers import *  # all the html stuff
import hvplot.pandas
import holoviews as hv

# basepair plotting
import basepair
from basepair.plot.config import paper_config, get_figsize
from basepair.plot.tracks import plot_tracks, filter_tracks

# basepair / kipoi
from basepair.config import get_data_dir, create_tf_session
from basepair.modisco.results import MultipleModiscoResult, ModiscoResult, Seqlet
from basepair.modisco.score import append_pattern_loc
from basepair.modisco.utils import shorten_pattern, longer_pattern
from basepair.cli.schemas import DataSpec, TaskSpec
from basepair.cli.imp_score import ImpScoreFile
from basepair.utils import read_json, write_json, write_pkl, read_pkl
from kipoi.readers import HDF5Reader
from kipoi.writers import HDF5BatchWriter
from basepair.functions import mean
from basepair.BPNet import BPNet
from basepair.seqmodel import SeqModel
from basepair.data import NumpyDataset

# keras
from keras.models import load_model, Model
import keras.layers as kl

# saving to html
from basepair.config import get_data_dir, get_repo_root
