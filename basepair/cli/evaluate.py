"""
Evaluate the model

1. Count predition
  - pearson correlation
  - spearman correlation
  - scatterplot
2. Profile prediction
  - log-likelihood
  - auPRC for the top counts
"""
import numpy as np
import pandas as pd
import matplotlib
from tqdm import tqdm
from basepair.functions import softmax
import json
import functools
import os
from concise.eval_metrics import auprc
from concise.utils.helper import write_json
from basepair.cli.schemas import DataSpec, HParams
from basepair.datasets import chip_exo_nexus
from basepair.config import create_tf_session
from basepair.utils import _listify
from keras.models import load_model
from basepair.utils import read_pkl
import matplotlib.pyplot as plt
import logging
from scipy.stats import pearsonr, spearmanr
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
# matplotlib.use('Agg')


def load_data(model_dir, cache_data=False,
              dataspec=None, hparams=None, data=None, preprocessor=None):
    if dataspec is not None:
        ds = DataSpec.load(dataspec)
    else:
        ds = DataSpec.load(os.path.join(model_dir, "dataspec.yaml"))
    if hparams is not None:
        hp = HParams.load(hparams)
    else:
        hp = HParams.load(os.path.join(model_dir, "hparams.yaml"))
    if data is not None:
        data_path = data
    else:
        data_path = os.path.join(model_dir, 'data.pkl')
    if os.path.exists(data_path):
        train, valid, test = read_pkl(data_path)
    else:
        train, valid, test = chip_exo_nexus(ds,
                                            peak_width=hp.data.peak_width,
                                            shuffle=hp.data.shuffle,
                                            valid_chr=hp.data.valid_chr,
                                            test_chr=hp.data.test_chr)

        # Pre-process the data
        logger.info("Pre-processing the data")
        if preprocessor is not None:
            preproc_path = os.path.join(model_dir, "preprocessor.pkl")
        else:
            preproc_path = preprocessor
        preproc = read_pkl(preproc_path)
        train[1] = preproc.transform(train[1])
        valid[1] = preproc.transform(valid[1])
        try:
            test[1] = preproc.transform(test[1])
        except Exception:
            logger.warn("Test set couldn't be processed")
            test = None
    return train, valid, test



def bin_counts_amb(x, binsize=2):
    """Bin the counts
    """
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    # maximum used instead of np.any.
    has_peak = np.maximum.reduceat(x == 1, np.arange(x.shape[1], step=binsize), axis=1)
    has_amb = np.maximum.reduceat(x == -1, np.arange(x.shape[1], step=binsize), axis=1)
    return (has_peak - (1 - has_peak) * has_amb).astype(float)


def bin_counts_summary(x, binsize=2, fn=np.max):
    """Bin the counts
    """
    ufunc = {np.max: np.maximum,
             np.sum: np.add,
             np.mean: np.add,
            }[fn]
    if binsize == 1:
        return x
    assert len(x.shape) == 3
    outlen = x.shape[1] // binsize
    output = ufunc.reduceat(x, np.arange(x.shape[1], step=binsize), axis=1)
    if fn == np.mean:
        return output / binsize
    else:
        return output

bin_counts_max = functools.partial(bin_counts_summary, fn=np.max)


def permute_array(arr, axis=0):
    """Permute array along a certain axis

    Args:
      arr: numpy array
      axis: axis along which to permute the array
    """
    if axis == 0:
        return np.random.permutation(arr)
    else:
        return np.random.permutation(arr.swapaxes(0, axis)).swapaxes(0, axis)


def eval_profile(yt, yp,
                 pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=2.5,
                 binsizes=[1, 2, 4, 10]):
    """
    Evaluate the profile in terms of auPR

    Args:
      yt: true profile (counts)
      yp: predicted profile (fractions)
      pos_min_threshold: fraction threshold above which the position is
         considered to be a positive
      neg_max_threshold: fraction threshold bellow which the position is
         considered to be a negative
      required_min_pos_counts: smallest number of reads the peak should be
         supported by. All regions where 0.05 of the total reads would be
         less than required_min_pos_counts are excluded
    """
    # The filtering
    # criterion assures that each position in the positive class is
    # supported by at least required_min_pos_counts  of reads
    do_eval = yt.sum(axis=1).mean(axis=1) > required_min_pos_counts / pos_min_threshold
    
    # make sure everything sums to one
    yp = yp / yp.sum(axis=1, keepdims=True)
    fracs = yt / yt.sum(axis=1, keepdims=True)
    
    yp_random = permute_array(permute_array(yp[do_eval], axis=1), axis=0)
    out = []
    for binsize in binsizes:
        is_peak = (fracs >= pos_min_threshold).astype(float)
        ambigous = (fracs < pos_min_threshold) & (fracs >= neg_max_threshold)
        is_peak[ambigous] = -1
        y_true = np.ravel(bin_counts_amb(is_peak[do_eval], binsize))

        imbalance = np.sum(y_true == 1) / np.sum(y_true >= 0)
        n_positives = np.sum(y_true == 1)
        n_ambigous = np.sum(y_true == -1)
        frac_ambigous = n_ambigous / y_true.size

        # TODO - I used to have bin_counts_max over here instead of bin_counts_sum
        try:
            res = auprc(y_true,
                        np.ravel(bin_counts_max(yp[do_eval], binsize)))
            res_random = auprc(y_true,
                               np.ravel(bin_counts_max(yp_random, binsize)))
        except ValueError:
            res = np.nan
            res_random = np.nan

        out.append({"binsize": binsize,
                    "auprc": res,
                    "random_auprc": res_random,
                    "n_positives": n_positives,
                    "frac_ambigous": frac_ambigous,
                    "imbalance": imbalance
                    })

    return pd.DataFrame.from_dict(out)


def evaluate(model_dir,
             output_dir=None,
             gpu=0,
             exclude_metrics=False,
             splits=['train', 'valid'],
             model_path=None,
             data=None,
             hparams=None,
             dataspec=None,
             preprocessor=None):
    """
    Args:
      model_dir: path to the model directory
      splits: For which data splits to compute the evaluation metrics
      model_metrics: if True, metrics computed using mode.evaluate(..)
    """
    if gpu is not None:
        create_tf_session(gpu)
    if dataspec is not None:
        ds = DataSpec.load(dataspec)
    else:
        ds = DataSpec.load(os.path.join(model_dir, "dataspec.yaml"))
    if hparams is not None:
        hp = HParams.load(hparams)
    else:
        hp = HParams.load(os.path.join(model_dir, "hparams.yaml"))
    if model_path is not None:
        model = load_model(model_path)
    else:
        model = load_model(os.path.join(model_dir, "model.h5"))
    if output_dir is None:
        output_dir = os.path.join(model_dir, "eval")
    train, valid, test = load_data(model_dir,
                                   dataspec=dataspec,
                                   hparams=hparams,
                                   data=data,
                                   preprocessor=preprocessor)
    data = dict(train=train, valid=valid, test=test)

    metrics = {}
    profile_metrics = []
    os.makedirs(os.path.join(output_dir, "plots"),
                exist_ok=True)
    for split in tqdm(splits):
        y_pred = model.predict(data[split][0])
        y_true = data[split][1]
        if not exclude_metrics:
            eval_metrics_values = model.evaluate(data[split][0],
                                                 data[split][1])
            eval_metrics = dict(zip(_listify(model.metrics_names),
                                    _listify(eval_metrics_values)))
            eval_metrics = {split + "/" + k.replace("_", "/"): v
                            for k, v in eval_metrics.items()}
            metrics = {**eval_metrics, **metrics}
        for task in ds.task_specs:
            # Counts

            yp = y_pred[ds.task2idx(task, "counts")].sum(axis=-1)
            yt = y_true["counts/" + task].sum(axis=-1)
            # compute the correlation
            rp = pearsonr(yt, yp)[0]
            rs = spearmanr(yt, yp)[0]
            metrics = {**metrics,
                       split + f"/counts/{task}/pearsonr": rp,
                       split + f"/counts/{task}/spearmanr": rs,
                       }

            fig = plt.figure(figsize=(5, 5))
            plt.scatter(yp, yt, alpha=0.5)
            plt.xlabel("Predicted")
            plt.ylabel("Observed")
            plt.title(f"R_pearson={rp:.2f}, R_spearman={rs:.2f}")
            plt.savefig(os.path.join(output_dir, f"plots/counts.{split}.{task}.png"))

            # Profile
            yp = softmax(y_pred[ds.task2idx(task, "profile")])
            yt = y_true["profile/" + task]
            df = eval_profile(yt, yp,
                              pos_min_threshold=hp.evaluate.pos_min_threshold,
                              neg_max_threshold=hp.evaluate.neg_max_threshold,
                              required_min_pos_counts=hp.evaluate.required_min_pos_counts,
                              binsizes=hp.evaluate.binsizes)
            df['task'] = task
            df['split'] = split
            # Evaluate for the smallest binsize
            auprc_min = df[df.binsize == min(hp.evaluate.binsizes)].iloc[0].auprc
            metrics[split + f'/profile/{task}/auprc'] = auprc_min
            profile_metrics.append(df)

    # Write the count metrics
    write_json(metrics, os.path.join(output_dir, "metrics.json"))

    # write the profile metrics
    dfm = pd.concat(profile_metrics)
    dfm.to_csv(os.path.join(output_dir, "profile_metrics.tsv"),
               sep='\t', index=False)
    return dfm, metrics


def pprint_metrics():
    pass
