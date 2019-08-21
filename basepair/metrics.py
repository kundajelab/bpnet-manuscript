import pandas as pd
import numpy as np
from collections import OrderedDict
import concise
from concise.eval_metrics import auprc, auc, accuracy
from basepair.functions import softmax, mean
import gin
# Metric helpers


def average_profile(pe):
    tasks = list(pe)
    binsizes = list(pe[tasks[0]])
    return {binsize: {"auprc": mean([pe[task][binsize]['auprc'] for task in tasks])}
            for binsize in binsizes}


def average_counts(pe):
    tasks = list(pe)
    metrics = list(pe[tasks[0]])
    return {metric: mean([pe[task][metric] for task in tasks])
            for metric in metrics}


@gin.configurable
class BPNetSeparatePostproc:

    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self, y_true, preds):
        profile_preds = {task: softmax(preds[task_i])
                         for task_i, task in enumerate(self.tasks)}
        count_preds = {task: preds[len(self.tasks) + task_i].sum(axis=-1)
                       for task_i, task in enumerate(self.tasks)}
        profile_true = {task: y_true[f'profile/{task}']
                        for task in self.tasks}
        counts_true = {task: y_true[f'counts/{task}'].sum(axis=-1)
                       for task in self.tasks}
        return ({"profile": profile_true, "counts": counts_true},
                {"profile": profile_preds, "counts": count_preds})


@gin.configurable
class BPNetSinglePostproc:
    """Example where we predict a single track
    """

    def __init__(self, tasks):
        self.tasks = tasks

    def __call__(self, y_true, preds):
        profile_preds = {task: preds[task_i] / preds[task_i].sum(axis=-2, keepdims=True)
                         for task_i, task in enumerate(self.tasks)}
        count_preds = {task: np.log(1 + preds[task_i].sum(axis=(-2, -1)))
                       for task_i, task in enumerate(self.tasks)}

        profile_true = {task: y_true[f'profile/{task}']
                        for task in self.tasks}
        counts_true = {task: np.log(1 + y_true[f'profile/{task}'].sum(axis=(-2, -1)))
                       for task in self.tasks}
        return ({"profile": profile_true, "counts": counts_true},
                {"profile": profile_preds, "counts": count_preds})


@gin.configurable
class BPNetMetric:
    """BPNet metrics when the net is predicting counts and profile separately
    """

    def __init__(self, tasks, count_metric,
                 profile_metric=None,
                 postproc_fn=None):
        """

        Args:
          tasks: tasks
          count_metric: count evaluation metric
          profile_metric: profile evaluation metric
        """
        self.tasks = tasks
        self.count_metric = count_metric
        self.profile_metric = profile_metric

        if postproc_fn is None:
            self.postproc_fn = BPNetSeparatePostproc(tasks=self.tasks)
        else:
            self.postproc_fn = postproc_fn

    def __call__(self, y_true, preds):
        # extract the profile and count predictions

        y_true, preds = self.postproc_fn(y_true, preds)

        out = {}
        out["counts"] = {task: self.count_metric(y_true['counts'][task],
                                                 preds['counts'][task])
                         for task in self.tasks}
        out["counts"]['avg'] = average_counts(out["counts"])

        out["avg"] = {"counts": out["counts"]['avg']}  # new system compatibility
        if self.profile_metric is not None:
            out["profile"] = {task: self.profile_metric(y_true['profile'][task],
                                                        preds['profile'][task])
                              for task in self.tasks}
            out["profile"]['avg'] = average_profile(out["profile"])
            out["avg"]['profile'] = out["profile"]['avg']
        return out


@gin.configurable
class BPNetMetricSingleProfile:
    """BPNet metrics when the net is predicting the total counts + profile at the same time
    """

    def __init__(self, count_metric,
                 profile_metric=None):
        """

        Args:
          tasks: tasks
          count_metric: count evaluation metric
          profile_metric: profile evaluation metric
        """
        # self.tasks = tasks
        self.count_metric = count_metric
        self.profile_metric = profile_metric

    def __call__(self, y_true, preds):
        # extract the profile and count predictions
        out = {}

        # sum across positions + strands
        out["counts"] = self.count_metric(np.log(1 + y_true.sum(axis=(-2, -1))),
                                          np.log(1 + preds.sum(axis=(-2, -1))))

        if self.profile_metric is not None:
            out["profile"] = self.profile_metric(y_true, preds)
        return out


@gin.configurable
class PeakPredictionProfileMetric:

    def __init__(self, pos_min_threshold=0.05,
                 neg_max_threshold=0.01,
                 required_min_pos_counts=2.5,
                 binsizes=[1, 10]):

        self.pos_min_threshold = pos_min_threshold
        self.neg_max_threshold = neg_max_threshold
        self.required_min_pos_counts = required_min_pos_counts
        self.binsizes = binsizes

    def __call__(self, y_true, y_pred):
        from basepair.cli.evaluate import eval_profile
        out = eval_profile(y_true, y_pred,
                           pos_min_threshold=self.pos_min_threshold,
                           neg_max_threshold=self.neg_max_threshold,
                           required_min_pos_counts=self.required_min_pos_counts,
                           binsizes=self.binsizes)

        return {f"binsize={k}": v for k, v in out.set_index("binsize").to_dict("index").items()}


@gin.configurable
def pearson_spearman(yt, yp):
    from scipy.stats import pearsonr, spearmanr
    return {"pearsonr": pearsonr(yt, yp)[0],
            "spearmanr": spearmanr(yt, yp)[0]}
