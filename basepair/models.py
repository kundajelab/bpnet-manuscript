import numpy as np
import keras.layers as kl
from keras.optimizers import Adam
from keras.models import Model
from concise.utils.helper import get_from_module
import basepair
import basepair.losses as blosses
from basepair.losses import twochannel_multinomial_nll, MultichannelMultinomialNLL
import gin
import keras


@gin.configurable
def multihead_seq_model(tasks,
                        filters,
                        n_dil_layers,
                        conv1_kernel_size,
                        tconv_kernel_size,
                        b_loss_weight=1,
                        c_loss_weight=1,
                        p_loss_weight=1,
                        c_splines=20,
                        p_splines=0,
                        merge_profile_reg=False,
                        lr=0.004,
                        padding='same',
                        batchnorm=False,
                        use_bias=False,
                        n_profile_bias_tracks=2,
                        n_bias_tracks=2,
                        seqlen=None,
                        skip_type='residual'):
    from basepair.seqmodel import SeqModel
    from basepair.layers import DilatedConv1D, DeConv1D, GlobalAvgPoolFCN
    from basepair.metrics import BPNetMetricSingleProfile
    from basepair.heads import ScalarHead, ProfileHead
    from gin_train.metrics import ClassificationMetrics, RegressionMetrics
    from basepair.losses import mc_multinomial_nll_2, CountsMultinomialNLL
    from basepair.exp.paper.config import peak_pred_metric
    from basepair.activations import clipped_exp
    from basepair.functions import softmax

    assert p_loss_weight >= 0
    assert c_loss_weight >= 0
    assert b_loss_weight >= 0

    # Heads -------------------------------------------------
    heads = []
    # Profile prediction
    if p_loss_weight > 0:
        if not merge_profile_reg:
            heads.append(ProfileHead(target_name='{task}/profile',
                                     net=DeConv1D(n_tasks=2,
                                                  filters=filters,
                                                  tconv_kernel_size=tconv_kernel_size,
                                                  padding=padding,
                                                  n_hidden=0,
                                                  batchnorm=batchnorm
                                                  ),
                                     loss=mc_multinomial_nll_2,
                                     loss_weight=p_loss_weight,
                                     postproc_fn=softmax,
                                     use_bias=use_bias,
                                     bias_input='bias/{task}/profile',
                                     bias_shape=(None, n_profile_bias_tracks),
                                     metric=peak_pred_metric
                                     ))
        else:
            heads.append(ProfileHead(target_name='{task}/profile',
                                     net=DeConv1D(n_tasks=2,
                                                  filters=filters,
                                                  tconv_kernel_size=tconv_kernel_size,
                                                  padding=padding,
                                                  n_hidden=1,  # use 1 hidden layer in that case
                                                  batchnorm=batchnorm
                                                  ),
                                     activation=clipped_exp,
                                     loss=CountsMultinomialNLL(2, c_task_weight=c_loss_weight),
                                     loss_weight=p_loss_weight,
                                     bias_input='bias/{task}/profile',
                                     use_bias=use_bias,
                                     bias_shape=(None, n_profile_bias_tracks),
                                     metric=BPNetMetricSingleProfile(count_metric=RegressionMetrics(),
                                                                     profile_metric=peak_pred_metric)
                                     ))
            c_loss_weight = 0  # don't need to use the other count loss

    # Count regression
    if c_loss_weight > 0:
        heads.append(ScalarHead(target_name='{task}/counts',
                                net=GlobalAvgPoolFCN(n_tasks=2,
                                                     n_splines=p_splines,
                                                     batchnorm=batchnorm),
                                activation=None,
                                loss='mse',
                                loss_weight=c_loss_weight,
                                bias_input='bias/{task}/counts',
                                use_bias=use_bias,
                                bias_shape=(n_bias_tracks, ),
                                metric=RegressionMetrics(),
                                ))

    # Binary classification
    if b_loss_weight > 0:
        heads.append(ScalarHead(target_name='{task}/class',
                                net=GlobalAvgPoolFCN(n_tasks=1,
                                                     n_splines=c_splines,
                                                     batchnorm=batchnorm),
                                activation='sigmoid',
                                loss='binary_crossentropy',
                                loss_weight=b_loss_weight,
                                metric=ClassificationMetrics(),
                                ))
    # -------------------------------------------------
    m = SeqModel(
        body=DilatedConv1D(filters=filters,
                           conv1_kernel_size=conv1_kernel_size,
                           n_dil_layers=n_dil_layers,
                           padding=padding,
                           batchnorm=batchnorm,
                           skip_type=skip_type),
        heads=heads,
        tasks=tasks,
        optimizer=Adam(lr=lr),
        seqlen=seqlen,
    )
    return m


@gin.configurable
def binary_seq_model(tasks,
                     net_body,
                     net_head,
                     lr=0.004,
                     seqlen=None):
    """NOTE: This doesn't work with gin-train since
    the classes injected by gin-config can't be pickled.

    Instead, I created `basset_seq_model`

    ```
    Can't pickle <class 'basepair.layers.BassetConv'>: it's not the same
    object as basepair.layers.BassetConv
    ```

    """
    from basepair.seqmodel import SeqModel
    from basepair.heads import ScalarHead, ProfileHead
    from gin_train.metrics import ClassificationMetrics
    # Heads -------------------------------------------------
    heads = [ScalarHead(target_name='{task}/class',
                        net=net_head,
                        activation='sigmoid',
                        loss='binary_crossentropy',
                        metric=ClassificationMetrics(),
                        )]
    # -------------------------------------------------
    m = SeqModel(
        body=net_body,
        heads=heads,
        tasks=tasks,
        optimizer=Adam(lr=lr),
        seqlen=seqlen,
    )
    return m


@gin.configurable
def basset_seq_model(tasks,
                     hidden=(1000, 1000),
                     dropout=(0.0, 0.0),
                     final_dropout=0.0,
                     batchnorm=False,
                     lr=0.004,
                     body='Basset',
                     seqlen=None):
    from basepair.seqmodel import SeqModel
    from basepair.heads import ScalarHead, ProfileHead
    from gin_train.metrics import ClassificationMetrics
    from basepair.layers import FCN
    from basepair import layers
    # Heads -------------------------------------------------
    heads = [ScalarHead(target_name='{task}/class',
                        net=FCN(dropout=final_dropout,
                                batchnorm=batchnorm),
                        activation='sigmoid',
                        loss='binary_crossentropy',
                        metric=ClassificationMetrics(),
                        )]
    # -------------------------------------------------
    m = SeqModel(
        body=layers.get(body)(hidden=hidden,
                              batchnorm=batchnorm,
                              dropout=dropout),
        heads=heads,
        tasks=tasks,
        optimizer=Adam(lr=lr),
        seqlen=seqlen,
    )
    return m


def sklearn_estimator(estimator_name: str, kwargs: dict = None):
    """Get a sklearn estimator by name

    Args:
      estimator_name [str]: estimator name in sklearn
      kwargs [dict]: kwargs passed to the estimator
    """
    if kwargs is None:
        kwargs = dict()
    from collections import OrderedDict
    from sklearn.utils.testing import all_estimators
    return OrderedDict(all_estimators())[estimator_name](**kwargs)


def get(name):
    return get_from_module(name, globals())
