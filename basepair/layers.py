import keras.layers as kl
from keras.engine.topology import Layer
import tensorflow as tf
from concise.utils.helper import get_from_module
from concise.layers import SplineWeight1D
from keras.models import Model, Sequential
import numpy as np
import gin


@gin.configurable
class GlobalAvgPoolFCN:

    def __init__(self,
                 n_tasks=1,
                 dropout=0,
                 hidden=None,
                 dropout_hidden=0,
                 n_splines=0,
                 batchnorm=False):
        self.n_tasks = n_tasks
        self.dropout = dropout
        self.dropout_hidden = dropout_hidden
        self.batchnorm = batchnorm
        self.n_splines = n_splines
        self.hidden = hidden if hidden is not None else []
        assert self.n_splines >= 0

    def __call__(self, x):
        if self.n_splines == 0:
            x = kl.GlobalAvgPool1D()(x)
        else:
            # Spline-transformation for the position aggregation
            # This allows to up-weight positions in the middle
            x = SplineWeight1D(n_bases=self.n_splines,
                               share_splines=True)(x)
            x = kl.GlobalAvgPool1D()(x)

        if self.dropout:
            x = kl.Dropout(self.dropout)(x)

        # Hidden units (not used by default)
        for h in self.hidden:
            if self.batchnorm:
                x = kl.BatchNormalization()(x)
            x = kl.Dense(h, activation='relu')(x)
            if self.dropout_hidden:
                x = kl.Dropout(self.dropout_hidden)(x)

        # Final dense layer
        if self.batchnorm:
            x = kl.BatchNormalization()(x)
        x = kl.Dense(self.n_tasks)(x)
        return x


@gin.configurable
class FCN:

    def __init__(self,
                 n_tasks=1,
                 hidden=None,
                 dropout=0,
                 dropout_hidden=0,
                 batchnorm=False):
        self.n_tasks = n_tasks
        self.dropout = dropout
        self.dropout_hidden = dropout_hidden
        self.batchnorm = batchnorm
        self.hidden = hidden if hidden is not None else []

    def __call__(self, x):
        if self.dropout:
            x = kl.Dropout(self.dropout)(x)

        # Hidden units (not used by default)
        for h in self.hidden:
            if self.batchnorm:
                x = kl.BatchNormalization()(x)
            x = kl.Dense(h, activation='relu')(x)
            if self.dropout_hidden:
                x = kl.Dropout(self.dropout_hidden)(x)

        # Final dense layer
        if self.batchnorm:
            x = kl.BatchNormalization()(x)
        x = kl.Dense(self.n_tasks)(x)
        return x


@gin.configurable
class Basset:

    def __init__(self,
                 filters=(300, 200, 200),
                 conv_width=(19, 11, 7),
                 batchnorm=False,
                 pool_width=(3, 4, 4),
                 pool_stride=(3, 4, 4),
                 kernel_initializer='he_normal',
                 # FCN
                 hidden=(1000, 1000),
                 dropout=(0.0, 0.0),
                 ):
        self.filters = filters
        self.conv_width = conv_width
        self.batchnorm = batchnorm
        self.pool_width = pool_width
        self.pool_stride = pool_stride
        self.kernel_initializer = kernel_initializer
        self.hidden = hidden
        self.dropout = dropout
        self.batchnorm = batchnorm

    def __call__(self, inp):
        # Conv-layers
        seq_preds = inp
        for i, (nb_filter, nb_col) in enumerate(zip(self.filters, self.conv_width)):
            seq_preds = kl.Conv1D(filters=nb_filter,
                                  kernel_size=nb_col,
                                  kernel_initializer=self.kernel_initializer)(seq_preds)
            if self.batchnorm:
                seq_preds = kl.BatchNormalization()(seq_preds)
            seq_preds = kl.Activation('relu')(seq_preds)
            # pool
            seq_preds = kl.MaxPooling1D(self.pool_width[i], self.pool_stride[i])(seq_preds)

        seq_preds = kl.Flatten()(seq_preds)

        # fully connected
        for drop_rate, fc_layer_size in zip(self.dropout, self.hidden):
            seq_preds = kl.Dropout(drop_rate)(seq_preds)
            seq_preds = kl.Dense(fc_layer_size)(seq_preds)
            if self.batchnorm:
                seq_preds = kl.BatchNormalization()(seq_preds)
            seq_preds = kl.Activation('relu')(seq_preds)
        # seq_preds = kl.Dropout(self.final_dropout)(seq_preds)

        return seq_preds


@gin.configurable
class FactorizedBasset:

    def __init__(self,
                 filters=(48, 64, 100, 150, 300, 200, 200, 200, 200),
                 conv_widths=(3, 3, 3, 7, 7, 7, 3, 3, 7),
                 batchnorm=False,
                 pool_layers_indices=(4, 7, 8),
                 pool_widths=(3, 4, 4),
                 pool_strides=(3, 4, 4),
                 kernel_initializer='he_normal',
                 activation_func='relu',
                 # FCN
                 hidden=(1000, 1000),
                 dropout=(0.0, 0.0),
                 final_dropout=0.0,
                 ):
        """
        Factorized basset architecture body
        https://www.biorxiv.org/content/biorxiv/early/2017/12/05/229385.full.pdf

        Taken from: https://raw.githubusercontent.com/kundajelab/kerasAC/master/kerasAC/architectures/functional_fbasset.py


        Use with BassetHead
        fc_layer_sizes = (1000, 1000)
        dropouts = (0.3, 0.3)
        """
        assert len(filters) == len(conv_widths)
        self.filters = filters
        self.conv_widths = conv_widths
        self.batchnorm = batchnorm
        self.pool_layers_indices = pool_layers_indices
        self.pool_widths = pool_widths
        self.pool_strides = pool_strides
        self.kernel_initializer = kernel_initializer
        self.activation_func = activation_func
        self.hidden = hidden
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.batchnorm = batchnorm

    def __call__(self, inp):
        # Conv-layers
        x = inp
        for i, (n_filter, n_col) in enumerate(zip(self.filters, self.conv_widths)):
            if i == 0:  # if first layer, specify input shape to enable auto build
                x = kl.Conv1D(
                    filters=n_filter,
                    kernel_size=n_col,
                    kernel_initializer='he_normal')(x)
            else:
                x = kl.Conv1D(
                    filters=n_filter,
                    kernel_size=n_col,
                    kernel_initializer='he_normal')(x)
                x = kl.BatchNormalization()(x)
                x = kl.Activation(self.activation_func)(x)

                if i in self.pool_layers_indices:  # Add pool layers where appropriate
                    x = kl.MaxPool1D(pool_size=self.pool_widths[self.pool_layers_indices.index(i)],
                                     strides=self.pool_strides[self.pool_layers_indices.index(i)])(x)

        seq_preds = kl.Flatten()(x)

        # fully connected
        for drop_rate, fc_layer_size in zip(self.dropout, self.hidden):
            seq_preds = kl.Dropout(drop_rate)(seq_preds)
            seq_preds = kl.Dense(fc_layer_size)(seq_preds)
            if self.batchnorm:
                seq_preds = kl.BatchNormalization()(seq_preds)
            seq_preds = kl.Activation('relu')(seq_preds)
        seq_preds = kl.Dropout(self.final_dropout)(seq_preds)

        return seq_preds


@gin.configurable
class DilatedConv1D:
    """Dillated convolutional layers

    - add_pointwise -> if True, add a 1by1 conv right after the first conv
    """

    def __init__(self, filters=21,
                 conv1_kernel_size=25,
                 n_dil_layers=6,
                 skip_type='residual',  # or 'dense', None
                 padding='same',
                 batchnorm=False,
                 add_pointwise=False):
        self.filters = filters
        self.conv1_kernel_size = conv1_kernel_size
        self.n_dil_layers = n_dil_layers
        self.skip_type = skip_type
        self.padding = padding
        self.batchnorm = batchnorm
        self.add_pointwise = add_pointwise

    def __call__(self, inp):
        """inp = (None, 4)
        """
        first_conv = kl.Conv1D(self.filters,
                               kernel_size=self.conv1_kernel_size,
                               padding='same',
                               activation='relu')(inp)
        if self.add_pointwise:
            if self.batchnorm:
                first_conv = kl.BatchNormalization()(first_conv)
            first_conv = kl.Conv1D(self.filters,
                                   kernel_size=1,
                                   padding='same',
                                   activation='relu')(first_conv)

        prev_layer = first_conv
        for i in range(1, self.n_dil_layers + 1):
            if self.batchnorm:
                x = kl.BatchNormalization()(prev_layer)
            else:
                x = prev_layer
            conv_output = kl.Conv1D(self.filters, kernel_size=3, padding='same',
                                    activation='relu', dilation_rate=2**i)(x)

            # skip connections
            if self.skip_type is None:
                prev_layer = conv_output
            elif self.skip_type == 'residual':
                prev_layer = kl.add([prev_layer, conv_output])
            elif self.skip_type == 'dense':
                prev_layer = kl.concatenate([prev_layer, conv_output])
            else:
                raise ValueError("skip_type needs to be 'add' or 'concat' or None")

        combined_conv = prev_layer

        if self.padding == 'valid':
            # Trim the output to only valid sizes
            # (e.g. essentially doing valid padding with skip-connections)
            combined_conv = kl.Cropping1D(cropping=-self.get_len_change() // 2)(combined_conv)

        # add one more layer in between for densly connected layers to reduce the
        # spatial dimension
        if self.skip_type == 'dense':
            combined_conv = kl.Conv1D(self.filters,
                                      kernel_size=1,
                                      padding='same',
                                      activation='relu')(combined_conv)
        return combined_conv

    def get_len_change(self):
        """How much will the length change
        """
        if self.padding == 'same':
            return 0
        else:
            d = 0
            # conv
            d -= 2 * (self.conv1_kernel_size // 2)
            for i in range(1, self.n_dil_layers + 1):
                dillation = 2**i
                d -= 2 * dillation
            return d


@gin.configurable
class DeConv1D:
    def __init__(self, filters, n_tasks,
                 tconv_kernel_size=25,
                 padding='same',
                 n_hidden=0,
                 batchnorm=False):
        self.filters = filters
        self.n_tasks = n_tasks
        self.tconv_kernel_size = tconv_kernel_size
        self.n_hidden = n_hidden
        self.batchnorm = batchnorm
        self.padding = padding

    def __call__(self, x):

        # `hidden` conv layers
        for i in range(self.n_hidden):
            if self.batchnorm:
                x = kl.BatchNormalization()(x)
            x = kl.Conv1D(self.filters,
                          kernel_size=1,
                          padding='same',  # anyway doesn't matter
                          activation='relu')(x)

        # single de-conv layer
        x = kl.Reshape((-1, 1, self.filters))(x)
        if self.batchnorm:
            x = kl.BatchNormalization()(x)
        x = kl.Conv2DTranspose(self.n_tasks, kernel_size=(self.tconv_kernel_size, 1), padding='same')(x)
        x = kl.Reshape((-1, self.n_tasks))(x)

        # TODO - allow multiple de-conv layers

        if self.padding == 'valid':
            # crop to make it translationally invariant
            x = kl.Cropping1D(cropping=-self.get_len_change() // 2)(x)
        return x

    def get_len_change(self):
        """How much will the length change
        """
        if self.padding == 'same':
            return 0
        else:
            return - 2 * (self.tconv_kernel_size // 2)


def dense_dilated_valid_conv(filters=21,
                             conv1_kernel_size=25,
                             n_dil_layers=6,
                             batchnorm=False):
    """Multiple densly connected dillated layers
    \   /\   /
     ---  ---
        \ /
    """
    assert conv1_kernel_size % 2 == 1

    inp = kl.Input(shape=(None, 4), name='seq')
    first_conv = kl.Conv1D(filters,
                           kernel_size=conv1_kernel_size,
                           padding='same',
                           activation='relu')(inp)
    prev_layers = [first_conv]
    for i in range(1, n_dil_layers + 1):
        if i == 1:
            prev_sum = first_conv
        else:
            prev_sum = kl.add(prev_layers)
        if batchnorm:
            prev_sum = kl.BatchNormalization()(prev_sum)
        conv_output = kl.Conv1D(filters, kernel_size=3, padding='same',
                                activation='relu', dilation_rate=2**i)(prev_sum)
        prev_layers.append(conv_output)
    combined_conv = kl.add(prev_layers)

    # Trim the output to only valid sizes
    # (e.g. essentially doing valid padding with skip-connections)
    change = -len_change_dense_dilated_valid_conv(conv1_kernel_size=conv1_kernel_size,
                                                  n_dil_layers=n_dil_layers)
    out = kl.Cropping1D(cropping=change // 2)(combined_conv)
    return Model(inp, out, name='conv_model')


def len_change_dense_dilated_valid_conv(conv1_kernel_size=25, n_dil_layers=6):
    """Compute length change"""
    d = 0

    # conv
    d -= 2 * (conv1_kernel_size // 2)
    for i in range(1, n_dil_layers + 1):
        dillation = 2**i
        d -= 2 * dillation
    return d


def cropped_deconv_1d(filters, n_tasks,
                      tconv_kernel_size=25,
                      n_hidden=0,
                      batchnorm=False):
    """De-convolutional layer with valid-trimming

          output
        ==========
     \ /          \ / <- deconv
      --------------
          input
    """
    input_shape = (None, filters)

    hconvs = []
    for i in range(n_hidden):
        # `hidden` conv layers
        if batchnorm:
            hconvs.append(kl.BatchNormalization(input_shape=input_shape))
        hconvs.append(kl.Conv1D(filters,
                                kernel_size=1,
                                activation='relu',
                                input_shape=input_shape))

    if batchnorm:
        bn = [kl.BatchNormalization()]
    else:
        bn = []
    return Sequential(hconvs + [
        kl.Reshape((-1, 1, filters), input_shape=input_shape)] + bn + [
        kl.Conv2DTranspose(n_tasks, kernel_size=(tconv_kernel_size, 1), padding='same'),
        kl.Reshape((-1, n_tasks)),
        # crop to make it translationally invariant
        kl.Cropping1D(cropping=tconv_kernel_size // 2),
    ], name='cropped_deconv_1d')


def len_change_cropped_deconv_1d(tconv_kernel_size=25):
    """Compute length change"""
    return - 2 * (tconv_kernel_size // 2)


def split_output(inp, outputs_per_task, tasks, bias_profile_inputs=dict(), name_prefix=''):
    """Split the output convolutional layer of size:
    `seq x sum(outputs_per_task)` to `len(outputs_per_task)` outputs
    each with shape `seq x outputs_per_task[i]`

    Args:
      inp: input tensor
      outputs_per_task: list of output task. Example: [2, 2, 1]
      bias_profile_inputs: dictionary of numpy arrays

    Returns:
      list of tensors with shapes (inp[0], outputs_per_task[i])
    """
    start_idx = np.cumsum([0] + outputs_per_task[:-1])
    end_idx = np.cumsum(outputs_per_task)

    def get_output_name(task):
        if task in bias_profile_inputs:
            return "lambda/profile/" + task
        else:
            return name_prefix + "profile/" + task
    output = [kl.Lambda(lambda x, i, sidx, eidx: x[:, :, sidx:eidx],
                        output_shape=(None, outputs_per_task[i]),
                        name=get_output_name(task),
                        arguments={"i": i, "sidx": start_idx[i], "eidx": end_idx[i]})(inp)
              for i, task in enumerate(tasks)]

    # Optional - bias correction
    for i, task in enumerate(tasks):
        if task in bias_profile_inputs:
            output_with_bias = kl.concatenate([output[i],
                                               bias_profile_inputs[task]], axis=-1)
            # batch x seqlen x (2+2)
            output[i] = kl.Conv1D(outputs_per_task[i],
                                  1,
                                  name=name_prefix + "profile/" + task)(output_with_bias)
    return output


def count_prediction(bottleneck, tasks, outputs_per_task,
                     bias_counts_inputs=[],
                     batchnorm=False):
    x = kl.GlobalAvgPool1D()(bottleneck)
    if bias_counts_inputs:
        # add bias as additional features if present
        x = kl.concatenate([x] + bias_counts_inputs, axis=-1)

    if batchnorm:
        x = kl.BatchNormalization()(x)
    # final layer
    counts = [kl.Dense(outputs_per_task[i], name="counts/" + task)(x)
              for i, task in enumerate(tasks)]
    return counts


AVAILABLE = []


def get(name):
    return get_from_module(name, globals())
