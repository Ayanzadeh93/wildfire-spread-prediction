# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common functions for building TF models (adapted for standalone use)."""

import glob
import json
import logging
import os
import time
from typing import Text, Dict, Any, Tuple, Optional

import tensorflow as tf
from tensorflow.compat.v2 import keras


CONV2D_FILTERS_DEFAULT = 64
CONV2D_KERNEL_SIZE_DEFAULT = 3
CONV2D_STRIDES_DEFAULT = 1
CONV2D_PADDING_DEFAULT = 'same'
CONV2D_BIAS_DEFAULT = False
RES_SHORTCUT_KERNEL_SIZE = 1
RES_STRIDES_LIST_DEFAULT = (2, 1)
RES_DECODER_STRIDES = (1, 1)
RES_POOL_SIZE_DEFAULT = 2
DROPOUT_DEFAULT = 0.0
BATCH_NORM_DEFAULT = 'none'
L1_REGULARIZATION_DEFAULT = 0.0
L2_REGULARIZATION_DEFAULT = 0.0
CLIPNORM_DEFAULT = 1e6


def conv2d_layer(
    filters=CONV2D_FILTERS_DEFAULT,
    kernel_size=CONV2D_KERNEL_SIZE_DEFAULT,
    strides=CONV2D_STRIDES_DEFAULT,
    padding=CONV2D_PADDING_DEFAULT,
    use_bias=CONV2D_BIAS_DEFAULT,
    bias_initializer=None,
    l1_regularization=L1_REGULARIZATION_DEFAULT,
    l2_regularization=L2_REGULARIZATION_DEFAULT
):
  """Creates a Conv2D layer with optional L1/L2 regularization."""
  if bias_initializer is None:
    bias_initializer = keras.initializers.zeros()
  return keras.layers.Conv2D(
      filters=filters,
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      use_bias=use_bias,
      bias_initializer=bias_initializer,
      kernel_regularizer=keras.regularizers.l1_l2(
          l1=l1_regularization, l2=l2_regularization))


def res_block(
    input_tensor,
    filters,
    strides=RES_STRIDES_LIST_DEFAULT,
    pool_size=RES_POOL_SIZE_DEFAULT,
    dropout=DROPOUT_DEFAULT,
    batch_norm=BATCH_NORM_DEFAULT,
    l1_regularization=L1_REGULARIZATION_DEFAULT,
    l2_regularization=L2_REGULARIZATION_DEFAULT):
  """Creates convolution layer blocks with residual connections.

  Args:
    input_tensor: Input to the residual block.
    filters: Filters to use in successive layers (tuple of 2).
    strides: Strides to use in successive layers (tuple of 2).
    pool_size: Size of the max pool window.
    dropout: Dropout rate.
    batch_norm: Controls batch normalization: 'none', 'some', or 'all'.
    l1_regularization: L1 regularization factor applied on the kernel.
    l2_regularization: L2 regularization factor applied on the kernel.

  Returns:
    Output of the residual block.
  """
  res_path = input_tensor
  if batch_norm != 'none':
    res_path = keras.layers.BatchNormalization()(res_path)
  res_path = keras.layers.LeakyReLU()(res_path)
  res_path = keras.layers.Dropout(dropout)(res_path)

  if strides[0] == 1:
    res_path = conv2d_layer(
        filters=filters[0],
        l1_regularization=l1_regularization,
        l2_regularization=l2_regularization)(res_path)
  else:
    res_path = keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=strides[0])(res_path)

  if batch_norm == 'all':
    res_path = keras.layers.BatchNormalization()(res_path)
  res_path = keras.layers.LeakyReLU()(res_path)
  res_path = keras.layers.Dropout(dropout)(res_path)
  res_path = conv2d_layer(
      filters=filters[1],
      strides=strides[1],
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)(res_path)

  # Residual shortcut connection.
  shortcut = conv2d_layer(
      filters=filters[1],
      kernel_size=RES_SHORTCUT_KERNEL_SIZE,
      strides=strides[0],
      l1_regularization=l1_regularization,
      l2_regularization=l2_regularization)(input_tensor)
  if batch_norm == 'all':
    shortcut = keras.layers.BatchNormalization()(shortcut)
  res_path = keras.layers.Dropout(dropout)(res_path)

  res_path = shortcut + res_path
  return res_path


class BestModelExporter(tf.keras.callbacks.Callback):
  """Saves the best model checkpoint based on a monitored metric."""

  def __init__(self, metric_key, min_or_max, output_dir):
    super().__init__()
    self.metric_key = metric_key
    self.output_dir = output_dir
    if min_or_max not in ('min', 'max'):
      raise ValueError("min_or_max must be 'min' or 'max'")
    self.mode = min_or_max
    self.best = None
    os.makedirs(output_dir, exist_ok=True)

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    metric = logs.get(self.metric_key)
    if metric is None:
      return
    if (self.best is None or
        (self.mode == 'min' and metric < self.best) or
        (self.mode == 'max' and metric > self.best)):
      self.best = metric
      save_path = os.path.join(self.output_dir, f'best_model_epoch{epoch:03d}.weights.h5')
      self.model.save_weights(save_path)
      logging.info('Best model saved: %s (%s=%.4f)', save_path,
                   self.metric_key, metric)
