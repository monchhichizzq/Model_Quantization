# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet50 model for Keras.
Adapted from tf.keras.applications.resnet50.ResNet50().
This is ResNet model version 1.5.
Related papers/blogs:
- https://arxiv.org/abs/1512.03385
- https://arxiv.org/pdf/1603.05027v2.pdf
- http://torch.ch/blog/2016/02/04/resnets.html
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import ReLU

def _gen_l2_regularizer(use_l2_regularizer=True, l2_weight_decay=1e-4):
    return tf.keras.regularizers.L2(
        l2_weight_decay) if use_l2_regularizer else None

def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True,
                   batch_norm_decay=0.9,
                   batch_norm_epsilon=1e-5):
    """The identity block is the block that has no conv layer at shortcut.
    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        use_l2_regularizer: whether to use L2 regularizer on Conv layer.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # layer 1
    x = layers.Conv2D(
        filters1, 
        kernel_size,
        use_bias=False,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
            input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(
            x)
    x = ReLU()(x)

    # layer2
    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(
            x)
    x = ReLU()(x)

    # add
    x = layers.add([x, input_tensor])
    x = ReLU()(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True,
               batch_norm_decay=0.9,
               batch_norm_epsilon=1e-5):
    """A block that has a conv layer at shortcut.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    Args:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the second conv layer in the block.
        use_l2_regularizer: whether to use L2 regularizer on Conv layer.
        batch_norm_decay: Moment of batch norm layers.
        batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
        Output tensor for the block.
    """
    filters1, filters2 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # layer 1
    x = layers.Conv2D(
        filters1, 
        kernel_size,
        use_bias=False,
        padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
            input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2a')(
            x)
    x = ReLU()(x)

    # layer 2
    x = layers.Conv2D(
        filters2,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(
            x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '2b')(
            x)
    x = ReLU()(x)

    # short cut
    shortcut = layers.Conv2D(
        filters2, (1, 1),
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '1')(
            input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name=bn_name_base + '1')(
            shortcut)

    x = layers.add([x, shortcut])
    x = ReLU()(x)
    return x


def resnet18(num_classes,
             batch_size=None,
             use_l2_regularizer=True,
             # rescale_inputs=False,
             batch_norm_decay=0.9,
             batch_norm_epsilon=1e-5):
    """Instantiates the ResNet18 architecture.
    Args:
    num_classes: `int` number of classes for image classification.
    batch_size: Size of the batches for each step.
    use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
    rescale_inputs: whether to rescale inputs from 0 to 1.
    batch_norm_decay: Moment of batch norm layers.
    batch_norm_epsilon: Epsilon of batch borm layers.
    Returns:
        A Keras model instance.
    """
    input_shape = (224, 224, 3)
    img_input = layers.Input(shape=input_shape, batch_size=batch_size)
    x = img_input 

    if tf.keras.backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        bn_axis = 1
    else:  # channels_last
        bn_axis = 3

    block_config = dict(
        use_l2_regularizer=use_l2_regularizer,
        batch_norm_decay=batch_norm_decay,
        batch_norm_epsilon=batch_norm_epsilon)
    print(block_config)
    
    # change 1
    # x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    # x = layers.Conv2D(
    #     64, (7, 7),
    #     strides=(2, 2),
    #     padding='valid',
    #     use_bias=False,
    #     kernel_initializer='he_normal',
    #     kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
    #     name='conv1')(
    #         x)
    
    x = layers.Conv2D(
                64, (7, 7),
                strides=(2, 2),
                padding='same',
                use_bias=False,
                kernel_initializer='he_normal',
                kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
                name='conv1')(
                    x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=batch_norm_decay,
        epsilon=batch_norm_epsilon,
        name='bn_conv1')(
            x)
    x = ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # resnet18 2,2,2,2
    # resnet34 3,4,6,3
    x = conv_block(
        x, 3, [64, 64], stage=2, block='a', strides=(1, 1), **block_config)
    x = identity_block(x, 3, [64, 64], stage=2, block='b', **block_config)

    x = conv_block(x, 3, [128, 128], stage=3, block='a', **block_config)
    x = identity_block(x, 3, [128, 128], stage=3, block='b', **block_config)
 
    x = conv_block(x, 3, [256, 256], stage=4, block='a', **block_config)
    x = identity_block(x, 3, [256, 256], stage=4, block='b', **block_config)

    x = conv_block(x, 3, [512, 512], stage=5, block='a', **block_config)
    x = identity_block(x, 3, [512, 512], stage=5, block='b', **block_config)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(
        num_classes,
        kernel_initializer=tf.initializers.random_normal(stddev=0.01),
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name='fc1000')(
            x)

    # A softmax that is followed by the model loss must be done cannot be done
    # in float16 due to numeric issues. So we pass dtype=float32.
    x = layers.Activation('softmax', dtype='float32')(x)

    # Create model.
    return tf.keras.Model(img_input, x, name='resnet50')