from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import os
from typing import Any, List, Mapping, Optional, Tuple, Union
from official_data_loader.augment import AutoAugment
from official_data_loader.augment import RandAugment
from official_data_loader.preprocessing_train import preprocess_for_train
from official_data_loader.preprocessing_eval import preprocess_for_eval



class DatasetBuilder:
    def __init__(self, data_dir, batch_size, one_hot, is_training, dtype):
        self.name: str = 'imagenet2012'
        self.builder: str = 'tfds'
        self.image_size: int = 224
        self.num_channels: int = 3
        num_examples: int = 1281167
        self.num_classes: int = 1000
        self.batch_size: int = batch_size

        self.skip_decoding = True
        self.cache: bool = False
        self.shuffle_buffer_size = 1024 # ??
        self.one_hot = one_hot

        self.mean_subtract = True
        self.standardize = True
        self.data_dir = data_dir
        self.augmenter = None

        self.is_training = is_training # or False
        self.split = "train" if is_training else "validation"
        self.dtype = dtype

    @property
    def global_batch_size(self):
        """The global batch size across all replicas."""
        return self.batch_size
        
    def build_augment(self):

        AUGMENTERS = {
                        'autoaugment': AutoAugment,
                        'randaugment': RandAugment,
                    }
        # params and name unknown
        params = self.params or {}
        augmenter = AUGMENTERS.get(self.name, None)
        return augmenter(**params) if augmenter is not None else None
        
    def build(self, strategy) -> tf.data.Dataset:
        """Construct a dataset end-to-end and return it using an optional strategy.

        Args:
        strategy: a strategy that, if passed, will distribute the dataset
            according to that strategy. If passed and `num_devices > 1`,
            `use_per_replica_batch_size` must be set to `True`.

        Returns:
        A TensorFlow dataset outputting batched images and labels.
        """
        if strategy:
            dataset = strategy.distribute_datasets_from_function(self._build)
        else:
            dataset = self._build()

        return dataset

    def _build(self, input_context) -> tf.data.Dataset:
        """Construct a dataset end-to-end and return it.

        Args:
        input_context: An optional context provided by `tf.distribute` for
            cross-replica training.

        Returns:
        A TensorFlow dataset outputting batched images and labels.
        """
        builders = {
            'tfds': self.load_tfds,
            # 'records': self.load_records,
            # 'synthetic': self.load_synthetic,
        }

        builder = builders.get(self.builder, None)

        self.input_context = input_context
        dataset = builder()
        dataset = self.pipeline(dataset)

        return dataset


    def load_tfds(self) -> tf.data.Dataset:
        """Return a dataset loading files from TFDS."""

        logging.info('Using TFDS to load data.')
        print("data_dir: ", self.data_dir)
        builder = tfds.builder(self.name, data_dir=self.data_dir)

        # self.config.download = True
        # if self.config.download:
        #     builder.download_and_prepare(download_dir = self.data_dir)

        decoders = {}

        if self.skip_decoding:
            decoders['image'] = tfds.decode.SkipDecoding()

        read_config = tfds.ReadConfig(
            interleave_cycle_length=10,
            interleave_block_length=1,
            input_context=self.input_context)

        dataset = builder.as_dataset(
            split=self.split,
            as_supervised=True,
            shuffle_files=False, # True
            decoders=decoders,
            read_config=read_config)

        return dataset

    def preprocess(self, image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Apply image preprocessing and augmentation to the image and label."""


        if self.is_training:
            image = preprocess_for_train(
                    image,
                    image_size=self.image_size,
                    mean_subtract=self.mean_subtract,
                    standardize=self.standardize,
                    dtype=self.dtype,
                    augmenter=self.augmenter)
        else:
            image = preprocess_for_eval(
                    image,
                    image_size=self.image_size,
                    num_channels=self.num_channels,
                    mean_subtract=self.mean_subtract,
                    standardize=self.standardize,
                    dtype=self.dtype)

        label = tf.cast(label, tf.int32)
        if self.one_hot:
            label = tf.one_hot(label, self.num_classes)
            label = tf.reshape(label, [self.num_classes])

        # tf.print("original:", tf.reduce_min(image), tf.reduce_max(image))
        # image = tf.random.uniform(
        #         shape=image.shape, 
        #         minval=tf.cast(tf.reduce_min(image), dtype=tf.float32), 
        #         maxval=tf.cast(tf.reduce_max(image), dtype=tf.float32)
        #     )
        # tf.print("random: ", tf.reduce_min(image), tf.reduce_max(image))
        return image, label

    def pipeline(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Build a pipeline fetching, shuffling, and preprocessing the dataset.

        Args:
        dataset: A `tf.data.Dataset` that loads raw files.

        Returns:
        A TensorFlow dataset outputting batched images and labels.
        """
        # dataset = dataset.take(1024)
        if self.is_training and not self.cache:
            dataset = dataset.repeat()

        if self.cache:
            dataset = dataset.cache()

        if self.is_training:
            dataset = dataset.shuffle(self.shuffle_buffer_size)
            dataset = dataset.repeat()

        # Parse, pre-process, and batch the data in parallel
        preprocess = self.preprocess
        dataset = dataset.map(
            preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.batch(
            self.global_batch_size, drop_remainder=self.is_training)

        # Prefetch overlaps in-feed with training
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # dataset = dataset.take(1024)

        return dataset


