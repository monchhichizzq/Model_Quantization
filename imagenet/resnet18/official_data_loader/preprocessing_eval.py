import tensorflow as tf
from official_data_loader.preprocessing_fn import *


def preprocess_for_eval(
    image_bytes: tf.Tensor,
    image_size: int = IMAGE_SIZE,
    num_channels: int = 3,
    mean_subtract: bool = False,
    standardize: bool = False,
    dtype: tf.dtypes.DType = tf.float32
) -> tf.Tensor:
  """Preprocesses the given image for evaluation.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image height/width dimension.
    num_channels: number of image input channels.
    mean_subtract: whether or not to apply mean subtraction.
    standardize: whether or not to apply standardization.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
  images = decode_and_center_crop(image_bytes, image_size)
  images = tf.reshape(images, [image_size, image_size, num_channels])

  if mean_subtract:
    images = mean_image_subtraction(image_bytes=images, means=MEAN_RGB)
  if standardize:
    images = standardize_image(image_bytes=images, stddev=STDDEV_RGB)
  if dtype is not None:
    images = tf.image.convert_image_dtype(images, dtype=dtype)

  return images


def load_eval_image(filename: Text, image_size: int = IMAGE_SIZE) -> tf.Tensor:
  """Reads an image from the filesystem and applies image preprocessing.

  Args:
    filename: a filename path of an image.
    image_size: image height/width dimension.

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
  image_bytes = tf.io.read_file(filename)
  image = preprocess_for_eval(image_bytes, image_size)

  return image


def build_eval_dataset(filenames: List[Text],
                       labels: Optional[List[int]] = None,
                       image_size: int = IMAGE_SIZE,
                       batch_size: int = 1) -> tf.Tensor:
  """Builds a tf.data.Dataset from a list of filenames and labels.

  Args:
    filenames: a list of filename paths of images.
    labels: a list of labels corresponding to each image.
    image_size: image height/width dimension.
    batch_size: the batch size used by the dataset

  Returns:
    A preprocessed and normalized image `Tensor`.
  """
  if labels is None:
    labels = [0] * len(filenames)

  filenames = tf.constant(filenames)[:1024]
  labels = tf.constant(labels)[:1024]
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))

  dataset = dataset.map(
      lambda filename, label: (load_eval_image(filename, image_size), label))
  dataset = dataset.batch(batch_size)

  return dataset