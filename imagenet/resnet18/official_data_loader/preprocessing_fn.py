import tensorflow as tf
from typing import List, Optional, Text, Tuple

# Calculated from the ImageNet training set
MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)

IMAGE_SIZE = 224
CROP_PADDING = 32

# resize_image
# decode_and_center_crop
# mean_image_subtraction
# standardize_image
# decode_crop_and_flip



def decode_crop_and_flip(image_bytes: tf.Tensor) -> tf.Tensor:
  """Crops an image to a random part of the image, then randomly flips.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.

  Returns:
    A decoded and cropped image `Tensor`.

  """
  decoded = image_bytes.dtype != tf.string
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  shape = (tf.shape(image_bytes) if decoded
           else tf.image.extract_jpeg_shape(image_bytes))
  sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
      shape,
      bounding_boxes=bbox,
      min_object_covered=0.1,
      aspect_ratio_range=[0.75, 1.33],
      area_range=[0.05, 1.0],
      max_attempts=100,
      use_image_if_no_bounding_boxes=True)
  bbox_begin, bbox_size, _ = sample_distorted_bounding_box

  # Reassemble the bounding box in the format the crop op requires.
  offset_height, offset_width, _ = tf.unstack(bbox_begin)
  target_height, target_width, _ = tf.unstack(bbox_size)
  crop_window = tf.stack([offset_height, offset_width,
                          target_height, target_width])
  if decoded:
    cropped = tf.image.crop_to_bounding_box(
        image_bytes,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=target_height,
        target_width=target_width)
  else:
    cropped = tf.image.decode_and_crop_jpeg(image_bytes,
                                            crop_window,
                                            channels=3)

  # Flip to add a little more random distortion in.
  cropped = tf.image.random_flip_left_right(cropped)
  return cropped


def standardize_image(
    image_bytes: tf.Tensor,
    stddev: Tuple[float, ...],
    num_channels: int = 3,
    dtype: tf.dtypes.DType = tf.float32,
) ->  tf.Tensor:
  """Divides the given stddev from each image channel.

  For example:
    stddev = [123.68, 116.779, 103.939]
    image_bytes = standardize_image(image_bytes, stddev)

  Note that the rank of `image` must be known.

  Args:
    image_bytes: a tensor of size [height, width, C].
    stddev: a C-vector of values to divide from each channel.
    num_channels: number of color channels in the image that will be distorted.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `stddev`.
  """
  if image_bytes.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(stddev) != num_channels:
    raise ValueError('len(stddev) must match the number of channels')

  # We have a 1-D tensor of stddev; convert to 3-D.
  # Note(b/130245863): we explicitly call `broadcast` instead of simply
  # expanding dimensions for better performance.
  stddev = tf.broadcast_to(stddev, tf.shape(image_bytes))
  if dtype is not None:
    stddev = tf.cast(stddev, dtype=dtype)

  return image_bytes / stddev


def mean_image_subtraction(
    image_bytes: tf.Tensor,
    means: Tuple[float, ...],
    num_channels: int = 3,
    dtype: tf.dtypes.DType = tf.float32,
) ->  tf.Tensor:
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image_bytes = mean_image_subtraction(image_bytes, means)

  Note that the rank of `image` must be known.

  Args:
    image_bytes: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.
    num_channels: number of color channels in the image that will be distorted.
    dtype: the dtype to convert the images to. Set to `None` to skip conversion.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image_bytes.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')

  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  # We have a 1-D tensor of means; convert to 3-D.
  # Note(b/130245863): we explicitly call `broadcast` instead of simply
  # expanding dimensions for better performance.
  means = tf.broadcast_to(means, tf.shape(image_bytes))
  if dtype is not None:
    means = tf.cast(means, dtype=dtype)

  return image_bytes - means

def decode_and_center_crop(image_bytes: tf.Tensor,
                           image_size: int = IMAGE_SIZE,
                           crop_padding: int = CROP_PADDING) -> tf.Tensor:
  """Crops to center of image with padding then scales image_size.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    image_size: image height/width dimension.
    crop_padding: the padding size to use when centering the crop.

  Returns:
    A decoded and cropped image `Tensor`.
  """
  decoded = image_bytes.dtype != tf.string
  shape = (tf.shape(image_bytes) if decoded
           else tf.image.extract_jpeg_shape(image_bytes))
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + crop_padding)) *
       tf.cast(tf.minimum(image_height, image_width), tf.float32)),
      tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  if decoded:
    image = tf.image.crop_to_bounding_box(
        image_bytes,
        offset_height=offset_height,
        offset_width=offset_width,
        target_height=padded_center_crop_size,
        target_width=padded_center_crop_size)
  else:
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

  image = resize_image(image_bytes=image,
                       height=image_size,
                       width=image_size)

  return image

def resize_image(image_bytes: tf.Tensor,
                 height: int = IMAGE_SIZE,
                 width: int = IMAGE_SIZE) -> tf.Tensor:
  """Resizes an image to a given height and width.

  Args:
    image_bytes: `Tensor` representing an image binary of arbitrary size.
    height: image height dimension.
    width: image width dimension.

  Returns:
    A tensor containing the resized image.

  """
  print(height, width)
  return tf.compat.v1.image.resize(
      image_bytes,
      tf.convert_to_tensor([height, width]),
      method=tf.image.ResizeMethod.BILINEAR,
      align_corners=False)
