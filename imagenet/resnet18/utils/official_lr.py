import tensorflow as tf
from absl import logging
from typing import Any, Mapping, Optional

def build_learning_rate(base_lr, batch_size, train_steps, warmup_epochs=5, examples_per_epoch=1281167):
    """Build the learning rate given the provided configuration."""
    decay_rate = None
    decay_steps = 0
    boundaries = [30, 60, 80]
    multipliers = [0.000390625, 3.90625e-05, 3.90625e-06, 3.90625e-07]

    if warmup_epochs is not None:
        warmup_steps = warmup_epochs * train_steps
    else:
        warmup_steps = 0

    # 'stepwise':
    steps_per_epoch = examples_per_epoch // batch_size
    boundaries = [boundary * steps_per_epoch for boundary in boundaries]
    multipliers = [batch_size * multiplier for multiplier in multipliers]
    logging.info(
        'Using stepwise learning rate. Parameters: '
        'boundaries: %s, values: %s', boundaries, multipliers)
    lr = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=multipliers)

    if warmup_steps > 0:
        logging.info('Applying %d warmup steps to the learning rate',
                    warmup_steps)
        lr = WarmupDecaySchedule(
            lr, warmup_steps, warmup_lr=base_lr)
    return lr


class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""

  def __init__(self,
               lr_schedule: tf.keras.optimizers.schedules.LearningRateSchedule,
               warmup_steps: int,
               warmup_lr: Optional[float] = None):
    """Add warmup decay to a learning rate schedule.

    Args:
      lr_schedule: base learning rate scheduler
      warmup_steps: number of warmup steps
      warmup_lr: an optional field for the final warmup learning rate. This
        should be provided if the base `lr_schedule` does not contain this
        field.
    """
    super(WarmupDecaySchedule, self).__init__()
    self._lr_schedule = lr_schedule
    self._warmup_steps = warmup_steps
    self._warmup_lr = warmup_lr

  def __call__(self, step: int):
    lr = self._lr_schedule(step)
    if self._warmup_steps:
      if self._warmup_lr is not None:
        initial_learning_rate = tf.convert_to_tensor(
            self._warmup_lr, name="initial_learning_rate")
      else:
        initial_learning_rate = tf.convert_to_tensor(
            self._lr_schedule.initial_learning_rate,
            name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      global_step_recomp = tf.cast(step, dtype)
      warmup_steps = tf.cast(self._warmup_steps, dtype)
      warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
      lr = tf.cond(global_step_recomp < warmup_steps, lambda: warmup_lr,
                   lambda: lr)
    # tf.print("lr: ", lr)
    return lr

  def get_config(self) -> Mapping[str, Any]:
    config = self._lr_schedule.get_config()
    config.update({
        "warmup_steps": self._warmup_steps,
        "warmup_lr": self._warmup_lr,
    })
    return config



BASE_LEARNING_RATE = 0.1

# class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
#   """Class to generate learning rate tensor."""

#   def __init__(self, batch_size: int, total_steps: int, warmup_steps: int):
#     """Creates the cosine learning rate tensor with linear warmup.

#     Args:
#       batch_size: The training batch size used in the experiment.
#       total_steps: Total training steps.
#       warmup_steps: Steps for the warm up period.
#     """
#     super(CosineDecayWithWarmup, self).__init__()
#     base_lr_batch_size = 256
#     self._total_steps = total_steps
#     self._init_learning_rate = BASE_LEARNING_RATE * batch_size / base_lr_batch_size
#     self._warmup_steps = warmup_steps

#   def __call__(self, global_step: int):
#     global_step = tf.cast(global_step, dtype=tf.float32)
#     warmup_steps = self._warmup_steps
#     init_lr = self._init_learning_rate
#     total_steps = self._total_steps

#     linear_warmup = global_step / warmup_steps * init_lr

#     cosine_learning_rate = init_lr * (tf.cos(np.pi *
#                                              (global_step - warmup_steps) /
#                                              (total_steps - warmup_steps)) +
#                                       1.0) / 2.0

#     learning_rate = tf.where(global_step < warmup_steps, linear_warmup,
#                              cosine_learning_rate)
#     return learning_rate

#   def get_config(self):
#     return {
#         "total_steps": self._total_steps,
#         "warmup_learning_rate": self._warmup_learning_rate,
#         "warmup_steps": self._warmup_steps,
#         "init_learning_rate": self._init_learning_rate,
#     }
