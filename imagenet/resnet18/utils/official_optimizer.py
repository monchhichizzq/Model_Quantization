from typing import Any, Dict, Optional, Text
from absl import logging
import tensorflow as tf
import tensorflow_addons as tfa

params = {'decay': 0.1,
          'epsilon': 0.01,
          'momentum': 0.9,
          'nesterov': None}

def build_optimizer(
    optimizer_name: Text,
    base_learning_rate: tf.keras.optimizers.schedules.LearningRateSchedule,
    params: Dict[Text, Any],
    model: Optional[tf.keras.Model] = None):
    """Build the optimizer based on name.

    Args:
        optimizer_name: String representation of the optimizer name. Examples: sgd,
        momentum, rmsprop.
        base_learning_rate: `tf.keras.optimizers.schedules.LearningRateSchedule`
        base learning rate.
        params: String -> Any dictionary representing the optimizer params. This
        should contain optimizer specific parameters such as `base_learning_rate`,
        `decay`, etc.
        model: The `tf.keras.Model`. This is used for the shadow copy if using
        `ExponentialMovingAverage`.

    Returns:
        A tf.keras.Optimizer.

    Raises:
        ValueError if the provided optimizer_name is not supported.

    """
    optimizer_name = optimizer_name.lower()
    logging.info('Building %s optimizer with params %s', optimizer_name, params)

    if optimizer_name == 'sgd':
        logging.info('Using SGD optimizer')
        nesterov = params.get('nesterov', False)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=base_learning_rate, nesterov=nesterov)
    elif optimizer_name == 'momentum':
        logging.info('Using momentum optimizer')
        nesterov = params.get('nesterov', False)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=base_learning_rate,
            momentum=params['momentum'],
            nesterov=nesterov)
    elif optimizer_name == 'rmsprop':
        logging.info('Using RMSProp')
        rho = params.get('decay', None) or params.get('rho', 0.9)
        momentum = params.get('momentum', 0.9)
        epsilon = params.get('epsilon', 1e-07)
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=base_learning_rate,
            rho=rho,
            momentum=momentum,
            epsilon=epsilon)
    elif optimizer_name == 'adam':
        logging.info('Using Adam')
        beta_1 = params.get('beta_1', 0.9)
        beta_2 = params.get('beta_2', 0.999)
        epsilon = params.get('epsilon', 1e-07)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=base_learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon)
    elif optimizer_name == 'adamw':
        logging.info('Using AdamW')
        weight_decay = params.get('weight_decay', 0.01)
        beta_1 = params.get('beta_1', 0.9)
        beta_2 = params.get('beta_2', 0.999)
        epsilon = params.get('epsilon', 1e-07)
        optimizer = tfa.optimizers.AdamW(
            weight_decay=weight_decay,
            learning_rate=base_learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon)
    else:
        raise ValueError('Unknown optimizer %s' % optimizer_name)

    if params.get('lookahead', None):
        logging.info('Using lookahead optimizer.')
        optimizer = tfa.optimizers.Lookahead(optimizer)

    # # Moving average should be applied last, as it's applied at test time
    # moving_average_decay = params.get('moving_average_decay', 0.)
    # if moving_average_decay is not None and moving_average_decay > 0.:
    #     if model is None:
    #         raise ValueError(
    #             '`model` must be provided if using `ExponentialMovingAverage`.')
    #     logging.info('Including moving average decay.')
    #     optimizer = optimization.ExponentialMovingAverage(
    #         optimizer=optimizer, average_decay=moving_average_decay)
    #     optimizer.shadow_copy(model)
    return optimizer



def configure_optimizer(optimizer,
                        use_float16=False,
                        loss_scale=None,
                        use_graph_rewrite=None):
    """Configures optimizer object with performance options."""
    if use_graph_rewrite is not None:
        logging.warning('`use_graph_rewrite` is deprecated inside '
                        '`configure_optimizer`. Please remove the usage.')
    del use_graph_rewrite
    if use_float16:
        if loss_scale in (None, 'dynamic'):
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)
        else:
            # loss_scale is a number. We interpret that as a fixed loss scale.
            optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer, dynamic=False, initial_scale=loss_scale)
    return optimizer


def set_mixed_precision_policy(dtype, loss_scale=None):
    """Sets the global `tf.keras.mixed_precision.Policy`."""
    # TODO(b/191894773): Remove loss_scale argument
    assert loss_scale is None, (
        'The loss_scale argument must be None. The argument exists for '
        'historical reasons and will be removed soon.')
    if dtype == tf.float16:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    elif dtype == tf.bfloat16:
        tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    elif dtype == tf.float32:
        tf.keras.mixed_precision.set_global_policy('float32')
    else:
        raise ValueError('Unexpected dtype: %s' % dtype)