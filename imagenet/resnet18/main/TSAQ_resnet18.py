import os
from tabnanny import verbose
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1' 
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from absl import logging
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.mixed_precision import experimental as mixed_precision
# from Trainer import *
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.optimizer.set_jit(True)

from utils.official_lr import build_learning_rate
from utils.official_optimizer import build_optimizer
from utils.official_optimizer import configure_optimizer
from models.resnet18 import resnet18
from official_data_loader.data_loader import DatasetBuilder

# import sparsification lib
from hat.build_net_graph.net_quantize import convert_model
from hat.build_loss.training_loss import Sparsification_Metrics

from hat.net_activation_sparse.utils.set_coefficients import apply_uniform_regularizer
from hat.net_quantization.callbacks.lsq_monitor import lsq_params_monitor
# tf.executing_eagerly()
# tf.config.experimental_run_functions_eagerly(True)
#speedup?

base_configs = {
    'mixed_precision': True,
                
    'origin_lr': 1e-3, #0.1, 1e-3
    'warmup_epochs': 5,

    'batch_size': 64,
    'input_shape': (224, 224, 3),
    'sparse_loss_coefficient': 0,
    
    'epoch_num': 500,
    # 'save_dir': "logs/resnet18",
    'pretrained_model': '../pretrained/rn18_ep_137-top1_0.6933-val_top1_0.7111.h5'
}

use_l2_regularizer = False

bits = 32
per_channel = False
is_symmetric = True
wq_mode = "uniform" # "uniform"

act_bits = 4
act_per_channel = False
act_mode = "LSQ"
calibrate = "" # "max_calibrate" # "percentile"
use_EMA_for_max = True
add_rnd_act_thresh = True

save_dir = "logs/LSQ_WA_resnet18_{}/lr_{}_stepwise-w-{}-{}_bits-per_c_{}-sym_{}-a-{}-{}_bits-per_c_{}-calibrate_{}".format(
    base_configs["origin_lr"], use_l2_regularizer, wq_mode, bits, per_channel, is_symmetric, act_mode, act_bits, act_per_channel, calibrate
)

os.makedirs(save_dir, exist_ok=True)

if __name__ == '__main__':
    dtype = tf.float32
    if base_configs['mixed_precision']:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        dtype = tf.float16

    strategy = tf.distribute.MirroredStrategy()
    num_devices = strategy.num_replicas_in_sync if strategy else 1
    print('number of devices: {}'.format(num_devices))

    BATCH_SIZE_PER_REPLICA = base_configs['batch_size']
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    print('global batch size: {}'.format(BATCH_SIZE))

    examples_per_epoch=1281167
    train_steps = int(examples_per_epoch/BATCH_SIZE)
    validation_steps = int(50000/BATCH_SIZE)
    print('train batch per epoch: {}'.format(train_steps))
    print('val batch per epoch: {}'.format(validation_steps))

    initial_lr = base_configs['origin_lr'] * (BATCH_SIZE/256)
    print('initial lr: {}'.format(initial_lr))

    label_smoothing = 0.1
    one_hot = label_smoothing and label_smoothing > 0
    print('label_smoothing: {}, one_hot: {}'.format(label_smoothing, one_hot))


    # load data
    train_data_iterator = DatasetBuilder(
        data_dir = "/media/DataDrive/zeqi/imagenet", 
        batch_size = BATCH_SIZE_PER_REPLICA,
        one_hot = one_hot,
        is_training = True, dtype=dtype).build(strategy)

    test_data_iterator = DatasetBuilder(
        data_dir = "/media/DataDrive/zeqi/imagenet", 
        batch_size = BATCH_SIZE_PER_REPLICA,
        one_hot = one_hot,
        is_training = False, dtype=dtype).build(strategy)

    logging.info('Running train and eval.')
    
    # TODO: whether it is necessary?
    from utils.setup import configure_cluster
    num_workers = configure_cluster()

    from utils.setup import get_strategy_scope
    strategy_scope = get_strategy_scope(strategy)
    logging.info('strategy_scope: {}'.format(strategy_scope))

    logging.info('Detected %d devices.',
                strategy.num_replicas_in_sync if strategy else 1)
    #############################################################

    with strategy_scope:
        # Load model
        img_input=Input(shape=base_configs['input_shape'])
        original_model = resnet18(num_classes=1000, use_l2_regularizer=True)
        original_model.load_weights(base_configs["pretrained_model"], by_name=True, skip_mismatch=True)
        
        # QAT model w/ hard thresh
        sparse_params = {
                    'mix_precision': base_configs['mixed_precision'],
                    'batch_size':    BATCH_SIZE,
                    'count_act_events': True,
                    'save_act_states':  False,
                    # spatial thresh
                    'add_act_thresh':   False,
                    'add_act_thresh_init': -2,
                    'act_thresh_training': "ste",
                    # quantization
                    'add_rnd_act_thresh': add_rnd_act_thresh,
                    'rnd_act_type': act_mode, #"po2_non_uniform",
                    'add_act_rnd_thresh_init': -1,
                    'activation_bits': act_bits, 
                    'activation_per_channel':  act_per_channel,
                    'act_calibrate': calibrate, 
                    'use_EMA_for_max': use_EMA_for_max,
                    'execute_act_calibration': True,
                    # regularization
                    'add_act_regularizer': False,
                    'act_regularizer_range': "",# "below_thresh",
                    'act_regularizer_position': "before_thresh",
                    'act_regularizer_type': "l1",
                    'graicore_mapping':  "",
                    
                }

        q_params = {
            "fold_bn": True,
            "q_target_layers": ["BatchNormalization"], 
            "q_sensitive_layers": [],
            "q_base_bits": bits,
            "layer_config": 
                {
                    "wq_mode": wq_mode,
                    "bits": bits,
                    "per_channel": per_channel,
                    "is_symmetric": is_symmetric,
                    "execute_w_calibration": True,
                    "use_EMA_for_max": use_EMA_for_max,
                },
            "q_ignore_layer_names": [],
            "q_position": "replace",
        }

        t_params = {
            "sparse_params": sparse_params,
            "t_target_layers": [
                'ReLU', 'LeakyReLU'], 
                # "InputLayer"],
                # "MaxPooling2D",
                # "GlobalAveragePooling2D"],
            "t_ignore_layer_names": [],
            "insert_type": "after_relu",
        }

        p_params = {
            "prune": False,
            "prune_metrics": "l2",
            "global_thresh":  1e-7, # 7.856e-5, # 3.90638e-5, # 
            # "target_sparsity": 0, 
            "show_sparsity": 0,
            "sensitive_layers": ["conv1"],
        }

        p_model = convert_model(
            model = original_model, 
            q_params = q_params,
            t_params = t_params,
            p_params = p_params
        )

        p_model.summary()
        model = p_model


        # learning_rate = build_learning_rate(
        #     initial_lr, BATCH_SIZE, 
        #     train_steps, 
        #     warmup_epochs=base_configs["warmup_epochs"], 
        #     examples_per_epoch=examples_per_epoch)
        
        params = {'decay': 0.1,
                'epsilon': 0.01,
                'momentum': 0.9,
                'nesterov': None}

        optimizer =build_optimizer(
                optimizer_name='momentum',
                base_learning_rate=initial_lr,
                params=params,
                model=model)
        
        loss_scale = 128. # or dynamic
        optimizer = configure_optimizer(
                optimizer,
                use_float16=dtype == 'float16',
                loss_scale=loss_scale)

        opt_metrics = Sparsification_Metrics(model)


        def _get_metrics(one_hot: bool):
            """Get a dict of available metrics to track."""
            if one_hot:
                return {
                    # (name, metric_fn)
                    'acc':
                        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    'accuracy':
                        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    'top_1':
                        tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    'top_5':
                        tf.keras.metrics.TopKCategoricalAccuracy(
                            k=5, name='top_5_accuracy'),
                    'import_metrics':
                        opt_metrics.import_metrics,
                    'sparse_loss':
                        opt_metrics.act_regularization_loss,
                    'sparse_events':
                        opt_metrics.num_sparse_events,
                    'dense_events':
                        opt_metrics.num_dense_events,
                    'activation_density':
                        opt_metrics.activation_density,
                    'current_coefficient':
                        opt_metrics.current_coefficient
                }
            else:
                return {
                    # (name, metric_fn)
                    'acc':
                        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                    'accuracy':
                        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                    'top_1':
                        tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                    'top_5':
                        tf.keras.metrics.SparseTopKCategoricalAccuracy(
                            k=5, name='top_5_accuracy'),
                }

        
        metrics_map = _get_metrics(one_hot)
        train_metrics = [
            'top_1', 'top_5', 
            'import_metrics',
            'sparse_loss', 
            'sparse_events', 
            'dense_events', 
            'activation_density',
            'current_coefficient'
        ]
        metrics = [metrics_map[metric] for metric in train_metrics]
        set_epoch_loop = False
        steps_per_loop = train_steps if set_epoch_loop else 1

        if one_hot:
            loss_obj = tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing)
        else:
            loss_obj = tf.keras.losses.SparseCategoricalCrossentropy()
        model.compile(
            optimizer=optimizer,
            loss=loss_obj,
            metrics=metrics,
            steps_per_execution=steps_per_loop,
            run_eagerly=False)


        apply_uniform_regularizer(
            model, base_configs['sparse_loss_coefficient'], 
            [], verbose=True)

        
        """ Create checkpoint
        """
        os.makedirs(save_dir, exist_ok=True)
        model_name = "ep_{epoch:03d}-top1_{accuracy:.4f}-val_top1_{val_accuracy:.4f}.h5"
        model_path = os.path.join(save_dir, model_name)
        checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            mode="max",
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
        )

        """ Create tensorboard
        """
        # Please use your own log folder (for tensorboard and model weights)
        tb_cb = tf.keras.callbacks.TensorBoard(
            log_dir=save_dir,
            histogram_freq=1,
            profile_batch="1000,1010",  # a few batches to inspect
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.1,
            patience=20,
            mode="max",
            verbose=1
        )

        # mt_cb = Monitor_threshold_Callback(verbose=True)


        lb = lsq_params_monitor(model, verbose=False)

        callbacks = [reduce_lr, checkpoint_cb, lb] # tb_cb]

        # calibration
        calibration_metrics = model.evaluate(
            test_data_iterator, 
            steps=validation_steps,
            verbose=1,
            callbacks=callbacks
        )
        print(calibration_metrics)

        # no calibration
        for layer in model.layers:
            if layer.__class__.__name__ in ["Sparse_QActivation"]:
                layer.is_calibration.assign(False)
                print(layer.name, layer.is_calibration)
        
        calibration_metrics = model.evaluate(
            test_data_iterator, 
            steps=validation_steps,
            verbose=1,
            callbacks=callbacks
        )
        print(calibration_metrics)


        history = model.fit(
            train_data_iterator,
            epochs=base_configs['epoch_num'],
            steps_per_epoch=train_steps, 
            callbacks=callbacks,
            verbose=1,
            validation_data=test_data_iterator,
            validation_steps=validation_steps)