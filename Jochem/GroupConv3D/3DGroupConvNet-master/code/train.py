# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import sys
sys.path.append('/data/agelgazzar/Work/AgePrediction/3DResnet/dltk') 
from dltk.io.abstract_reader import Reader
from dltk.networks.regression_classification.group_convnet import groupnet_3d
from reader import read_fn



nepochs = 100
dataset_size = 2640
training_size = 0.7 * dataset_size
validation_size = 0.1 * dataset_size
BATCH_SIZE = 8
EVAL_EVERY_N_STEPS = int(training_size/ BATCH_SIZE)
EVAL_STEPS = int(validation_size/ BATCH_SIZE)
MAX_STEPS = EVAL_EVERY_N_STEPS * nepochs
NUM_CLASSES = 1
NUM_CHANNELS = 1
SHUFFLE_CACHE_SIZE = 32


def model_fn(features, labels, mode, params):
    """Model function to construct a tf.estimator.EstimatorSpec. It creates a
        network given input features (e.g. from a dltk.io.abstract_reader) and
        training targets (labels). Further, loss, optimiser, evaluation ops and
        custom tensorboard summary ops can be added. For additional information,
         please refer to https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#model_fn.

    Args:
        features (tf.Tensor): Tensor of input features to train from. Required
            rank and dimensions are determined by the subsequent ops
            (i.e. the network).
        labels (tf.Tensor): Tensor of training targets or labels. Required rank
            and dimensions are determined by the network output.
        mode (str): One of the tf.estimator.ModeKeys: TRAIN, EVAL or PREDICT
        params (dict, optional): A dictionary to parameterise the model_fn
            (e.g. learning_rate)

    Returns:
        tf.estimator.EstimatorSpec: A custom EstimatorSpec for this experiment
    """

    # 1. create a model and its outputs
    net_output_ops = groupnet_3d(
        features['x'],
        num_classes=NUM_CLASSES,
        filters=(2, 4, 8, 16, 32),
        strides=((2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)),
        mode=mode,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(net_output_ops)})

    # 2. set up a loss function
    loss = tf.losses.mean_squared_error(
        labels=labels['y'],
        predictions=net_output_ops['logits'])

    # 3. define a training op and ops for updating moving averages (i.e. for
    # batch normalisation)
    global_step = tf.train.get_global_step()
    optimiser = tf.train.AdamOptimizer(
        learning_rate=params["learning_rate"],
        epsilon=1e-5)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimiser.minimize(loss, global_step=global_step)

    # 4.1 (optional) create custom image summaries for tensorboard
    my_image_summaries = {}
    my_image_summaries['feat_t1'] = features['x'][0, 32, :, :, 0]

    expected_output_size = [1, 121, 145, 1]  # [B, W, H, C]
    [tf.summary.image(name, tf.reshape(image, expected_output_size))
     for name, image in my_image_summaries.items()]

    # 4.2 (optional) track the rmse (scaled back by 100, see reader.py)
    rmse = tf.metrics.root_mean_squared_error
    mae = tf.metrics.mean_absolute_error
    eval_metric_ops = {"rmse": rmse(labels['y'], net_output_ops['logits']),
                       "mae": mae(labels['y'], net_output_ops['logits'])}

    # 5. Return EstimatorSpec object
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=net_output_ops,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops)


def train(args):
    np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up...')

    # Parse csv files for file names

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    # Set up a data reader to handle the file i/o.
    reader_params = {'n_examples': 2,
                     'example_size': [121, 145, 121],
                     'extract_examples': False}

    reader_example_shapes = {'features': {'x': reader_params['example_size'] + [NUM_CHANNELS]},
                             'labels': {'y': [1]}}
    reader = Reader(read_fn,
                    {'features': {'x': tf.float32},
                     'labels': {'y': tf.float32}})

    # Get input functions and queue initialisation hooks for training and
    # validation data
    train_input_fn, train_qinit_hook = reader.get_inputs(
        file_references=train_df,
        mode=tf.estimator.ModeKeys.TRAIN,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    val_input_fn, val_qinit_hook = reader.get_inputs(
        file_references=val_df,
        mode=tf.estimator.ModeKeys.EVAL,
        example_shapes=reader_example_shapes,
        batch_size=BATCH_SIZE,
        shuffle_cache_size=SHUFFLE_CACHE_SIZE,
        params=reader_params)

    # Instantiate the neural network estimator
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_path,
        params={"learning_rate": 0.001},
        config=tf.estimator.RunConfig())

    # Hooks for validation summaries
    val_summary_hook = tf.contrib.training.SummaryAtEndHook(
        os.path.join(args.model_path, 'eval'))
    step_cnt_hook = tf.train.StepCounterHook(every_n_steps=EVAL_EVERY_N_STEPS,
                                             output_dir=args.model_path)

    print('Starting training...')

    best_model_path = os.path.join(args.model_path, 'best')
    best_val_loss = None

    try:
        for _ in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            print('Training epoch {}/{}'.format(_,MAX_STEPS//EVAL_EVERY_N_STEPS))
            nn.train(
                input_fn=train_input_fn,
                hooks=[train_qinit_hook, step_cnt_hook],
                steps=EVAL_EVERY_N_STEPS)

            if args.run_validation:
                results_val = nn.evaluate(
                    input_fn=val_input_fn,
                    hooks=[val_qinit_hook, val_summary_hook],
                    steps=EVAL_STEPS)
                print('Step = {}; val loss = {:.5f};'.format(
                    results_val['global_step'],
                    results_val['loss']))
                if best_val_loss is None or results_val['loss'] < best_val_loss:
                    os.system('rm -rf {}/{}'.format(best_model_path,'*'))
                    export_dir = nn.export_savedmodel(
                        export_dir_base= os.path.join(args.model_path,'best'),
                        serving_input_receiver_fn=reader.serving_input_receiver_fn(
                            {'features': {'x': [None, None, None, NUM_CHANNELS]},
                             'labels': {'y': [1]}}))
                    print('Best Model saved to {}.'.format(export_dir))
                    best_val_loss = results_val['loss']

    except KeyboardInterrupt:
        pass

    # When exporting we set the expected input shape to be arbitrary.
    export_dir = nn.export_savedmodel(
        export_dir_base=args.model_path,
        serving_input_receiver_fn=reader.serving_input_receiver_fn(
            {'features': {'x': [None, None, None, NUM_CHANNELS]},
             'labels': {'y': [1]}}))
    print('Model saved to {}.'.format(export_dir))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='PAC Age regression training script')
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--restart', default=True, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path', '-p', default='../models/group_net_O')
    parser.add_argument('--train_csv', default='csvfiles/PAC_train.csv')
    parser.add_argument('--val_csv', default= 'csvfiles/PAC_validate.csv')
    args = parser.parse_args()

    # Set verbosity
    if args.verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
        tf.logging.set_verbosity(tf.logging.INFO)
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.logging.set_verbosity(tf.logging.ERROR)

    # GPU allocation options
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # Handle restarting and resuming training
    if args.restart:
        print('Restarting training from scratch.')
        os.system('rm -rf {}'.format(args.model_path))

    if not os.path.isdir(args.model_path):
        os.system('mkdir -p {}/{}'.format(args.model_path,'best'))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))

    # Call training
    train(args)
