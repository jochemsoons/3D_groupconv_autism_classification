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
import h5py

# sys.path.append('/home/jsoons/afstudeerproject_KI/Jochem/3DGroupConv_')
sys.path.insert(0, "/home/jsoons/afstudeerproject_KI/Jochem/3DGroupConv_/code")
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
# sys.path.append('/home/jsoons/afstudeerproject_KI/Jochem/3DGroupConv_')
# print(sys.path)
from dltk.io.abstract_reader import Reader
from dltk.networks.regression_classification.group_convnet import groupnet_3d,convnet_3d
from reader import read_fn
from AbidaData import create_input_fn, write_subset_files



nepochs = 100
dataset_size = 2122
training_size = 0.7 * dataset_size
validation_size = 0.1 * dataset_size
BATCH_SIZE = 8
EVAL_EVERY_N_STEPS = int(training_size/ BATCH_SIZE)
EVAL_STEPS = int(validation_size/ BATCH_SIZE)
MAX_STEPS = EVAL_EVERY_N_STEPS * nepochs
NUM_CLASSES = 2
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
        filters=(8, 8, 16, 16),
        mode=mode,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(net_output_ops)})

    # 2. set up a loss function
    one_hot_labels = tf.reshape(tf.one_hot(labels['y'], depth=NUM_CLASSES), [-1, NUM_CLASSES])

    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=net_output_ops['logits'])

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

    expected_output_size = [1, 54, 45, 1]  # [B, W, H, C]
    [tf.summary.image(name, tf.reshape(image, expected_output_size))
     for name, image in my_image_summaries.items()]

    # 4.2 (optional) track the rmse (scaled back by 100, see reader.py)
    acc = tf.metrics.accuracy
    prec = tf.metrics.precision
    eval_metric_ops = {"accuracy": acc(labels['y'], net_output_ops['y_']),
                       "precision": prec(labels['y'], net_output_ops['y_'])}

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
    if args.resplit_data:
        data_file = h5py.File(DATA_PATH  + 'fmri_summary_abideI_II.hdf5', 'r')
        write_subset_files(data_file, DATA_PATH, args.summary, args.test_ratio, args.train_val_ratio)

    train_f = h5py.File(args.train_file, 'r')
    val_f = h5py.File(args.val_file, 'r')

    # Get input functions and queue initialisation hooks for training and
    # validation data
    train_input_fn = create_input_fn(train_f, args.summary, BATCH_SIZE, nepochs)
    val_input_fn = create_input_fn(val_f, args.summary, BATCH_SIZE, nepochs)

    # Instantiate the neural network estimator
    nn = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_path,
        params={"learning_rate": 0.0005},
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
                print('val loss = {:.5f}; val acc = {:.5}'.format(
                    results_val['loss'],results_val['accuracy']))
                if best_val_loss is None or results_val['loss'] < best_val_loss:
                    os.system('rm -rf {}/{}'.format(best_model_path,'*'))
                    export_dir = nn.export_savedmodel(
                        export_dir_base= os.path.join(args.model_path,'best'),
                        serving_input_receiver_fn=reader.serving_input_receiver_fn(
                            {'features': {'x': [45, 54, 45, NUM_CHANNELS]},
                             'labels': {'y': [1]}}))
                    print('Best Model saved to {}.'.format(export_dir))
                    best_val_loss = results_val['loss']
            print()
    except KeyboardInterrupt:
        pass

    # When exporting we set the expected input shape to be arbitrary.
    export_dir = nn.export_savedmodel(
        export_dir_base=args.model_path,
        serving_input_receiver_fn=reader.serving_input_receiver_fn(
            {'features': {'x': [45, 54, 45, NUM_CHANNELS]},
             'labels': {'y': [1]}}))
    print('Model saved to {}.'.format(export_dir))


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='ASD classification training script')
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--restart', default=True, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--resplit_data', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--summary', type=str, required=True, default='T1')
    parser.add_argument('--model_path', '-p', default='../models/Abide_summaries')
    parser.add_argument('--data_path', default='/home/jsoons/afstudeerproject_KI/Jochem/Datasets/')
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--train_file', default='/home/jsoons/afstudeerproject_KI/Jochem/Datasets/train.hdf5')
    parser.add_argument('--val_file', default= '/home/jsoons/afstudeerproject_KI/Jochem/Datasets/validation.hdf5')
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
