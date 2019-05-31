# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from tensorflow.python.client import device_lib
import time
import datetime
import argparse
import os
import pandas as pd
import tensorflow as tf
import numpy as np
import sys
# sys.path.insert(0, "/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/afstudeerproject_KI/Jochem/3DGroupConv_adapted/")
sys.path.insert(0, "/home/jsoons/afstudeerproject_KI/Jochem/3DGroupConv_/")
import h5py
from abstract_reader import Reader
from dltk.networks.regression_classification.group_convnet import groupnet_3d,convnet_3d
from AbidaData import create_input_fn, write_subset_files, explore_data
from config import print_config
from plot import *
from deploy import predict_after_train

BATCH_SIZE = 8
NUM_CLASSES = 2
NUM_CHANNELS = 1
SHUFFLE_CACHE_SIZE = 32

def model_fn_group3d(features, labels, mode, params):
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

    # print("CREATE MODEL")
    # 1. create a model and its outputs
    net_output_ops = groupnet_3d(
        features['x'],
        batch_size=BATCH_SIZE,
        num_classes=NUM_CLASSES,
        mode=mode,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))

    # print("MODEL CREATED")
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

def model_fn_conv3d(features, labels, mode, params):
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
    net_output_ops = convnet_3d(
        features['x'],
        num_classes=NUM_CLASSES,
        mode=mode,
        kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-4))
    # print("Output", net_output_ops)
    # 1.1 Generate predictions only (for `ModeKeys.PREDICT`)
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=net_output_ops,
            export_outputs={'out': tf.estimator.export.PredictOutput(net_output_ops)})

    # 2. set up a loss function
    one_hot_labels = tf.reshape(tf.one_hot(labels['y'], depth=NUM_CLASSES), [-1, NUM_CLASSES])
    # print("Labels:", one_hot_labels)
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
    # np.random.seed(42)
    tf.set_random_seed(42)

    print('Setting up...')
    if args.resplit_data:
        print("Data is being resplitted...")
        data_file = h5py.File(args.data_path  + 'fmri_summary_abideI_II.hdf5', 'r')
        write_subset_files(data_file, args.data_path, args.test_ratio, args.train_val_ratio)
        print("Data resplitted.")
        print("#" * 80)

    if args.explore_data:
        data_file = h5py.File(args.data_path  + 'fmri_summary_abideI_II.hdf5', 'r')
        explore_data(data_file)
        print("#" * 80)

    train_f = h5py.File(args.train_file, 'r')
    val_f = h5py.File(args.val_file, 'r')

    nepochs = args.epochs
    BATCH_SIZE = int(args.batch_size)
    learning_rate = float(args.lr)
    NUM_CLASSES = args.num_classes
    NUM_CHANNELS = 1
    SHUFFLE_CACHE_SIZE = 32
    model_path = str(args.model_path + args.model + '/')
    training_size = len(train_f[args.summary])
    validation_size = len(val_f[args.summary])
    EVAL_EVERY_N_STEPS = int(training_size/ BATCH_SIZE)
    EVAL_STEPS = int(validation_size/ BATCH_SIZE)
    MAX_STEPS = EVAL_EVERY_N_STEPS * nepochs

    reader = Reader(dtypes={'features': {'x': tf.float32},
                     'labels': {'y': tf.int32}})

    # Get input functions for training and validation data
    train_input_fn = create_input_fn(train_f, args.summary, BATCH_SIZE, nepochs, shuffle=True)
    val_input_fn = create_input_fn(val_f, args.summary, BATCH_SIZE, nepochs, shuffle=False)

    # Instantiate the neural network estimator
    if args.model == 'conv3d':
        nn = tf.estimator.Estimator(
            model_fn=model_fn_conv3d,
            model_dir=args.model_path,
            params={"learning_rate": learning_rate},
            config=tf.estimator.RunConfig())

    elif args.model == 'group3d':
        nn = tf.estimator.Estimator(
            model_fn=model_fn_group3d,
            model_dir=args.model_path,
            params={"learning_rate": learning_rate},
            config=tf.estimator.RunConfig())

    start = time. time()
    print('\nStarting training... ({})'.format(datetime.datetime.now()))
    print('Training on {} samples, validating on {} samples\n'.format(training_size, validation_size))
    best_model_path_loss = os.path.join(model_path, 'best_loss')
    best_model_path_acc = os.path.join(model_path, 'best_acc')
    best_val_loss = None
    best_val_acc = None
    val_loss_list = []
    val_acc_list = []
    precision_list = []
    train_loss_list = []
    train_acc_list = []
    best_loss_dirs = []
    best_acc_dirs = []
    best_loss_epoch = None
    best_acc_epoch = None

    try:
        for epoch in range(MAX_STEPS // EVAL_EVERY_N_STEPS):
            nn.train(
                input_fn=train_input_fn,
                hooks=None,
                steps=EVAL_EVERY_N_STEPS)

            # Optional: keep track of training acc and loss
            if args.run_train_validation:
                results_train = nn.evaluate(
                    input_fn=train_input_fn,
                    hooks=None,
                    steps=EVAL_EVERY_N_STEPS)

                train_loss_list.append(results_train['loss'])
                train_acc_list.append(results_train['accuracy'])

            if args.run_validation:
                results_val = nn.evaluate(
                    input_fn=val_input_fn,
                    hooks=None,
                    steps=EVAL_STEPS)

                val_loss_list.append(results_val['loss'])
                val_acc_list.append(results_val['accuracy'])
                precision_list.append(results_val['precision'])

                if args.run_train_validation:
                    print('Epoch [{}/{}] train loss = {:.4f}, train acc = {:.4f}, val loss = {:.4f}; val acc = {:.4f}'.format(epoch+1,MAX_STEPS//EVAL_EVERY_N_STEPS,
                            results_train['loss'], results_train['accuracy'], results_val['loss'],results_val['accuracy']))
                else:
                    print('Epoch [{}/{}] val loss = {:.4f}; val acc = {:.4f}'.format(epoch+1,MAX_STEPS//EVAL_EVERY_N_STEPS,
                            results_val['loss'],results_val['accuracy']))

                if best_val_loss is None or results_val['loss'] < best_val_loss:
                    # os.system('rm -rf {}/{}'.format(best_model_path_loss,'*'))
                    export_dir = nn.export_savedmodel(
                        export_dir_base= os.path.join(model_path, 'best_loss'),
                        serving_input_receiver_fn=reader.serving_input_receiver_fn(
                            {'features': {'x': [45, 54, 45, NUM_CHANNELS]},
                             'labels': {'y': [1]}}))
                    print('Best Loss Model saved to {}.'.format(export_dir))
                    best_val_loss = results_val['loss']
                    best_loss_epoch = epoch + 1
                    best_loss_dirs.append(export_dir)

                if best_val_acc is None or results_val['accuracy'] > best_val_acc:
                    # os.system('rm -rf {}/{}'.format(best_model_path_acc,'*'))
                    export_dir = nn.export_savedmodel(
                        export_dir_base= os.path.join(model_path, 'best_acc'),
                        serving_input_receiver_fn=reader.serving_input_receiver_fn(
                            {'features': {'x': [45, 54, 45, NUM_CHANNELS]},
                             'labels': {'y': [1]}}))
                    print('Best Acc. Model saved to {}.'.format(export_dir))
                    best_val_acc = results_val['accuracy']
                    best_acc_epoch = epoch + 1
                    best_acc_dirs.append(export_dir)

            print()
    except KeyboardInterrupt:
        nepochs = epoch
        pass

    print("Done with training. lowest loss: {:.4f}, highest acc.: {:.4f}".format(best_val_loss, best_val_acc))

    # Plot accuracy and loss
    print("Creating plots...")
    if args.run_train_validation:
        plot_accuracy_train_val(args, nepochs, train_acc_list, val_acc_list, args.plot_store_path)
        plot_loss_train_val(args, nepochs, train_loss_list, val_loss_list, val_acc_list, args.plot_store_path)
    else:
        plot_accuracy_val(args, nepochs, val_acc_list, args.plot_store_path)
        plot_loss_val(args, nepochs, val_loss_list, val_acc_list, args.plot_store_path)
    end = time. time()
    print("Done.({})".format(datetime.datetime.now()))
    end = time. time()
    minutes = (end-start)
    print("Training time: {:.0f} minutes, {:.0f} seconds".format((end-start) // 60, (end-start) % 60))
    print("#" * 80)
    print("Starting testing phase...")
    print("Testing best loss model (loss = {:.4f} at epoch {})".format(best_val_loss, best_loss_epoch))
    if len(best_loss_dirs) > 1:
        predict_after_train(args, best_loss_dirs[-1])
        # predict_after_train(args, best_loss_dirs[-2])
    else:
        predict_after_train(args, best_loss_dirs[-1])
    print("\n")
    print("Testing best acc model (acc = {:.4f} at epoch {})".format(best_val_acc, best_acc_epoch))
    if len(best_acc_dirs) > 1:
        predict_after_train(args, best_acc_dirs[-1])
        # predict_after_train(args, best_acc_dirs[-2])
    else:
        predict_after_train(args, best_acc_dirs[-1])
    print("Done testing. Exiting program.")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='ASD classification training script')
    parser.add_argument('--run_validation', default=True)
    parser.add_argument('--run_train_validation', default=False, action='store_true')
    parser.add_argument('--explore_data', default=False, action='store_true')
    parser.add_argument('--restart', default=True, action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--resplit_data', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--summary', type=str, required=True)
    parser.add_argument('--model_path', '-p', default='../models/')
    parser.add_argument('--data_path', default='/home/jsoons/afstudeerproject_KI/Jochem/Datasets/')
    parser.add_argument('--plot_store_path', type=str, default='/home/jsoons/afstudeerproject_KI/Jochem/3DGroupConv_/plots/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--train_file', default='/home/jsoons/afstudeerproject_KI/Jochem/Datasets/train.hdf5')
    # parser.add_argument('--train_file', default='/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/train.hdf5')
    parser.add_argument('--val_file', default= '/home/jsoons/afstudeerproject_KI/Jochem/Datasets/validation.hdf5')
    # parser.add_argument('--val_file', default= '/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/validation.hdf5')
    parser.add_argument('--test_file', default= '/home/jsoons/afstudeerproject_KI/Jochem/Datasets/test.hdf5')
    # parser.add_argument('--test_file', default= '/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/test.hdf5')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='ratio that defines size of test set')
    parser.add_argument('--train_val_ratio', type=float, default=0.8, help='ratio of train/val set sizes')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
    parser.add_argument('--num_classes', type=int, default=2)
    args = parser.parse_args()
    print_config(args)
    print("#" * 80)
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
        os.system('mkdir -p {}/{}'.format(args.model_path, args.model))
    else:
        print('Resuming training on model_path {}'.format(args.model_path))
    # Call training
    train(args)