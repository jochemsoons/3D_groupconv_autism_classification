# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import h5py

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib import predictor

from dltk.io.augmentation import extract_random_example_array
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from plot import plot_roc_auc

def predict(args):
    test_model = '1558805223'
    test_f = h5py.File(args.test_file, 'r')
    images = test_f[args.summary]
    labels_ = test_f['labels']

    # From the model_path, parse the latest saved model and restore a
    # predictor from it
    #export_dir = \
        #[os.path.join(args.model_path, o) for o in sorted(os.listdir(args.model_path))
         #if os.path.isdir(os.path.join(args.model_path, o)) and o.isdigit()][-1]
    model_path = str(args.model_path + args.model + '/')
    export_dir = os.path.join(model_path, 'best_loss/{}/'.format(test_model))
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Iterate through data and labels and fetch results.
    accuracy = []
    labels = np.empty([], dtype=int)
    predictions = np.empty([], dtype=int)

    for img, lbl in zip(images, labels_):
        t0 = time.time()

        img = np.expand_dims(img,0)

        y_ = my_predictor.session.run(
            fetches=my_predictor._fetch_tensors['y_prob'],
            feed_dict={my_predictor._feed_tensors['x']: img})

        # Average the predictions on the test inputs:
        y_ = np.mean(y_, axis=0)
        predicted_class = np.argmax(y_)
        labels = np.append(labels, lbl)
        predictions = np.append(predictions, predicted_class)

        # Calculate the accuracy for this subject
        accuracy.append(predicted_class == lbl)

        # Print outputs
        # print('prob={}, pred={}; true={}; run time={:0.2f} s; '
        #         ''.format(y_, predicted_class, lbl, time.time() - t0))

    fpr, tpr, threshold = sklearn.metrics.roc_curve(np.array(labels[1:]), np.array(predictions[1:]))
    cm1 = confusion_matrix(labels[1:] , predictions[1:])
    print('accuracy={}'.format(np.mean(accuracy)))
    print('sens:', cm1[0,0]/(cm1[0,0]+cm1[0,1]))
    print('spec:', cm1[1,1]/(cm1[1,0]+cm1[1,1]) )
    print("Plotting roc/auc curve...")
    plot_roc_auc(args, np.mean(accuracy), fpr, tpr)


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='PAC age prediction test script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path', '-p', default='../models/')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--summary', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--plot_store_path', type=str, default='/home/jsoons/afstudeerproject_KI/Jochem/3DGroupConv_/plots/')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--test_file', default= '/home/jsoons/afstudeerproject_KI/Jochem/Datasets/test.hdf5')
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

    # Call training
    predict(args)
