# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function

import argparse
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.contrib import predictor

from dltk.io.augmentation import extract_random_example_array
import matplotlib.pyplot as plt

from reader import read_fn
from sklearn.metrics import r2_score
READER_PARAMS = {'extract_examples': False}
N_VALIDATION_SUBJECTS = int(0.2 * 2640)
from sklearn.metrics import confusion_matrix


def predict(args):


    test_df = pd.read_csv(args.test_csv)


    # From the model_path, parse the latest saved model and restore a
    # predictor from it
    #export_dir = \
        #[os.path.join(args.model_path, o) for o in sorted(os.listdir(args.model_path))
         #if os.path.isdir(os.path.join(args.model_path, o)) and o.isdigit()][-1]
    export_dir = os.path.join(args.model_path, 'best/1557349239/')
    print('Loading from {}'.format(export_dir))
    my_predictor = predictor.from_saved_model(export_dir)

    # Iterate through the files, predict on the full volumes and compute a Dice
    # coefficient
    accuracy = []
    labels = np.empty([], dtype=int)
    pred = np.empty([], dtype=int)
    for output in read_fn(test_df,
                          mode=tf.estimator.ModeKeys.EVAL,
                          params=READER_PARAMS):
        t0 = time.time()

        # Parse the read function output and add a dummy batch dimension as
        # required
        img = output['features']['x']
        lbl = output['labels']['y']
        test_id = output['img_id']

        # We know, that the training input shape of [64, 96, 96] will work with
        # our model strides, so we collect several crops of the test image and
        # average the predictions. Alternatively, we could pad or crop the input
        # to any shape that is compatible with the resolution scales of the
        # model:
        img = np.expand_dims(img,0)
        '''num_crop_predictions = 4
        crop_batch = extract_random_example_array(
            image_list=img,
            example_size=[64, 96, 96],
            n_examples=num_crop_predictions)'''

        y_ = my_predictor.session.run(
            fetches=my_predictor._fetch_tensors['y_prob'],
            feed_dict={my_predictor._feed_tensors['x']: img})

        # Average the predictions on the cropped test inputs:
        y_ = np.mean(y_, axis=0)
        predicted_class = np.argmax(y_)
        labels = np.append(labels,lbl)
        pred = np.append(pred,predicted_class)

        # Calculate the accuracy for this subject
        accuracy.append(predicted_class == lbl)

        # Print outputs
        print('id={}; pred={}; true={}; run time={:0.2f} s; '
              ''.format(test_id, predicted_class, lbl[0], time.time() - t0))
    print('accuracy={}'.format(np.mean(accuracy)))
    cm1 = confusion_matrix(labels[1:] , pred[1:])

    print('sens:', cm1[0,0]/(cm1[0,0]+cm1[0,1]))
    print('spec:', cm1[1,1]/(cm1[1,0]+cm1[1,1]) )


if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='PAC age prediction test script')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--cuda_devices', '-c', default='0')
    parser.add_argument('--model_path', '-p', default='/data/agelgazzar/Work/AgePrediction/Two_obj_3DResnet/models/group_net_O/')
    parser.add_argument('--test_csv', default='/data/agelgazzar/Work/AgePrediction/3DResnet/code/csvfiles/PAC_test.csv')

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
