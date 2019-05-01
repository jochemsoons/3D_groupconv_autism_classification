# A script to load images and make batch.
# Dependency: 'nibabel' to load MRI (NIFTI) images
# Reference: http://blog.naver.com/kjpark79/220783765651

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
# import nibabel as nib
import h5py
import glob
from keras.utils.np_utils import to_categorical

width = 45
height = 54
depth = 45# 3
batch_index = 0
filenames = {}

# user selection
# data_dir = '/home/lsalomons'
data_dir = '/home/lisasalomons/Desktop'
num_class = 2

def get_filenames(data_set):
    global filenames
    labels = []
    filenames.update({data_set:glob.glob(os.path.join(data_dir, '{}.hdf5'.format(data_set)))})
    # with h5py.File(data_file) as hf:
        # labels = hf['labels']
    # with open(data_dir + '/train.hdf5') as f:
    #     for line in f:
    #         inner_list = [elt.strip() for elt in line.split(',')]
    #         labels += inner_list
    #
    # for i, label in enumerate(labels):
    #     list = os.listdir(data_dir  + '/' + data_set + '/' + label)
    #     for filename in list:
    #         filenames.append([label + '/' + filename, i])
    #
    # random.shuffle(filenames)


def get_data_MRI(sess, data_set, batch_size, summary):
    global batch_index, filenames
    # See if the data_set was already looked up or not
    try:
        filenames[data_set]
    except:
        get_filenames(data_set)

    # Open the dataset file
    with h5py.File(filenames[data_set][0]) as hf:
        max = len(hf[summary])

        begin = batch_index
        end = batch_index + batch_size

        # Make sur the end doesn't exceed the amount of samples
        if end >= max:
            end = max
            batch_index = 0

        x_data = np.array([], np.float32)
        # y_data = np.zeros((batch_size, num_class)) # zero-filled list for 'one hot encoding'
        index = 0

        # Retreive the amount of batch data samples
        x_data = hf[summary][begin:end]

        # Get the labels and transform them to one hot encoding
        y_data = hf['labels'][begin:end]
        y_data= to_categorical(y_data, num_classes=2)

        batch_index += batch_size  # update index for the next batch
        # Reshape the data into number of batch samples per row in the array
        x_data_ = x_data.reshape(x_data.shape[0], height * width * depth)
        # print(x_data.shape)
        hf.close()

    return x_data_, y_data
