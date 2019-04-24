import h5py
import numpy as np

train_f = h5py.File('/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/train.hdf5', 'r')
val_f = h5py.File('/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/val.hdf5', 'r')
test_f = h5py.File('/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/test.hdf5', 'r')

print("Found {} train examples of shape {}".format(len(train_f['T1']), train_f
['T1'][0].shape))

print("Found {} validation examples of shape {}".format(len(val_f['T1']), val_f['T1'][0].shape))

print("Found {} test examples of shape {}".format(len(test_f['T1']), test_f
['T1'][0].shape))