import h5py
import numpy as np

train_f = h5py.File('/home/lisasalomons/Desktop/afstudeerproject_KI/Lisa/train.hdf5', 'r')
val_f = h5py.File('/home/lisasalomons/Desktop/afstudeerproject_KI/Lisa/val.hdf5', 'r')
test_f = h5py.File('/home/lisasalomons/Desktop/afstudeerproject_KI/Lisa/test.hdf5', 'r')

print("Found {} train examples of shape {}".format(len(train_f['autocorr']), train_f
['autocorr'][0].shape))

print("Found {} validation examples of shape {}".format(len(val_f['autocorr']), val_f['autocorr'][0].shape))

print("Found {} test examples of shape {}".format(len(test_f['autocorr']), test_f
['autocorr'][0].shape))
