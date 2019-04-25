"""
A class for creating an abide dataset object as is done with the rest of the
datasets. I used as a refrences the blender.py script will see if we need other
stuff as well from the other functions later.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import numpy as np
import h5py

import torch
from torch.utils.data import Dataset

class AbideDataset(Dataset):
    """ root_path = Path to directionary where the data can be found
        subset = Name of the dataset we use, we don't yet have split the data so
        this comes later, so right now I put it standard to train to test.
        data_to_train = Says on which part of the summaries the CNN should train,
        Standard is the T1 input, aka the first input. Later we should make it
        possible to loop over more parts of the summaries list.
    """
    def __init__(self, root_path, subset='train',data_to_train='T1', spatial_transform=None,
                 temporal_transform=None, target_transform=None):

        assert temporal_transform is None, 'Temporal transform not supported for AbideDataset'
        assert target_transform is None,   'Target transform not supported for AbideDataset'
        assert subset in ('train', 'validation', 'test')

        self._root_path = root_path
        self._subset = subset
        self._data_to_train = data_to_train

        self._spatial_transform = spatial_transform
        self._temporal_transform = temporal_transform
        self._target_transform = target_transform

        self._make_dataset()

    # Get's the standard data of the abide dataset.
    def _make_dataset(self):

        self._data_files = glob.glob(os.path.join(self._root_path, '{}.hdf5'.format(self._subset)))

        # Get the number of samples
        with h5py.File(self._data_files[0]) as hf:
            self._num_examples_per_file = len(hf[self._data_to_train])
            self._num_examples = len(self._data_files)*self._num_examples_per_file


        # We have two classe
        self._classes = set([0,1])

        print('  Number of {} HDF5 files found: {}'.format(self._subset, len(self._data_files)))
        print('  Number of {} examples found:   {}'.format(self._subset, len(self)))
        print('  Number of {} targets found:    {}'.format(self._subset, len(self._classes)))

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):

        # Get the asked example from the datafile from the wanted dataset
        with h5py.File(self._data_files[0], 'r') as hf:
            img = hf[self._data_to_train][idx]
            target = hf['labels'][idx]

        img = torch.from_numpy(img)
        target = torch.from_numpy(np.asarray(target))
        return img, target

    @property
    def classes(self):
        return self._classes

    @property
    def subset(self):
        return self._subset

    @property
    def num_classes(self):
        return len(self._classes)

    @property
    def target_offset(self):
        return self._target_offset


# You can run/test this file using the below adapt the root path though!
abide = AbideDataset("/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis", subset='val')
