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
    def __init__(self, root_path, subset, summarie):

        assert subset in ('train', 'val', 'test')

        self._root_path = root_path
        self._subset = subset
        self._data_to_train = summarie

        self._make_dataset()

    # Get's the standard data of the abide dataset.
    def _make_dataset(self):

        self._data_files = glob.glob(os.path.join(self._root_path, '{}.hdf5'.format(self._subset)))
        # Get the number of samples
        with h5py.File(self._data_files[0]) as hf:
            self._num_examples_per_file = len(hf[self._data_to_train])
            self._num_examples = len(self._data_files)*self._num_examples_per_file


        # We have two classes
        self._classes = set([0,1])

        print('  Number of {} HDF5 files found: {}'.format(self._subset, len(self._data_files)))
        print('  Number of {} examples found:   {}'.format(self._subset, len(self)))
        print('  Number of {} targets found:    {}'.format(self._subset, len(self._classes)))
        print(' ')

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):

        container_idx = idx // self._num_examples_per_file
        example_idx = (idx-(container_idx*self._num_examples_per_file))

        # Get the asked example from the datafile from the wanted dataset
        with h5py.File(self._data_files[container_idx], 'r') as hf:
            pic = hf[self._data_to_train][example_idx]
            target =  hf['labels'][example_idx]

        # Convert both numpy arrays to tensors
        pic = torch.from_numpy(pic)
        pic = pic.squeeze(-1)
        pic = pic.unsqueeze(0)
        target = torch.from_numpy(np.asarray(target, np.int64))
        return pic, target
