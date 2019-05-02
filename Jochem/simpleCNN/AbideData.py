# A script to load images and make batch.
# Dependency: 'nibabel' to load MRI (NIFTI) images
# Reference: http://blog.naver.com/kjpark79/220783765651

import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import h5py
import glob


class AbideDataset(Dataset):
    """ root_path = Path to directionary where the data can be found
        subset = Name of the dataset we use, we don't yet have split the data so
        this comes later, so right now I put it standard to train to test.
        data_to_train = Says on which part of the summaries the CNN should train,
        Standard is the T1 input, aka the first input. Later we should make it
        possible to loop over more parts of the summaries list.
    """
    def __init__(self, root_path, subset, summary):


        self._root_path = root_path
        self._subset = subset
        self._data_to_train = summary

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
        img = torch.squeeze(img, -1)
        img = torch.unsqueeze(img, 0)
        target = torch.from_numpy(np.asarray(target, np.int64))
        return img, target

    # def get_batch(self, low_index, high_index):
    #     x_data = []
    #     y_data = []

    #     if low_index < 0 or high_index > len(self):
    #         return [], []

    #     for i in range(low_index, high_index):
    #         image, label = self[i]
    #         x_data.append(np.asarray(image, dtype='float32'))
    #         if label == 0:
    #             y_data.append([1, 0])
    #         else:
    #             y_data.append([0, 1])

    #     return np.asarray(x_data), np.asarray(y_data)

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


