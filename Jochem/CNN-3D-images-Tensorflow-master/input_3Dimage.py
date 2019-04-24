# A script to load images and make batch.
# Dependency: 'nibabel' to load MRI (NIFTI) images
# Reference: http://blog.naver.com/kjpark79/220783765651

import os
import numpy as np

import h5py
import glob


class AbideDataset():
    """ root_path = Path to directionary where the data can be found
        subset = Name of the dataset we use, we don't yet have split the data so
        this comes later, so right now I put it standard to train to test.
        data_to_train = Says on which part of the summaries the CNN should train,
        Standard is the T1 input, aka the first input. Later we should make it
        possible to loop over more parts of the summaries list.
    """
    def __init__(self, root_path, subset="train", data_to_train='T1'):


        self._root_path = root_path
        self._subset = subset
        self._data_to_train = data_to_train

        self._make_dataset()

    # Get's the standard data of the abide dataset.
    def _make_dataset(self):

        self._data_files = glob.glob(os.path.join(self._root_path, '*.hdf5'))
        self._data_files.sort()

        # Get the number of samples
        with h5py.File(self._data_files[0]) as hf:
            self._num_examples_per_file = len(hf['summaries'][self._data_to_train])
            self._num_examples = len(self._data_files)*self._num_examples_per_file


        # We have two classes
        self._classes = set([0,1])
        self._target_offset = min(self._classes)

        print('  Number of {} HDF5 files found: {}'.format(self._subset, len(self._data_files)))
        print('  Number of {} examples found:   {}'.format(self._subset, len(self)))
        print('  Number of {} targets found:    {}'.format(self._subset, len(self._classes)))

    def __len__(self):
        return self._num_examples

    def __getitem__(self, idx):

        container_idx = idx // self._num_examples_per_file
        example_idx = (idx-(container_idx*self._num_examples_per_file))

        # Get the asked example from the datafile from the wanted dataset
        with h5py.File(self._data_files[container_idx], 'r') as hf:
            pic = np.asarray(hf['summaries'][self._data_to_train][example_idx])
            target = [0] # hf.attrs['DX_GROUP'][example_idx]

        return pic, target

    def get_batch(self, low_index, high_index):
        batch_x = []
        batch_y = []
        if low_index < 0 or high_index > len(self):
            return [], []
        for i in range(low_index, high_index):
            x, y = self[i]
            batch_x.append(x)
            batch_y.append(y)
        return np.asarray(batch_x), np.asarray(batch_y)

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

abide = AbideDataset("/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis")
train = abide.get_batch(0, 10)
batch = train[0:5]
