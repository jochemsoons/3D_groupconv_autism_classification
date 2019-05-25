import numpy as np
import random
# import os
import h5py
import tensorflow as tf
# import glob

# from torch.utils.data import Dataset, DataLoader

# class AbideDataset(Dataset):
#     def __init__(self, root_path, subset, summary):


#         self._root_path = root_path
#         self._subset = subset
#         self._data_to_train = summary

#         self._make_dataset()

#     # Get's the standard data of the abide dataset.
#     def _make_dataset(self):

#         self._data_files = glob.glob(os.path.join(self._root_path, '{}.hdf5'.format(self._subset)))

#         # Get the number of samples
#         with h5py.File(self._data_files[0]) as hf:
#             self._num_examples_per_file = len(hf[self._data_to_train])
#             self._num_examples = len(self._data_files)*self._num_examples_per_file

#         # We have two classe
#         self._classes = set([0,1])
#         print('Number of {} HDF5 files found: {}'.format(self._subset, len(self._data_files)))
#         print('Number of {} examples found:   {}'.format(self._subset, len(self)))
#         print('Number of {} targets found:    {}\n'.format(self._subset, len(self._classes)))

#     def __len__(self):
#         return self._num_examples

#     def __getitem__(self, idx):

#         with h5py.File(self._data_files[0], 'r') as hf:
#             img = hf[self._data_to_train][idx]
#             target = hf['labels'][idx]

#         img = torch.from_numpy(img)
#         img = torch.squeeze(img, -1)
#         img = torch.unsqueeze(img, 0)
#         target = torch.from_numpy(np.asarray(target, np.int64))
#         return img, target

#     @property
#     def classes(self):
#         return self._classes

#     @property
#     def subset(self):
#         return self._subset

#     @property
#     def num_classes(self):
#         return len(self._classes)

def write_subset_files(file, data_path, test_ratio, train_val_ratio):
    train_f = h5py.File(data_path +'train.hdf5', 'w')
    val_f = h5py.File(data_path + 'validation.hdf5', 'w')
    test_f = h5py.File(data_path +'test.hdf5', 'w')

    summaries = file['summaries']
    attrs = summaries.attrs
    labels = np.array(attrs['DX_GROUP'])

    length = len(labels)
    p = np.random.permutation(length)
    labels = labels[p]

    test_index = round(test_ratio * length)
    train_index = test_index + round(train_val_ratio * (length -test_index))

    test_labels = labels[0:test_index]
    train_labels = labels[test_index:train_index]
    val_labels = labels[train_index:]
    train_f.create_dataset('labels', data=train_labels)
    val_f.create_dataset('labels', data=val_labels)
    test_f.create_dataset('labels', data=test_labels)

    for summary in summaries:
        dataset = np.array(summaries[summary])
        dataset = dataset[p]

        test_data = dataset[0:test_index]
        train_data = dataset[test_index:train_index]
        val_data = dataset[train_index:]

        train_f.create_dataset(summary, data=train_data)
        val_f.create_dataset(summary, data=val_data)
        test_f.create_dataset(summary, data=test_data)

    train_f.close(), val_f.close(), test_f.close()

def create_input_fn(file, summary, batch_size, num_epochs, shuffle=True):
    feature = file[summary]
    labels = file['labels']

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.asarray(feature)},
        y={'y': np.asarray(labels)},
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        queue_capacity=5000,
        num_threads=1)
    return input_fn