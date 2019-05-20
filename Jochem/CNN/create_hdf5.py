import h5py
import numpy as np
import random

def write_subset_files(file, data_path, summary, test_ratio, train_val_ratio):
    train_f = h5py.File(data_path +'train.hdf5', 'w')
    val_f = h5py.File(data_path + 'validation.hdf5', 'w')
    test_f = h5py.File(data_path +'test.hdf5', 'w')

    summaries = file['summaries']
    attrs = summaries.attrs
    labels = attrs['DX_GROUP']
    dataset = summaries[summary]
    joined_set = list(zip(dataset, labels))
    random.shuffle(joined_set)

    dataset, labels = zip(*joined_set)
    test_index = round(test_ratio * len(dataset))
    train_index = test_index + round(train_val_ratio * (len(dataset) -
    test_index))

    test_data = dataset[0:test_index]
    test_labels = labels[0:test_index]

    train_data = dataset[test_index:train_index]
    train_labels = labels[test_index:train_index]

    val_data = dataset[train_index:]
    val_labels = labels[train_index:]

    train_f.create_dataset(summary, data=train_data)
    train_f.create_dataset('labels', data=train_labels)
    val_f.create_dataset(summary, data=val_data)
    val_f.create_dataset('labels', data=val_labels)
    test_f.create_dataset(summary, data=test_data)
    test_f.create_dataset('labels', data=test_labels)

    train_f.close(), val_f.close(), test_f.close()