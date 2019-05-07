import h5py
import numpy as np
import random


# f = h5py.File('/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/fmri_summary_abideI_II.hdf5',
# 'r')

# train_f = h5py.File('/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/train.hdf5', 'w')

# val_f = h5py.File('/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/validation.hdf5',
# 'w')

# test_f = h5py.File('/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/test.hdf5',
# 'w')

def write_subset_files(file, summary, test_ratio, train_val_ratio):
    summaries = file['summaries']
    attrs = summaries.attrs
    labels = attrs['DX_GROUP']
    dataset = summaries[summary]
    # dataset = dataset[:,20:23:,:,]
    # print(dataset.shape)
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