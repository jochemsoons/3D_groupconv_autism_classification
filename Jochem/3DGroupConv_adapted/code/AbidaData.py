import numpy as np
np.set_printoptions(threshold=np.inf)
import random
import h5py
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import normalize

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

def scale(X, x_min, x_max):
    x = np.array(X)
    # nom = X-X.min()*(x_max-x_min)
    # denom = X.max() - X.min()
    # denom = denom + (denom is 0)
    xmin, xmax = x.max(), x.min()
    x = (x - xmin)/(xmax - xmin)
    return x
    # return x_min + nom/denom

def create_input_fn(file, summary, batch_size, num_epochs, sample_ratio, shuffle=True):
    feature = file[summary]
    labels = file['labels']
    ratio_split = int(sample_ratio*len(feature))
    feature = feature[:ratio_split]
    labels = labels[:ratio_split]
    # feature = np.array(feature)
    # feature = stats.zscore(feature, axis=None)
    # feature = scale(feature, -1, 1)
    print("mean", feature.mean())
    print("max", feature.max())
    print("min", feature.min())
    feature = feature - feature.mean()
    feature = feature / feature.max()
    print("mean", feature.mean())
    print("max", feature.max())
    print("min", feature.min())

    print("Percentage of label 1: {:.4f} (baseline of random classification)".format((np.sum(labels) / len(labels))))
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.asarray(feature)},
        y={'y': np.asarray(labels)},
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        queue_capacity=5000,
        num_threads=1)
    return input_fn, len(labels)

def create_input_fn_test_mode(train_f, val_f, summary, batch_size, num_epochs, sample_ratio, shuffle=True):
    feature_t = train_f[summary]
    feature_v = val_f[summary]
    feature = np.concatenate((feature_t, feature_v))

    labels_t = train_f['labels']
    labels_v = val_f['labels']
    labels = np.concatenate((labels_t, labels_v))

    ratio_split = int(sample_ratio*len(feature))
    feature = feature[:ratio_split]
    labels = labels[:ratio_split]

    print("Percentage of label 1: {:.4f} (baseline of random classification)".format((np.sum(labels) / len(labels))))
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': np.asarray(feature)},
        y={'y': np.asarray(labels)},
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=shuffle,
        queue_capacity=5000,
        num_threads=1)
    return input_fn, len(labels)

def explore_data(file):
    print("Printing description of data contents...")
    summaries = file['summaries']
    print("Data consists of {} summaries:\n {}".format(len(summaries), list(summaries)))
    attrs = summaries.attrs
    labels = attrs['DX_GROUP']
    print("Data consists of {} samples".format(len(labels)))
    patients = 0
    controls = 0
    for label in labels:
        if label == 1:
            patients += 1
        else:
            controls += 1
    print("ASD patients: {} \nControl group: {}".format(patients, controls))
