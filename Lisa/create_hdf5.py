import h5py
import numpy as np

f = h5py.File('/home/lisasalomons/Desktop/fmri_summary_abideI_II.hdf5',
'r')

train_f = h5py.File('/home/lisasalomons/Desktop/afstudeerproject_KI/Lisa/train.hdf5', 'w')

val_f = h5py.File('/home/lisasalomons/Desktop/afstudeerproject_KI/Lisa/val.hdf5',
'w')

test_f = h5py.File('/home/lisasalomons/Desktop/afstudeerproject_KI/Lisa/test.hdf5',
'w')

def write_subset_files(file, summary, train_ratio, val_ratio):
    summaries = file['summaries']
    attrs = summaries.attrs
    labels = attrs['DX_GROUP']

    dataset = summaries[summary]

    train_index = round(train_ratio * len(dataset))
    val_index = train_index + round(val_ratio * len(dataset))

    train_data = dataset[0:train_index]

    # train_data = np.swapaxes(train_data, 1, 3)
    train_labels = labels[0:train_index]

    val_data = dataset[train_index:val_index]
    val_labels = labels[train_index:val_index]

    test_data = dataset[val_index:]
    test_labels = labels[val_index:]

    train_f.create_dataset(summary, data=train_data)
    train_f.create_dataset('labels', data=train_labels)
    val_f.create_dataset(summary, data=val_data)
    val_f.create_dataset('labels', data=val_labels)
    test_f.create_dataset(summary, data=test_data)
    test_f.create_dataset('labels', data=test_labels)

    train_f.close(), val_f.close(), test_f.close()

write_subset_files(f, 'T1', 0.05, 0.9)
