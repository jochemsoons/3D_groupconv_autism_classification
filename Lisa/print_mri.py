import numpy as np
import h5py
import matplotlib.pyplot as plt

f = h5py.File('/home/lisasalomons/Desktop/fmri_summary_abideI_II.hdf5',
'r')

def print_data(name, file, sn):
    summaries = file['summaries']
    print("possible keys are: {}".format(summaries.keys()))
    data = summaries[name][sn]
    data = np.rot90(data.squeeze(), 1)
    print(data.shape)
    fig, ax = plt.subplots(1, 9, figsize=[18,3])
    attrs = summaries.attrs
    labels = attrs['DX_GROUP']
    label = labels[sn]

    n = 0
    slice = 0
    for _ in range(9):
        ax[n].imshow(data[:,:, slice], 'gray')
        ax[n].set_xticks([])
        ax[n].set_yticks([])
        ax[n].set_title('label #{}'.format(label))
        n +=1
        slice += 5
    fig.subplots_adjust(wspace=0, hspace=0)
    plt.show()

print_data('reho', f, 2000)
