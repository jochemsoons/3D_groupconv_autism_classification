import h5py
import numpy as np

f = h5py.File("/home/lisasalomons/Desktop/fmri_summary_abideI_II.hdf5", 'r')
f1 = f['summaries']
labels = f1.attrs['DX_GROUP']
for group, i in f1:
    
