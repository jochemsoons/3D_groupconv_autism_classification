import h5py

f = h5py.File("fmri_summary_abideI_II.hdf5", 'r')
for key in f['summaries']:
    print(key)

print(list(f.attr.keys()))
