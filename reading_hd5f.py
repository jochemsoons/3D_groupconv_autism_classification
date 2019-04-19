import h5py

f = h5py.File("fmri_summary_abideI_II.hdf5", 'r')
for key in f['summaries']:
    print(key)

print(list(f.attrs.keys()))
with f:
    num_examples = len(f['summaries']['T1'])


    classes = set([0,1])

    target_offset = min(classes)

    print('  Number of examples found:   {}'.format(num_examples))
    print('  Number of targets found:    {}'.format(len(classes)))
