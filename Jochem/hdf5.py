import h5py

f = h5py.File('/home/jochemsoons/Documents/BG_jaar_3/Bsc_Thesis/fmri_summary_abideI_II.hdf5',
 'r')


print(f.keys())
summaries = f['summaries']
print(summaries.keys())
# T1 = summaries['T1']
for summary in summaries.keys():
    # print(summary)
    print(summaries[summary].shape)