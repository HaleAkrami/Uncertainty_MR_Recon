import h5py
import numpy as np
import sys

np_file_name = sys.argv[1]
h5_file_name = sys.argv[2]

mask = np.load(np_file_name)
print(mask.shape, mask.dtype)
f = h5py.File(h5_file_name, 'w')
f.create_dataset('mask', data=mask)
f.close()
