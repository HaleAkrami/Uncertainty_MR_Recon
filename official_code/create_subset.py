import h5py
import pathlib
from shutil import copy2


root = '/big_disk/akrami/git_repos_new/Uncertainty_MR_Recon/official_code/data/multicoil_test_v2/'
dst = '/big_disk/akrami/git_repos_new/Uncertainty_MR_Recon/official_code/data/multicoil_subset_372_test/'
files = list(pathlib.Path(root).iterdir())
for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            if kspace.shape[2] == 640 and kspace.shape[3] == 372:
                copy2(fname, dst)

