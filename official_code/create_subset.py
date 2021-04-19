import h5py
import pathlib
from shutil import copy2


root = '/big_disk/akrami/git_repos_new/Uncertainty_MR_Recon/official_code/data/singlecoil_test/'
dst = '/big_disk/akrami/git_repos_new/Uncertainty_MR_Recon/official_code/data/singlecoil_subset_372_test/'
files = list(pathlib.Path(root).iterdir())
for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            if kspace.shape[1] == 640 and kspace.shape[2] == 372:
                copy2(fname, dst)

