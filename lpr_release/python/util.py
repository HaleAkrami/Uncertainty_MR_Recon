import h5py
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import lpr
import pathlib

def load_fast_mri_data(vol_path, slice_n):
    """The code for loading FastMRI dataset.
    Input Parameters:
        vol_path:  (string) Path to the volume
        slice_n:   (int) Slice number 
    
    Output Parameter:
        data_k: (#channels x #rows x #columns) k-space data
    """
    with h5py.File(vol_path, 'r') as data:
        # load k-space data
        k_space = np.array(data['kspace'])
        #k_space = np.expand_dims(k_space, 1) for single coil

    return np.flip(k_space[slice_n, :, :, :], 1)


def load_fast_mri_data_single(vol_path, slice_n):
    """The code for loading FastMRI dataset.
    Input Parameters:
        vol_path:  (string) Path to the volume
        slice_n:   (int) Slice number

    Output Parameter:
        data_k: (#channels x #rows x #columns) k-space data
    """
    with h5py.File(vol_path, 'r') as data:
        # load k-space data
        k_space = np.array(data['kspace'])
        k_space = np.expand_dims(k_space, 1)

    return np.flip(k_space[slice_n, :, :, :], 1)

class load_test_mri_data:


    def __init__(self,root,challenge):

        if challenge not in ('singlecoil', 'multicoil'):
                raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        
        self.examples = []
        files = list(pathlib.Path(root).iterdir())
        
        for fname in sorted(files):
            kspace = h5py.File(fname, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(fname, slice) for slice in range(num_slices)]

    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        fname, slice = self.examples[i]
        with h5py.File(fname, 'r') as data:
            k_space = np.array(data['kspace'])
            #k_space = np.expand_dims(k_space, 1) #only for single coil
            return np.flip(k_space[slice, :, :, :], 1)

def write_fig_to_file(
        img, img_path, cmap='gray',
        vmin=None, vmax=None
    ):
    """Utility function for writing images to png files
    Input Parameters:
        img:      (#rows, #columns) image to be saved
        img_path: (string) path to image file
        cmap:     (string) colormap in which the image will be shown
        vmin:     (float) min. value of the image to display
        vmax:     (float) max. value of the image to display
    """
    plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.savefig(
        img_path, bbox_inches='tight',
        pad_inches=0, dpi=300
    )

def save_results(
        recon, perturb, 
        resp, out_dir, crop=False
    ):
    """Utility function for saving reconstructions and LPRs as images
    Input Parameters:
        recon:   (#rows, #columns) reconstructed image
        perturb: (Perturbation) perturbation object
        resp:    (#rows, #columns) LPR
        out_dir: (string) folder to which images will be saved
        crop:    (bool) whether to apply cropping
    """

    perturb_img = np.abs(perturb.perturb)
    if crop:
        crop_func = lambda img, size: \
            lpr.util.crop_around_perturb(
                img, perturb, size
            )
        recon = crop_func(recon, 160)
        resp = crop_func(resp, 160)
        
    # save normal reconstruction
    write_fig_to_file(
        recon, 
        join(out_dir, 'recon.png')
    )

    # save LPR
    write_fig_to_file(
        # lpr.util.add_color(resp),
        resp,
        join(out_dir, 'lpr.png')
    )

def save_results_up(
        recon, org_img,
        down_img, org_img_v1,
        out_dir, crop=False
    ):
    """Utility function for saving reconstructions and LPRs as images
    Input Parameters:
        recon:   (#rows, #columns) reconstructed image
        perturb: (Perturbation) perturbation object
        resp:    (#rows, #columns) LPR
        out_dir: (string) folder to which images will be saved
        crop:    (bool) whether to apply cropping
    """

 
        
    # save normal reconstruction
    write_fig_to_file(
        recon, 
        join(out_dir, 'recon.png')
    )

    # save LPR
    write_fig_to_file(
        org_img, 
        join(out_dir, ' org_img.png')
    )

    write_fig_to_file(
        down_img.squeeze(0).squeeze(0).cpu().numpy(), 
        join(out_dir, ' down_img.png')
    )

    write_fig_to_file(
        np.abs(org_img-recon.cpu().numpy()), 
        join(out_dir, ' error.png')
    )

    write_fig_to_file(
        np.abs(org_img - org_img_v1),
        join(out_dir, ' error_org.png')
    )

def save_results_uncer(
        recon, var,org_img,
        down_img,
        out_dir, crop=False
    ):
    """Utility function for saving reconstructions and LPRs as images
    Input Parameters:
        recon:   (#rows, #columns) reconstructed image
        perturb: (Perturbation) perturbation object
        resp:    (#rows, #columns) LPR
        out_dir: (string) folder to which images will be saved
        crop:    (bool) whether to apply cropping
    """

 
        
    # save normal reconstruction
    write_fig_to_file(
        recon, 
        join(out_dir, 'recon.png')
    )

    # save LPR
    write_fig_to_file(
        org_img, 
        join(out_dir, ' org_img.png')
    )

    write_fig_to_file(
        down_img.squeeze(0).squeeze(0).cpu().numpy(), 
        join(out_dir, ' down_img.png')
    )

    write_fig_to_file(
        np.abs(org_img-recon.cpu().numpy()), 
        join(out_dir, ' error.png')
    )

    write_fig_to_file(
        var,
        join(out_dir, ' var.png')
    )