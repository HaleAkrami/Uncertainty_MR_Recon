import numpy as np
import lpr.transforms as transforms
import scipy.io as sio
import subprocess
import os

def convert_to_image(data_k):
    """Convert multichannel k-space data to a single image using root-sum-of-squares
    Input Parameters:
      data_k: (#channels, #rows, #columns) Multichannel k-space data 
    Output Parameters:
      image_rsos: (#rows, #columns) Combined image
    """
    assert len(data_k.shape) == 3

    # Perform ifft
    image = transforms.ifft2(data_k)
    
    # Combine multichannel images
    image_rsos = transforms.root_sum_of_squares(image)

    return image_rsos

def normalize_image(image):
    """Normalize images
    Input Parameters:
      image: 2-D image (dim.>=2 with row and column being\
         the last two dimensions)
    Output Parameter:
      Normalized image
    """
    mu = np.mean(image, axis=(-2, -1))
    std = np.std(image, axis=(-2, -1))
    return (image - mu) / std, mu, std

def center_crop(image, target_sizes):
    """
    Crop center the of the input image
    Input Parameters:
      image:        2-D image to be cropped. (dim.>=2 with row\
                    and column being the last two dimensions)
      target_sizes: (int, int) The output shapes
    Output Parameter:
      Cropped image
    """
    assert 0 < target_sizes[0] <= image.shape[-2]
    assert 0 < target_sizes[1] <= image.shape[-1]
    r_from = (image.shape[-2] - target_sizes[0]) // 2
    c_from = (image.shape[-1] - target_sizes[1]) // 2
    r_to = r_from + target_sizes[0]
    c_to = c_from + target_sizes[1] 
    return image[..., r_from:r_to, c_from:c_to]

def crop_around_perturb(image, perturb, win_rad):
    """
    Crop the image around the perturbation
    
    Input Parameters:
      image: 2-D image to be cropped. (dim.>=2 with \
             row and column being the last two dimensions)
      perturb: a Perturbation object
      win_rad: (int, int) window radius
    Output Parameter:
      Cropped image
    """
    
    # Center of the perturbation
    r_cen, c_cen = perturb.pos_r, perturb.pos_c
    
    # Offset the center if image and perturbation have different dimensions
    r_cen -= (perturb.perturb.shape[0] - image.shape[-2]) // 2
    c_cen -= (perturb.perturb.shape[1] - image.shape[-1]) // 2
   
    # Calculate the boundaries of the cropped region
    r_from = r_cen - win_rad #np.floor(ts_r/2).astype(int)
    r_to = r_cen + win_rad #+ np.ceil(ts_r/2).astype(int)
    c_from = c_cen - win_rad #- np.floor(ts_c/2).astype(int)
    c_to = c_cen + win_rad #+ np.ceil(ts_c/2).astype(int)
    
    # Make sure the cropped region is valid
    assert r_from >= 0 and r_to < image.shape[-2]
    assert c_from >= 0 and c_to < image.shape[-1]

    return image[..., r_from:r_to, c_from:c_to]


def est_phase_sens_maps(down_k, calib):
    """
    Estimate phase and sensitivity maps
    Input Parameters:
      down_k: (#channels, #rows, #columns) k-space data
      calib:  (#channels, #rows, #columns) ACS region of `kspace`
    Output Parameter:
      est_phase: (#rows, #columns) Estimated phase
      sens_maps: (#channels, #rows, #columns) Sensitivity maps
    NOTE: this function relies on the MATLAB function `est_phase_sens_maps.m`
    """

    # Save inputs to .mat files
    sio.savemat('to_matlab.mat', dict(down_k=down_k, calib=calib))

    # Call MATLAB function
    print('Estimating phase and senstivity maps...Calling MATLAB function...')
    subprocess.run(['python', 'lpr/est_phase_sens_maps_wrapper.py'])
    
    # Load MATLAB results
    est_phase = sio.loadmat('est_phase.mat')['est_phase']
    sens_maps = sio.loadmat('sens_maps.mat')['sens_maps']

    os.remove('to_matlab.mat')
    os.remove('est_phase.mat')
    os.remove('sens_maps.mat')

    return est_phase, sens_maps


