import numpy as np
from lpr.perturbations import Perturbation
import lpr.transforms as transforms
import numpy.fft as ft

def add_perturbation( 
        down_data_k, 
        perturb_obj, 
        sampling_func, 
        est_phase, 
        sens_maps
    ):
    """Code for loading FastMRI dataset.
    Input Parameters:
      down_data_k:  (#rows x #columns x #channels) Undersampled k-space data
      perturb:      (#rows x #columns) Perturbation
      mask:         (#rows x #columns) Sampling mask
      est_phase:    (#rows x #columns) Estimated phase
      sens_maps:    (#rows x #columns x #channels) Sensitiviy maps
    Output Parameter:
      down_perturbed_data_k: (#rows x #columns x #channels) Perturbed k-space data
    """

    # Get perturbation
    perturb = perturb_obj.perturb.astype(np.complex128)

    # Check if the perturbation has the same size as the data
    #assert(down_data_k.shape[1] == perturb.shape[0]\
       # and down_data_k.shape[2] == perturb.shape[1])
      
    
    # Add phase to perturbation
    phase = np.exp(1j * est_phase)
    perturb *= phase

    # Multiply perturbation with sensitivity maps
    multi_perturb = perturb * sens_maps
    multi_perturb_k = transforms.fft2(multi_perturb)

    # Downsample perturbation
    down_perturb_k = sampling_func(multi_perturb_k)

    # Add perturbation
    down_perturbed_data_k = down_data_k + down_perturb_k

    return down_perturbed_data_k


def add_perturbation_single(
        down_data_k,
        perturb_obj,
        sampling_func
):
    """Code for loading FastMRI dataset.
    Input Parameters:
      down_data_k:  (#rows x #columns x #channels) Undersampled k-space data
      perturb:      (#rows x #columns) Perturbation
      mask:         (#rows x #columns) Sampling mask

    Output Parameter:
      down_perturbed_data_k: (#rows x #columns x #channels) Perturbed k-space data
    """

    # Get perturbation
    perturb = perturb_obj.perturb.astype(np.complex128)

    # Check if the perturbation has the same size as the data
    # assert(down_data_k.shape[1] == perturb.shape[0]\
    # and down_data_k.shape[2] == perturb.shape[1])

    acs_width = 16
    acs = np.zeros_like(down_data_k)
    acs[:, 310:310+acs_width, 177:177+acs_width] = down_data_k[:, 310:310+acs_width, 177:177+acs_width]
    # [est_phase, sens_maps] = lpr.util.est_phase_sens_maps(data_k, calib)
    temp = (0.54 + 0.46 * np.cos(np.linspace(-down_data_k.shape[1] / 2, down_data_k.shape[1] / 2 - 1, down_data_k.shape[1]) * np.pi / (acs_width / 2)))
    temp = np.expand_dims(temp , 0)
    temp = np.expand_dims(temp, 2)
    acs=acs*temp
    phase_est = np.exp(1j* np.angle(ft.ifftshift(transforms.ifft2(ft.ifftshift(acs)))))

    perturb=np.expand_dims(perturb, 0)
    # Multiply perturbation with sensitivity maps
    #perturb *= phase_est
    multi_perturb_k = transforms.fft2(perturb)

    # Downsample perturbation
    #multi_perturb_k = sampling_func(multi_perturb_k)

    # Add perturbation
    down_perturbed_data_k = down_data_k + multi_perturb_k

    return down_perturbed_data_k


def calculate_lpr(recon, perturb_recon, perturb):
    """Calculate the LPRs
    Input Parameters:
      recon:          (#rows x #columns) normal reconstruction
      puerturb_recon: (#rows x #columns) perturbed reconstruction
      perturb:        (Perturbation) perturbation object
    Output Parameter:
      lpr:            (#rows x #columns) LPR
    """
    lpr = (perturb_recon - recon) / perturb.mag
    return lpr