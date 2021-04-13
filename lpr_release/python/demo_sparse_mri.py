# This code shows how to compute LPRs for the Sparse-MRI MR image 
# reconstruction method [1].
# [1] Lustig M, Donoho D, Pauly JM. Sparse mri: The application of
#     compressed sensing for rapid mr imaging. Magn Reson Med 2007;
#     58:1182–1195.
# 
# This code implements the algorithm described in the following 
# abstract and paper [2][3]:
# [2] Chan CC, Haldar J. Local perturbation responses: A tool 
#     for understanding the characteristics of advanced nonlinear 
#     mr reconstruction algorithms. In: Proc. Int. Soc. Magn.
#     Reson. Med.. 2020.
# [3] (preprint version of the journal paper).
# 
# ***************
#   Parameters:
# ***************
#   perturb_r_pos: (int) The y coordinate of the perturbation to
#                  added.
#   perturb_c_pos: (int) The x coordinate of the perturbation to
#                  added.
#   perturb_mag:   (double) The magnitude of the perturbation
#   out_dir:       (string) The directory to which the results 
#                  will be saved.
# 
# *************************************
#   Additional Software Dependencies:
# *************************************
# In addition to the software listed in the README file, this code
# also requires the following Python package to run:
# 1. bart: The `bart` package can be downloaded from 
#          https://mrirecon.github.io/bart/. 
# 
# *******************************************
#   Using this code in different scenarios:
# *******************************************
# 1. To use a different dataset, replace the `util.load_data` function
#    in `main()` with appropriate functions that loads the data. The 
#    data loading function is expected to return k-space data with 
#    a shape (#channels, #rows, #columns)
# 2. To use a different reconstruction algorithm, replace the `cs_total_variation`
#    function in `main()` with appropriate functions that do the 
#    reconstruction. The reconstruction function should take the 
#    down-sampled k-space data as input and returns the reconstructed 
#    k-space.
# 3. To change the perturbation, replace the `perturbations.CheckerboardPerturbation`
#    function in `main()` with appropriate functions that generate the 
#    perturbation. The code for generating the impulse, Gaussian, and 
#    checkerboard perturbations can be found in `perturbations.py` under
#    the `lpr` folder.
# 4. To change the sampling scheme, replace the `SamplingFunction` with 
#     appropriate functions that perform the sampling process. In 
#    this example code, we use the Cartesian sampling with a pre-defined
#    mask. But more complicated sampling schemes can be used by modifying
#    the `subsample` method of the `SamplingFunction` class.
# 5. In applications that require a different way of perturbing data,
#    please replace the `lpr.add_perturbation` function in `main()` with 
#    appropriate functions that perturbs the data. The perturbation function
#    should return the perturbed k-space data.
# 
# This software is available from ...
# 
# As described on that page, use of this software (or its derivatives) in
# your own work requires that you at least cite [1] and [2].
# 
# V1.0 Chin-Cheng Chan and Justin P. Haldar 10/12/2020
# 
# This software is Copyright ©2020 The University of Southern California.
# All Rights Reserved. See the accompanying license.txt for additional
# license information.

import numpy as np
import util
import lpr
import lpr.perturbations as perturbations 
import bart
import h5py

# LPR-related parameters
perturb_r_pos = 320
perturb_c_pos = 186
perturb_mag = 5e-5

# Reconstruction-related parameters
num_iters = 300
reg_wt = 5e-5

# Other parameters
out_dir = './out/sparse_mri'

class SamplingFunction:
    """
    Sampling function
    """
    def __init__(self, mask_path):
        """
        Input Parameter:
          mask_path: (string) path to the sampling mask
        """
        with h5py.File(mask_path, 'r') as ma:
            self.mask = np.array(ma['mask'])

    def subsample(self, x):
        """ 
        Parameter:
        x: data to be sub-sampled (dim.>=2 with row  
           and column being the last two dimensions)
        """
        assert x.shape[-2] == self.mask.shape[-2] 
        assert x.shape[-1] == self.mask.shape[-1]

        return x*self.mask

def cs_total_variation(kspace):
    """
    Run Total Variation Minimization based
    reconstruction algorithm using the BART toolkit.
    """
    kspace = np.expand_dims(
        np.transpose(kspace, (1, 2, 0)), 0
    )

    sens_maps = bart.bart(1, f'ecalib -d0 -m1', kspace)

    kspace = kspace.astype(np.complex64)
    sens_maps = sens_maps.astype(np.complex64)

    pred = bart.bart(
        1, f'pics -d0 -S -R T:7:0:{reg_wt} -i {num_iters}', kspace, sens_maps
    )
    pred = np.abs(pred[0])
    return pred

def main():
    # Load data
    print('Loading data')
    data_k = util.load_fast_mri_data('../data/file1000308.h5', 17)

    # Subsample k-space data
    print('Loading sampling mask')
    sampling_func = SamplingFunction('../common/mask_x3.h5')
    down_data_k = sampling_func.subsample(data_k)

    # Estimate phase and sensitivity maps
    calib = down_data_k[:, 310:329, 177:195]
    [est_phase, sens_maps] = lpr.util.est_phase_sens_maps(down_data_k, calib)

    # Generate perturbation
    print('Generating perturbation')
    perturb = perturbations.CheckerboardPerturbation(
        perturb_mag,
        (perturb_r_pos, perturb_c_pos),
        (data_k.shape[1], data_k.shape[2]),
	    [20, 20],
	    [320, 320]
    )

    # Add perturbation
    print('Adding perturbation')
    down_perturbed_data_k = lpr.add_perturbation(
        down_data_k,
        perturb,
        sampling_func.subsample,
        est_phase,
        sens_maps
    )
    
    # Run reconstruction
    print('Running reconstruction')
    recon = cs_total_variation(
        down_data_k
    )
    perturb_recon = cs_total_variation(
        down_perturbed_data_k
    )
    
    # Calculate LPR
    print('Calculating LPR')
    resp = lpr.calculate_lpr(
        recon, perturb_recon, perturb
    )

    # Save results as images
    print('Saving results')
    util.save_results(
        recon,
        perturb,
        resp,
        out_dir,
        crop = True
    )


if __name__ == '__main__':
    main()
