# This code shows how to compute LPRs for a U-Net MR image
# reconstruction method [1].
# [1] Zbontar J, Knoll F, Sriram A, Muckley MJ, Bruno M, Defazio A,
#     Parente M, Geras KJ, Katsnelson J, Chandarana H, Zhang Z. fastMRI:
#     An open dataset and benchmarks for accelerated MRI. arXiv preprint
#     arXiv:1811.08839. 2018 Nov 21.
#
# This code implements the algorithm described in the following abstract
# and paper [2][3]:
# [2] Chan CC, Haldar J. Local perturbation responses: A tool
#     for understanding the characteristics of advanced nonlinear
#     mr reconstruction algorithms. In: Proc. Int. Soc. Magn.
#     Reson. Med.. 2020
# [3] (preprint version of the journal paper)
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
#   1. PyTorch (with CUDA enabled): This code was tested on PyTorch
#                   v 1.3.1 with CUDA v 10.1.

# *******************************************
#   Using this code in different scenarios:
# *******************************************
# 1. To use a different dataset, replace the `util.load_data` function
#    in `main()` with appropriate functions that loads the data. The
#    data loading function is expected to return k-space data with
#    a shape (#channels, #rows, #columns)
# 2. To use a different reconstruction algorithm, replace the `unet_recon`
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
import torch
import unet
import h5py

# LPR-related parameters
perturb_r_pos = 320
perturb_c_pos = 186
perturb_mag =20e-5
#perturbG=#Gussian, downsample, brightness, checkerboard, droppout,

# Other parameters
out_dir = './out_gussian/unet'
out_dir_pur = './out_gussian/pur'


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

        return x * self.mask


def load_model(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model = unet.UnetModel(1, 1, 32, 4, 0.0).to('cuda')
    model.load_state_dict(checkpoint['model'])
    return model


def unet_recon(kspace, kspace_org, R):
    """
    Run U-Net reconstruction
    """
    # Load the pre-trained model
    # model_x3.pt is the pre-trained model for R=3
    # model_x4.py is the pre-trained model for R=4
    model = load_model('unet/best_model.pt') if R == 3 else \
        load_model('unet/model_x4.pt')

    # Run reconstruction
    down_img = lpr.util.convert_to_image(kspace)
    down_img = lpr.util.center_crop(down_img, (320, 320))
    down_img, mu, std = lpr.util.normalize_image(down_img)
    down_img = torch.Tensor(down_img).cuda()
    down_img = down_img.unsqueeze(0).unsqueeze(0)
    recons = model(down_img).squeeze(0).squeeze(0).to('cpu').detach()
    recons = (recons * std) + mu

    down_img = (down_img * std) + mu
    return recons, down_img


def main():
    # Load data
    print('Loading data')
    data_k = util.load_fast_mri_data_single('../data/file1001995.h5', 17)

    # Subsample k-space data
    print('Loading sampling mask')
    sampling_func = SamplingFunction('../common/mask_x3.h5')
    down_data_k = sampling_func.subsample(data_k)

    org_img = lpr.util.convert_to_image(data_k)
    org_img = lpr.util.center_crop(org_img, (320, 320))
    # Estimate phase and sensitivity maps

    # Generate perturbation
    print('Generating perturbation')


    perturbG=1
    if perturbG==0:
        print('Generating perturbation')
        perturb = perturbations.CheckerboardPerturbation(
            perturb_mag,
            (perturb_r_pos, perturb_c_pos),
            (data_k.shape[1], data_k.shape[2]),
            [8, 8],
            [32, 32]
        )
    else:
        perturb = perturbations.GaussianPerturbation(
            perturb_mag,
            (perturb_r_pos, perturb_c_pos),
            10,
            img_size=(data_k.shape[1], data_k.shape[2])
        )
    print('Adding perturbation')
    perturbed_data_k = lpr.add_perturbation_single(
            data_k,
            perturb,
            sampling_func.subsample,
    )

    down_perturbed_data_k = sampling_func.subsample(perturbed_data_k)

    # Run reconstruction
    print('Running reconstruction')
    recon, down_img = unet_recon(
        down_data_k, data_k,
        R=3
    )
    perturb_recon, perturb_down = unet_recon(
        down_perturbed_data_k, perturbed_data_k,
        R=3
    )
    perturb_org = lpr.util.convert_to_image(perturbed_data_k)
    perturb_org = lpr.util.center_crop(perturb_org, (320, 320))

    # Save results as images
    print('Saving results')



    util.save_results_up(
        perturb_recon,
        perturb_org,
        perturb_down,org_img ,
        out_dir_pur,
        crop=False
    )


if __name__ == '__main__':
    main()
