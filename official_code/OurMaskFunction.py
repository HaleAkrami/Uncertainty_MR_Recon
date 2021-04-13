import numpy as np
import scipy.fftpack as fftpack
import torch

# Use double tensor
# torch.set_default_dtype(torch.float64)

class OurMaskFuncFixed:
    def __init__(self, acce, shape):
        self.acce = acce
        self.shape = shape
        self.mask_2d = self.__load_2d_mask(self.shape)

    def __load_2d_mask(self, shape):
        mask = np.load('uniform_{}_{}_x{}_w16.npy'.format(shape[0], shape[1], self.acce))
        return np.stack((mask, mask), axis=2)

    def __call__(self):
        return torch.from_numpy(self.mask_2d)

class OurMaskFunc:
    def __init__(self, acce, k_mask_init=None):
        self.acce = acce
        self.k_mask_init = k_mask_init


    def __call__(self, shape):
        N1, N2 = shape
        M = N1*N2/self.acce
        if self.k_mask_init is None:
            k_mask_c = np.zeros(shape)
        else:
            k_mask_c = self.k_mask_init

        k_mask_c = fftpack.ifftshift(k_mask_c)
        k_mask = np.zeros(shape)
        x, y = np.meshgrid(np.arange(-N2//2,N2//2), np.arange(-N1//2, N1//2))

        for radius in np.linspace(N1*N2//M, 0, N1*N2//M*10+1):
            if len(np.nonzero(k_mask + k_mask_c)[0]) < M:
                rad = np.sqrt(x**2 + y**2) <= radius
                frad = fftpack.fft2(fftpack.ifftshift(rad))
                out = np.real(fftpack.ifft2(fftpack.fft2(k_mask) * frad)) > 0.5
                while len(np.nonzero(k_mask + k_mask_c)[0]) < M and np.sum(out) < out.size:
                    nz = np.argwhere(out == 0)
                    p = (np.floor(np.random.rand(1)*nz.shape[0])).astype(np.int32)
                    # pc, pr = p // nz.shape[0], p//nz.shape[0]
                    k_mask[nz[p, 0], nz[p, 1]] = 1
                    # print(nz[pr, pc])
                    out = np.real(fftpack.ifft2(fftpack.fft2(k_mask) * frad)) > 0.5
        k_mask = fftpack.fftshift(k_mask + k_mask_c)
        k_mask = k_mask > 0

        # center band
        band_width = N2//20
        k_mask[N1//2-band_width:N1//2+band_width,N2//2-band_width:N2//2+band_width] = 1
        return k_mask
