import numpy as np
import scipy.fftpack as ft

class Perturbation:
    def __init__(self, mag, loc, img_size):
        """Template for perturbation
        Input Parameters:
          mag:      (double) Magnitude of the perturbation
          loc:      (int, int)) Row and column position of the center of the perturbation
          img_size: (int, int) Size of the perturbation
        """
        self.mag = mag
        [self.img_size_r, self.img_size_c] = img_size
        [self.pos_r, self.pos_c] = loc

    def get_perturbation(self):
        raise NotImplementedError()

class ImpulsePerturbation(Perturbation):
    '''
    Impulse perturbation
    '''
    def __init__(self, mag, loc, img_size=(320, 320)):
        """
        Input Parameters:
          mag:      (double) Magnitude of the perturbation
          loc:      (int, int)) Row and column position of the center of the perturbation
          img_size: (int, int) Size of the perturbation
        """
        super().__init__(mag, loc, img_size)
        self.mag = mag
        self.size = img_size
        self.loc = loc
        self.perturb = self.__get_perturbation()

    def __get_perturbation(self):
        impuls = np.zeros(self.size)
        impuls[self.loc[0], self.loc[1]] = self.mag

        return impuls

class GaussianPerturbation(Perturbation):
    '''
    Gaussian perturbation
    '''

    def __init__(self, mag, loc, var, img_size=(320, 320)):
        """
        Input Parameters:
          mag:      (double) Magnitude of the perturbation
          loc:      (int, int)) Row and column position of the center of the perturbation
          var:      (float) Variance (width) of the perturbation
          img_size: (int, int) Size of the perturbation
        """
        super().__init__(mag, loc, img_size)
        self.mag = mag
        self.size = img_size
        self.var = var
        self.loc = loc
        self.perturb = self.__get_perturbation()

    def __get_perturbation(self):
        gaus = np.zeros(self.size)
        for i in range(self.size[0]):
            for j in range(self.size[1]):
                dist = np.sqrt((i-self.loc[0])**2 + (j-self.loc[1])**2)
                gaus[i, j] = np.exp(-(dist)**2/(2*self.var)) * self.mag
        return gaus

class CheckerboardPerturbation(Perturbation):
    '''
    Checkerboard perturbation
    '''
    def __init__(
        self, mag, loc, 
        img_size, square_sizes, board_sizes
        ):
        """
        Input Parameters:
          mag:      (double) Magnitude of the perturbation
          loc:      (int, int)) Row and column position of the center of the perturbation
          img_size: (int, int) Size of the perturbation
          square_sizes:  (int, int) Size of each square
          board_sizes:   (int, int) Size of the checkerboard
        """
        super().__init__(mag, loc, img_size)
        [self.s_size_y, self.s_size_x] = square_sizes
        [self.b_size_y, self.b_size_x] = board_sizes
        self.perturb = self.__get_perturbation()

    def __get_perturbation(self):
        """Return the perturbation as a numpy array"""
        if self.b_size_x % self.s_size_x < self.s_size_x/2:
            s_num_x = np.floor(self.b_size_x / self.s_size_x)
        else:
            s_num_x = np.ceil(self.b_size_x / self.s_size_x)

        if self.b_size_y % self.s_size_y < self.s_size_y/2:
            s_num_y = np.floor(self.b_size_y / self.s_size_y)
        else:
            s_num_y = np.ceil(self.b_size_y / self.s_size_y)

        actual_b_size_x = s_num_x * self.s_size_x
        actual_b_size_y = s_num_y * self.s_size_y

        white_num_x = np.ceil(s_num_x / 2)
        white_num_y = np.ceil(s_num_y / 2)

        dx = 1 / actual_b_size_x * self.s_size_x
        dy = 1 / actual_b_size_y * self.s_size_y

        half_size_x = actual_b_size_x / 2
        half_size_y = actual_b_size_y / 2
        kx = np.linspace(int(np.floor(-half_size_x)), int(np.floor(half_size_x))-1, int(actual_b_size_x)).T
        ky = np.linspace(np.floor(-half_size_y), np.floor(half_size_y)-1, int(actual_b_size_y)).T
        kx = np.reshape(kx, (kx.shape[0], 1))
        ky = np.reshape(ky, (ky.shape[0], 1))

        sing_sq = dx*dy * np.sinc(ky * dy) @ np.sinc(kx * dx).T

        perturb = np.zeros((int(actual_b_size_y), int(actual_b_size_x))).astype(np.complex128)
        for x in range(int(white_num_x)):
            for y in range(int(white_num_y)):
                sq = sing_sq * (np.exp(-1j*2*np.pi*2*y*dy*ky) @ np.exp(-1j*2*np.pi*2*x*dx*kx).T)
                if (y != white_num_y-1 or s_num_y % 2 == 0) and (x != white_num_x-1 or s_num_x % 2 == 0):
                    sq += sing_sq * (np.exp(-1j*2*np.pi*2*(y+0.5)*dy*ky) @ np.exp(-1j*2*np.pi*2*(x+0.5)*dx*kx).T)
                perturb += sq

        # shift perturbation
        shift_x = dx * (s_num_x / 2 - 0.5)
        shift_y = dy * (s_num_y / 2 - 0.5)
        if actual_b_size_x % 2 == 0:
            shift_x += 0.5 / actual_b_size_x
        if actual_b_size_y % 2 == 0:
            shift_y += 0.5 / actual_b_size_y

        perturb *= (np.exp(1j*2*np.pi*shift_y*ky) @ np.exp(1j*2*np.pi*shift_x*kx).T)

        perturb *= (self.mag * kx.shape[0] * ky.shape[0])
        perturb = ft.fftshift(ft.ifft2(ft.ifftshift(perturb)))

        perturb_whole = np.zeros((self.img_size_r, self.img_size_c)).astype(np.complex128)
        wx = np.round(actual_b_size_x) / 2 + 1e-5
        wy = np.round(actual_b_size_y) / 2 + 1e-5
        r_start = self.pos_r-int(np.floor(wy))
        r_end = r_start + perturb.shape[0]
        c_start = self.pos_c-int(np.floor(wx))
        c_end = c_start + perturb.shape[1]
        perturb_whole[r_start:r_end, c_start:c_end] = np.abs(perturb)

        return perturb_whole
