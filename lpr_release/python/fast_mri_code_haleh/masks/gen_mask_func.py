from MaskFunctions import RandMaskFunc, UniformMask1D
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

acce_factor = 3
# mask_func = RandMaskFunc(acce_factor)
mask_func = UniformMask1D(acce_factor)
start = time.time()
mask = mask_func((1, 372))
mask = np.vstack([mask]*640)
np.save(f'uniform_640_372_x{acce_factor}_w16.npy', mask.astype(np.float64))
plt.imshow(mask, cmap='gray')
plt.show()
