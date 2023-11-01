import numpy as np
# from scipy.ndimage.filters import convolve
from scipy.interpolate import RegularGridInterpolator

def convert_abermap(abermap_raw,imagesize_x,imagesize_y):
    abermap_raw = abermap_raw.swapaxes(0, 2).swapaxes(1, 3).swapaxes(2, 3)
    x = np.arange(0, abermap_raw.shape[0], 1)
    y = np.arange(0, abermap_raw.shape[1], 1)
    # X, Y = np.meshgrid(x, y)
    x1 = np.arange(0, abermap_raw.shape[0], abermap_raw.shape[0]/imagesize_x)
    y1 = np.arange(0, abermap_raw.shape[1], abermap_raw.shape[1]/imagesize_y)
    X1, Y1 = np.meshgrid(x1, y1)
    abermap_buffer2 = []
    abermap_final = []
    for i in range(abermap_raw.shape[3]):
        abermap_buffer1 = []
        for j in range(abermap_raw.shape[2]):
            interp = RegularGridInterpolator((x, y), abermap_raw[:, :, j, i], bounds_error=False, fill_value=None)
            abermap_buffer1.append(interp((X1,Y1)))
        abermap_buffer2.append(np.array(abermap_buffer1).swapaxes(0, 2).swapaxes(0, 1))
    abermap_final = np.array(abermap_buffer2).swapaxes(0, 3).swapaxes(0, 1).swapaxes(1, 2)
    if abermap_final.shape[2]!=45 and abermap_final.shape[2]!=21:
        abermap_final = np.append(abermap_final,np.zeros((abermap_final.shape[0],abermap_final.shape[1],45-abermap_final.shape[2],abermap_final.shape[3])),axis=2)

    return abermap_final

# def gaussian(x, y, sigma):
#     return (1 / (2 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
#
# def generate_gaussian_kernel(size, sigma):
#     kernel = np.fromfunction(lambda x, y: gaussian(x - size//2, y - size//2, sigma), (size, size))
#     kernel /= np.sum(kernel)
#     return kernel
# def apply_gaussian_blur(image, size, sigma):
#     kernel = generate_gaussian_kernel(size, sigma)
#     blurred_image = convolve(image, kernel, mode='constant', cval=0)
#     return blurred_image
