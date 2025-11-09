# NAME:
#     shift_image
#
# PURPOSE:
#     Shift an image using FFTs
#
# CALLING SEQUENCE:
# INPUT PARAMETERS:
#     image : 2-d square image array (real or complex), ideally factor of 2 and padded.
#     xshift, yshift : Number of pixels to shift rotated image (may be subpixel)
#
# RESULT:
#     Returns complex-valued shifted image
#
# John Krist, JPL
# Revised Dec 2024: fixed issue with arrays being transposed because fft2, ifft2
# transforms input from C ordering to output Fortran ordering

import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from numpy.fft import fft2, ifft2
import proper

def shift_image( image, xshift_pix, yshift_pix ):

    dim = image.shape[1]

    x = -np.tile( (np.arange(dim) - dim//2) / (dim//2), (dim,1) )
    y = np.transpose(x)
    tilt = 1j * np.pi * (x*xshift_pix + y*yshift_pix)
    tilt = proper.prop_shift_center(tilt)

    new_image = proper.prop_shift_center(image)
    new_image[:,:] = fft2(new_image)
    new_image *= np.exp(tilt)
    new_image[:,:] = ifft2(new_image)
    new_image[:,:] = proper.prop_shift_center(new_image)

    return new_image

