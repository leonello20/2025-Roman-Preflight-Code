# NAME:
#     fft_rotate
#
# PURPOSE:
#     Shift and rotate an image in one step via FFTs
#
# EXPLANATION:
#     fft_rotate first rotates an image by ANGLE degrees about the image center
#     and then shifts it by xshift,yshift pixels.  The image must be square and
#     should be a factor of two in dimension, since FFTs are used. The image
#     should be padded as well.
#
# CALLING SEQUENCE:
#     result = fft_rotate( image, angle [, xshift, yshift] )
#
# INPUT PARAMETERS:
#     image : 2-d square image array (may be complex-valued)
#     angle : Angle in degrees counterclockwise to rotate image
#
# OPTIONAL INPUT PARAMETERS:
#     xshift, yshift : Number of pixels to shift rotated image (may be subpixel)
#
# RESULT:
#     Returns complex-valued rotated and shifted image
#
# AUTHOR:
#     John Krist, based on a method presented by Larkin et al. in Optics Communications (1997)

import os
os.environ['MKL_NUM_THREADS'] = '1'
import numpy as np
from numpy.fft import fft, ifft

def fft_rotate( image, t_deg, xshift=0, yshift=0 ):

    dim = image.shape[1]

    t = -t_deg / 180 * np.pi
    a = np.tan(t/2)
    b = -np.sin(t)

    x = np.tile( (np.arange(dim) - dim//2), (dim,1) )
    y = np.transpose(x)
    u = np.roll( x / dim, -dim//2, axis=1 )

    gx =  ifft(np.exp(-2*np.pi*1j*u*a*y) * fft(image,axis=1), axis=1)
    gyx = ifft(np.exp(-2*np.pi*1j*np.transpose(u)*(b*x+yshift)) * fft(gx, axis=0), axis=0)
    g =   ifft(np.exp(-2*np.pi*1j*u*(a*(y-yshift)+xshift)) * fft(gyx, axis=1), axis=1)

    return g

