# NAME:
#     MFT_ROTATE
#
# PURPOSE:
#     Shift and rotate an image in one step via FFTs
#
# EXPLANATION:
#     MFT_ROTATE rotates an image by ANGLE degrees about the image center,
#     magnifies it, and then shifts it by XSHIFT,YSHIFT pixels.  The image
#     must be square and should be padded.
#
# CALLING SEQUENCE:
#     result = mft_rotate( image, angle [, xshift, yshift [, mag]] )
#
# INPUT PARAMETERS:
#     IMAGE : 2-d square image array (may be complex-valued)
#     ANGLE : Angle in degrees counterclockwise to rotate image
#
# OPTIONAL INPUT PARAMETERS:
#     XSHIFT, YSHIFT : Number of pixels to shift rotated image (may be subpixel)
#     MAG: magnification (default = 1)
#
# RESULT:
#     Returns complex-valued rotated, magnified, and shifted image (complex-valued),
#
# AUTHOR:
#     John Krist, based on a method presented by Larkin et al. in Optics Communications (1997)

import numpy as np
from roman_preflight_proper import mft1

def mft_rotate( image, t_deg, xshift=0, yshift=0, mag=1 ):

    dim = image.shape[1]

    t = -t_deg / 180 * np.pi
    a = np.tan(t/2)
    b = -np.sin(t)

    x = np.tile( (np.arange(dim) - dim//2), (dim,1) )
    y = np.transpose(x)
    u = x / dim

    # shear in X
    transform_axis = 1
    gx = np.exp(-2*np.pi*1j*u*a*y) * mft1(image, 1., dim, dim, -1, transform_axis) * dim
    gx = mft1( gx, 1., dim*mag, dim, 1, transform_axis )

    # shear in Y
    transform_axis = 2
    gyx = np.exp(-2*np.pi*1j*np.transpose(u/mag)*(b*x+yshift)) * mft1(gx, 1., dim, dim, -1, transform_axis) * dim
    gyx = mft1( gyx, 1., dim*mag, dim, 1, transform_axis)

    # shear again in X
    transform_axis = 1
    g = np.exp(-2*np.pi*1j*u*(a*(y-yshift)+xshift)) * mft1(gyx, 1., dim, dim, -1, transform_axis) * dim
    g = (mag*mag) * mft1( g, 1., dim, dim, 1, transform_axis)

    return g

