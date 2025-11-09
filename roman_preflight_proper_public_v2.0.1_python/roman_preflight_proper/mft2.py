#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

import numpy as np

# mtf2:
#   Compute a 2D matrix fourier transform.  Based on Soummer et al. 2007.
#
# Input parameters:
#    in : 2-D wavefront to transform
#     dout: sampling in lambda/D of output (if pupil-to-focus) or input (if focus-to-pupil)
#        D: pupil size in pixels
#    nout : dimensions of output array (nout by nout)
#  direction : direction of transform (-1 or +1)
#
# Optional input parameters:
#   xoffset, yoffset : offsets in output field in cycles/D
#  Returns:
#    2-D Fourier transform of input array.
#
#  Written by Dimitri Mawet (JPL) in IDL, translated to Python by John Krist
#  Copyright 2022 California Institute of Technology

def mft2( field_in, dout, D, nout, direction, xoffset=0, yoffset=0, xc=0, yc=0 ):

    nfield_in = field_in.shape[1] 
    nfield_out = int(nout)
 
    x = (np.arange(nfield_in) - nfield_in//2 - xc) 
    y = (np.arange(nfield_in) - nfield_in//2 - yc) 

    u = (np.arange(nfield_out) - nfield_out//2 - xoffset/dout) * (dout/D)
    v = (np.arange(nfield_out) - nfield_out//2 - yoffset/dout) * (dout/D)

    xu = np.outer(x, u)
    yv = np.outer(y, v)

    expxu = dout/D * np.exp(direction * 2.0 * np.pi * 1j * xu)
    expyv = np.exp(direction * 2.0 * np.pi * 1j * yv).T

    t1 = np.dot(expyv, field_in)
    t2 = np.dot(t1, expxu)

    return t2

