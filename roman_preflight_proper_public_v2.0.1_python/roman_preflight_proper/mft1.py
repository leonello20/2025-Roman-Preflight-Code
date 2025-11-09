# mtf1:
#   Compute a 1-D matrix fourier transform for a.  Based on Soummer et al. 2007.
#
# Input parameters:
#    in : 1-D or 2-D wavefront to transform. If 2-D, then the transform is computed
#          separately for each row, returning a 2-D array
#     dout: sampling in lambda/D of output (if pupil-to-focus) or input (if focus-to-pupil)  
#        D: pupil size in pixels
#   nout: dimensions of output array. If input is 2-D and DIM=1 or is not specified, the 
#         output is "nout" columns by the number of input rows; if DIM=2, then the output
#         is the number of input columns by "nout" rows.
#  direction : direction of transform (-1 or +1) 
#
# Optional input parameters:
#  dim: If input is 2-D, specifies if transforms are done along rows (DIM=1) or
#       columns (DIM=2). The default is DIM=1.
#
#  Returns:
#    1-D Fourier transform of input array (or 1-D transforms of 2-D input array).
#
#  Written by John Krist, 10 June 2022
#  Based on 2-D MFT code by Dimitri Mawet

import numpy as np

def mft1( field_in, dout, D, nout, direction, dim=1 ):

    nfield_out = int(nout)

    if ( dim == 1 ):
        # compute MFT along rows
        if field_in.ndim == 2:
            nfield_in = field_in.shape[1] 
        else:
            nfield_in = field_in.shape[0] 
        x = np.arange(nfield_in) - nfield_in//2
        u = (np.arange(nfield_out) - nfield_out//2) * (dout/D)
        u = u * (direction * 2.0 * np.pi * 1j) 
        expxu = dout/D * np.exp(np.outer(x, u))
        return np.dot(field_in, expxu)
    else:
        # compute MFT along columns
        nfield_in = field_in.shape[0]
        y = np.arange(nfield_in) - nfield_in//2
        v = (np.arange(nfield_out) - nfield_out//2) * (dout/D)
        yv = np.outer(y, v)
        expyv = dout/D * np.exp(direction * 2.0 * np.pi * 1j * yv).T
        return np.dot(expyv, field_in)

