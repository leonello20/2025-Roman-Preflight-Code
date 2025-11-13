import numpy as np

from scipy import __version__ as ver 
if ver >= '1.8':
    from scipy.ndimage import map_coordinates
else:
    from scipy.ndimage.interpolation import map_coordinates

def shiftrot( image, xshift_pix, yshift_pix, rotation_deg, xmag=1, ymag=1 ):

    n = image.shape[0]

    xnew = np.arange(n, dtype=np.float64) - np.floor(n//2) - xshift_pix
    xnew = np.tile(xnew, (n,1))
    ynew = np.arange(n, dtype=np.float64) - np.floor(n//2) - yshift_pix
    ynew = np.tile(ynew, (n,1)).T

    t = -rotation_deg / 180 * np.pi
    x = (xnew * np.cos(t) - ynew * np.sin(t)) / xmag + np.floor(n//2)
    y = (xnew * np.sin(t) + ynew * np.cos(t)) / ymag + np.floor(n//2)

    new_image = map_coordinates( image.T, [x,y], order=3, mode="constant", cval=0.0 )

    return new_image

