import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)

from astropy.io import fits
from CAPyBARA.rst_design import _make_rst_aperture, _make_lyot_mask
from CAPyBARA.rst_functions import *
import scipy.fftpack
import os
import sys

from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import scipy.ndimage as ndimage

def get_field_object (arr, grid, is_cube=False):
    field_obj = Field(arr, grid)

    if is_cube is True: 
        field_obj = Field(arr[i], grid)

    return field_obj 

def plt_field (field, title, cmap=None, is_log=False, is_focal_plane=False): 
    if cmap is None: 
        cmap = 'gray'
    plt.title(title)

    if is_log is False:
        imshow_field(field, cmap=cmap)
    else:
        imshow_field(np.log10(field), cmap=cmap)

    if is_log and is_focal_plane is True:
        imshow_field(np.log10(field), cmap=cmap, vmin=-10, vmax=-5)
    
    plt.colorbar()
    # plt.xlabel('Pixel')
    # plt.ylabel('y/D [m]')
    plt.show()

def plt_focal_image (CAPyBARA, field, title, cmap=None):
    plt.title(title)
    imshow_field(np.log10(field/CAPyBARA.ref_img.max()), cmap=cmap,  vmin=-10, vmax=-5)
    plt.colorbar()
    plt.show()

# perturbation_evolution
def plt_field_cube (CAPyBARA, title, field_cube, radius_ppl_px=336, mp4=False):

    """Make the perturbation evolve in a .mp4 file and datacube
    TODO - organise this such that the mp4 and field cubes work together

    Parameters
    ------------
    title = title of the file, formalt ' .mp4'
    perturbations = array of fields
    radius_ppl_px = integer, radius of the pupil in pixels, 336 by default

    Return
    ----------
    turbulence_sequence = .mp4 file, evolution of the perturbation
    perturbation_datacube = datacube, perturbation
    """
    sequence = FFMpegWriter(title, framerate=5)
    # turbulence_sequence = FFMpegWriter(title, framerate=5)
    datacube = np.zeros((len(field_cube), radius_ppl_px*2, radius_ppl_px*2))
    # perturbation_datacube = np.zeros((len(perturbations), radius_ppl_px*2, radius_ppl_px*2))

    for i in range(len(field_cube)):
        perturb = Field(field_cube[i], CAPyBARA.pupil_grid)
        plt.clf()
        imshow_field(perturb, vmax = 0.2, vmin = -0.2) # vmax=0.001, vmin=-0.001)
        plt.colorbar()
        turbulence_sequence.add_frame()
        datacube[i,:,:] = perturb.shaped
    plt.close()
    sequence.close()

    if mp4 is True:
        return (turbulence_sequence, perturbation_datacube)
    else:
        return perturbation_datacube

def plt_std_vs_iteration (array, is_rad=False):
    std_cube = np.zeros(len(array))
    if is_rad is True: 
        factor = (2.* np.pi) 
    else: 
        factor = 1. 

    for i in range (len(array)):
        std_cube[i] = np.std(array[i]) / factor

    plt.plot(range(len(std_cube)), std_cube, label = 'std.')
    plt.xlabel('iteration')
    plt.ylabel('std [$\lambda$]')
    plt.legend()
    plt.show() 

def view_field_diff(arr1, arr2, log10: bool = False) -> np.ndarray:
    """
    View the diff(arr1-arr2) and show the image with a colorbar.
    If "log10" is True, the image will be shown with the log10 scale. 

    Args:
        arr1 (np.ndarray, 2D): 2D array. 
        arr2 (np.ndarray, 2D): Another 2D array. 
        log10 (bool, optional): Choose the scale of plotting. Defaults to False. 
        If set to True, the image will be shown with a log10 scale. 

    Returns:
        difference (np.ndarray, 2D): 2D array
    """
    plt.title('Diff')

    if log10 is True:
        imshow_field(np.log10(np.real(arr1 - arr2)))
        plt.colorbar()

    else: 
        imshow_field(np.real(arr1 - arr2))
        plt.colorbar()

    np.testing.assert_array_almost_equal(arr1, arr2)  

def view_arr_diff (arr1, arr2):
    print(' ==== Checking arrays ==== ')
    return np.allclose(arr1, arr2)

def calculate_broadband_image_and_contrast(CAPyBARA_list, coron_images, wvl_weights):
    """
    Calculate mean broadband image and contrast.
    
    Parameters
    ----------
    CAPyBARA_list : list
        List of CAPyBARA simulation instances
    coron_images : list
        Coronagraphic images for each iteration
    wvl_weights : list
        Weights for each wavelength (typically all 1s)
    
    Returns
    -------
    broadband_images : list
        Mean broadband images
    broadband_contrast : list
        Average contrast values
    """
    def ensure_boolean_mask(mask, grid):
        if callable(mask):
            mask = mask(grid)
        return mask.astype(bool)
    
    broadband_images = []
    broadband_contrast = []
    num_wavelengths = len(wvl_weights)
    
    for i in range(len(coron_images)):
        broadband_image = 0.0
        img_iteration = coron_images[i]
        
        # Average over wavelengths
        for wl in range(num_wavelengths):
            ref_img_flat = CAPyBARA_list[wl].ref_img.ravel()
            normalized_image = img_iteration[wl] / np.max(ref_img_flat)
            broadband_image += normalized_image * wvl_weights[wl]
        
        broadband_image /= np.sum(wvl_weights)
        broadband_images.append(broadband_image)
        
        # Calculate contrast in dark zone
        dark_zone_mask = ensure_boolean_mask(
            CAPyBARA_list[0].dark_zone_mask,
            CAPyBARA_list[0].focal_grid
        )
        mask_flat = dark_zone_mask.ravel()
        contrast = np.mean(broadband_image.ravel()[mask_flat])
        broadband_contrast.append(contrast)
    
    return broadband_images, broadband_contrast