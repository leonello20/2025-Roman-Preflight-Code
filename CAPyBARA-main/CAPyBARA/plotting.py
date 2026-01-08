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

def save2mp4 (): 

    pass

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

def plot_phase_as_opd(phase, wavelength, title='OPD Map'):
    """
    Convert a phase map (in radians) to OPD (in nm), plot it, and print the RMS.

    Args:
        phase (ndarray): Phase map (radians).
        wavelength (float): Wavelength in meters.
        title (str): Title for the plot.
    """
    # Convert phase to OPD (meters)
    opd_m = (phase * wavelength) / (2 * np.pi)

    # Convert OPD to nanometers
    opd_nm = opd_m * 1e9

    # Compute RMS in nm
    opd_rms_nm = np.sqrt(np.mean(opd_nm**2))
    print(f"RMS OPD: {opd_rms_nm:.2f} nm")

    # Plotting
    plt.figure(figsize=(6, 5))
    im = plt.imshow(opd_nm, cmap='RdBu_r')
    plt.colorbar(im, label='OPD (nm)')
    plt.title(f"{title} (RMS = {opd_rms_nm:.2f} nm)")
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    plt.tight_layout()
    plt.show()
