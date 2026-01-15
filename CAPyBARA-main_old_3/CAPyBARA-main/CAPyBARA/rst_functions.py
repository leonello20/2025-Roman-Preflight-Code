#!/usr/bin/python
#RST functions

import numpy as np
import matplotlib.colors as mpl
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.io import fits
import scipy.fftpack
from hcipy import *
from tqdm import tqdm

## TODO - only use image3
def get_jacobian_matrix(CAPyBARA, reference=0.0, eps=10e-9):
    responses = []
    amps = np.linspace(-eps, eps, 2)
    num_modes = 2 * len(CAPyBARA.influence_function)
    wvl = CAPyBARA.param['wvl']

    print('Current wavelength', wvl)

    # Initialize the progress bar
    with tqdm(total=num_modes, desc="Computing Jacobian Matrix", ncols=100, leave=True) as pbar:
        for i, mode in enumerate(np.eye(num_modes)):
            # Update the progress bar for each iteration without printing a new line
            pbar.update(1)
            pbar.set_postfix_str(f"Mode {i+1}/{num_modes}", refresh=False)  # Updates in-place

            response = 0
            for amp in amps:
                response += amp * CAPyBARA.get_image(
                    current_aberration=None, 
                    wvl=CAPyBARA.param['wvl']*1e-9, 
                    actuators=mode * amp + reference, 
                    include_aberration=None
                ).electric_field

            response /= np.var(amps)
            response = response[CAPyBARA.dark_zone_mask]

            responses.append(np.concatenate((response.real, response.imag)))

    jacobian_matrix = np.array(responses).T

    return jacobian_matrix

def perturbation_evolution(perturbations, ppl_grid, radius_ppl_px = 336):
    """Make the perturbation evolve in a .mp4 file and datacube

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
    # turbulence_sequence = FFMpegWriter(title, framerate=5)
    perturbation_datacube = np.zeros((len(perturbations), radius_ppl_px*2, radius_ppl_px*2))
    for i in range(len(perturbations)):
        perturb = Field(perturbations[i], ppl_grid)
        plt.clf()
        imshow_field(perturb, vmax = 0.2, vmin = -0.2) # vmax=0.001, vmin=-0.001)
        plt.colorbar()
        # turbulence_sequence.add_frame()
        perturbation_datacube[i,:,:] = perturb.shaped
    plt.close()
    # turbulence_sequence.close()

    return perturbation_datacube

def get_average_contrast(CAPyBARA, arr, is_mono=False):
    def ensure_boolean_mask(mask):
        """ Ensure the mask is a boolean array and has the correct shape. """
        if callable(mask):
            # If the mask is a function, evaluate it (assuming the grid is available)
            mask = mask(CAPyBARA.focal_grid)  # Adjust if grid changes
        return mask.astype(bool)

    # Ensure dark_zone_mask is boolean and flattened
    # CAPyBARA.dark_zone_mask = ensure_boolean_mask(CAPyBARA.dark_zone_mask)
    mask_flat = CAPyBARA.dark_zone_mask.ravel()

    # Ensure ref_img is flattened and 1D
    ref_img_flat = CAPyBARA.ref_img.ravel()

    if is_mono:
        # Check shape compatibility for monochromatic case
        average_contrast = np.asarray([
            np.mean(img.ravel()[mask_flat] / ref_img_flat.max()) for img in arr
        ])
    else:
        # For broadband, CAPyBARA is expected to be a list of simulations
        average_contrast = []
        num_iterations = CAPyBARA[0].param['num_iteration']  # Get number of iterations

        for i in range(num_iterations):
            for j in range(len(CAPyBARA)):
                CAPyBARA[j].dark_zone_mask = ensure_boolean_mask(CAPyBARA[j].dark_zone_mask)

                # Flatten the mask and ref_img if needed
                mask_flat = CAPyBARA[j].dark_zone_mask.ravel()
                ref_img_flat = CAPyBARA[j].ref_img.ravel()

                _contrast = np.asarray([
                    np.mean(img.ravel()[mask_flat] / ref_img_flat.max()) for img in arr[i]
                ])
                average_contrast.append(_contrast)

    return average_contrast
