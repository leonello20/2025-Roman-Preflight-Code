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
