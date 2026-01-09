import numpy as np
import matplotlib.colors as mpl
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.io import fits
import scipy.fftpack
from hcipy import *

def extract_component(modes: np.ndarray, wavefront) -> np.ndarray:
    coeffs = np.zeros(len(modes))
    for i in range(len(modes)):
        coeffs[i] = np.nansum(modes[i] * wavefront)
    return coeffs

def make_custom_orthogonal_modes (modal_basis, aperture):
    # Initialize an empty list to store orthogonalized modes
    orthogonal_modes = []
    aperture_mask = aperture > 0  # Mask for the custom pupil

    for mode in modal_basis:
        # Apply the aperture mask to the mode
        mode_values = mode * aperture_mask

        # Subtract projections onto previously orthogonalized modes
        for ortho_mode in orthogonal_modes:
            projection = np.sum(ortho_mode * mode_values) / np.sum(ortho_mode**2)
            mode_values -= projection * ortho_mode

        # Normalize the mode
        norm = np.sqrt(np.sum(mode_values**2))
        mode_values /= norm

        # Add the orthogonalized mode to the list
        orthogonal_modes.append(mode_values)

    return orthogonal_modes

class CustomBasis:
    def __init__(self, svd):
        # Store original SVD
        self.U = svd.U
        self.S = svd.S
        self.Vt = svd.Vt

        # Pre-scale all actuator modes to 1 nm RMS explicitly
        num_modes = self.Vt.shape[0]
        self.V_modes_physical = np.zeros_like(self.Vt.T)

        for i in range(num_modes):
            mode = self.Vt.T[:, i]
            mode /= np.sqrt(np.mean(mode**2))  # normalize to unit RMS
            self.V_modes_physical[:, i] = mode * 1e-9  # scale to 1 nm explicitly