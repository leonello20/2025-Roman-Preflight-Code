# contrast_curve.py
import numpy as np
import proper
def contrast_curve(unaberrated_wfo, final_wfo, grid_size):
    """
    Calculates the raw contrast curve from the central horizontal cross-section.
    
    Parameters:
    - unaberrated_wfo: 2D numpy array of the unaberrated star's wavefront.
    - final_wfo: 2D numpy array of the final coronagraphic wavefront.
    - grid_size: The size of the simulation grid (e.g., 512).
    
    Returns:
    - contrast_slice: A 1D array of contrast values along the cross-section.
    """
    unaberrated_wfo = proper.prop_get_amplitude(unaberrated_wfo)  # Ensure unaberrated_wfo is a numpy array

    final_wfo = proper.prop_get_amplitude(final_wfo)  # Ensure final_wfo is a numpy array

    half_grid_size = grid_size // 2
    final_wfo_slice = final_wfo[half_grid_size, half_grid_size:]  # Take the second half of the row

    # Calculate the intensity of the final wavefornt
    final_intensity_slice = np.abs(final_wfo_slice)**2
    # Calculate the maximum intensity of the unaberrated wavefront (normalization)
    peak_unaberrated_intensity = np.max(np.abs(unaberrated_wfo)**2)
    # Calculate the contrast
    contrast = final_intensity_slice / peak_unaberrated_intensity

    return contrast