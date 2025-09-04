# contrast_2D.py
import numpy as np
import proper
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
def contrast_2D(unaberrated_wfo, final_wfo, grid_size):
    """
    Calculates the 2D contrast map from the unaberrated and final wavefronts.
    
    Parameters:
    - unaberrated_wfo: 2D numpy array of the unaberrated star's wavefront.
    - final_wfo: 2D numpy array of the final coronagraphic wavefront.
    - grid_size: The size of the simulation grid (e.g., 512).
    
    Returns:
    - contrast_map: A 2D array of contrast values.
    """
    unaberrated_wfo = proper.prop_get_amplitude(unaberrated_wfo)  # Ensure unaberrated_wfo is a numpy array
    final_wfo = proper.prop_get_amplitude(final_wfo)  # Ensure final_wfo is a numpy array

    # Calculate the intensity of the final wavefront
    final_intensity = np.abs(final_wfo)**2
    # Calculate the maximum intensity of the unaberrated wavefront (normalization)
    peak_unaberrated_intensity = np.max(np.abs(unaberrated_wfo)**2)
    
    # Calculate the contrast map
    contrast_map = final_intensity / peak_unaberrated_intensity

    # Plot the contrast map
    plt.figure(figsize=(10, 8))
    plt.imshow(contrast_map, cmap='hot', norm=LogNorm(), extent=(-grid_size/2, grid_size/2, -grid_size/2, grid_size/2))
    plt.colorbar(label='Contrast')
    plt.title('2D Contrast Map')
    plt.xlabel('X Position (pixels)')
    plt.ylabel('Y Position (pixels)')
    plt.show()
    return