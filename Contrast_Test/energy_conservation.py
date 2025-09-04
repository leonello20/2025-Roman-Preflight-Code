# energy_conservation.py
import numpy as np
import proper

def energy(wfo):
    """
    Calculate the total energy in the wavefront.

    Parameters:
    - wfo: 2D numpy array representing the wavefront.

    Returns:
    - total_energy: The total energy calculated as the sum of the squared magnitudes of the wavefront.
    """
    wfo_amplitude = proper.prop_get_amplitude(wfo)  # Ensure wfo is a numpy array
    wfo_intensity = np.abs(wfo_amplitude)**2  # Intensity is the squared magnitude of the amplitude
    
    # Sum over all pixels to get total energy
    total_energy = np.sum(wfo_intensity)
    return total_energy