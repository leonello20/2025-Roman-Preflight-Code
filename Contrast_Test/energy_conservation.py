# energy_conservation.py
import numpy as np
import matplotlib.pyplot as plt
import proper

def energy(wfo, title=""):
    """
    Calculate the total energy in the wavefront and plots intensity distribution.

    Parameters:
    - wfo: 2D numpy array representing the wavefront.

    Returns:
    - total_energy: The total energy calculated as the sum of the squared magnitudes of the wavefront.
    """
    wfo_amplitude = proper.prop_get_amplitude(wfo)  # Ensure wfo is a numpy array
    wfo_intensity = np.abs(wfo_amplitude)**2  # Intensity is the squared magnitude of the amplitude
    
    # Sum over all pixels to get total energy
    total_energy = np.sum(wfo_intensity)

    gridsize = proper.prop_get_gridsize(wfo)
    x_axis = np.arange(gridsize)
    y_axis = np.arange(gridsize)
    
    plt.figure(figsize=(8, 8))
    plt.contourf(x_axis, y_axis, wfo_intensity, levels=50)
    plt.title(title)
    plt.xlabel("X Pixels")
    plt.ylabel("Y Pixels")
    plt.colorbar(label=r"Intensity $W/m^2$")
    plt.show()

    return total_energy