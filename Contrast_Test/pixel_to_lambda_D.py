# pixel_to_lambda_D.py
import numpy as np
import proper
def pixel_to_lambda_D(wavefront, diam_m, grid_size):
    """
    Convert pixel units to lambda/D units.

    Parameters:
    pixels (int): Number of pixels.
    wavelength_m (float): Wavelength in meters.
    diam_m (float): Diameter of the telescope in meters.

    Returns:
    float: Equivalent distance in lambda/D units.
    """
    # Calculate angular separation for sampling of the wavefront (radians)
    pixel_scale_radians = proper.prop_get_sampling_radians(wavefront)

    # Create pixel array for the grid size
    pixels = np.arange(grid_size//2)

    # Retrieve wavelength from the wavefront
    lambda_m = proper.prop_get_wavelength(wavefront)

    # Calculate lambda/D separation
    lambda_over_D = lambda_m / diam_m

    # Calculate conversion factor from pixels to lambda/D
    pixel_scale_lambda_D = pixel_scale_radians / lambda_over_D

    # Convert pixel units to lambda/D units
    lambda_D_separation = pixel_scale_lambda_D * pixels
    
    return lambda_D_separation