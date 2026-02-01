# main.py
import numpy as np
import matplotlib.pyplot as plt
import proper
from pixel_to_lambda_D import pixel_to_lambda_D
from contrast_curve import contrast_curve
from occulter import occulter
from contrast_2D import contrast_2D

wavelength = 0.5e-6  # Wavelength in meters (500 nm)
grid_size = 1024  # Grid size for the simulation
occulter_type = 'GAUSSIAN'  # Type of occulter to use
diam_telescope = 2  # Telescope diameter in meters
occrad = 1.0  # Occulter radius in lambda/D units

(unaberrated_star_wfo, sampling) = occulter(wavelength, diam_telescope, grid_size, occrad, PASSVALUE={'occulter_type': 'NONE'})
(final_wavefront_coronagraph, sampling) = occulter(wavelength, diam_telescope, grid_size, occrad, PASSVALUE={'occulter_type': occulter_type})

# lambda/D separation for the central horizontal cross-section
lambda_D_separation = pixel_to_lambda_D(final_wavefront_coronagraph, diam_telescope, grid_size)
# Calculate contrast curve
contrast = contrast_curve(unaberrated_star_wfo, final_wavefront_coronagraph, grid_size)

# Plot Contrast Curve

plt.figure(figsize=(12, 8))
plt.title("Contrast Curve")
plt.xlabel("Angular Separation ($\\lambda/D$)")
plt.ylabel("Contrast")
plt.semilogy(lambda_D_separation, contrast)
plt.show()

# 2D Contrast Map
contrast_2D(unaberrated_star_wfo, final_wavefront_coronagraph, grid_size)