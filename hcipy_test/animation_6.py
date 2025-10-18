import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import warnings
from gaussian_occulter import gaussian_occulter_generator
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- GLOBAL CONSTANTS ---
grid_size = 256 # Defined here for use in reshape/normalization
diameter = 1 # Defined here for use in normalization
sqrt_contrast = 1e-5 

# --- ANIMATION PARAMETERS ---
# Varying planet separation from 2 to 25 lambda/D
separation_range = np.linspace(2, 25, 50) 
vmin = -10 # Log contrast minimum for plotting
vmax = 0 # Adjusted vmax for visibility

# Focal grid definition
focal_grid = hp.make_focal_grid(q=8, num_airy=16)

# Initialize the figure and image plot
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111)

# Create a blank plot initially for the animation handle
im_handle = hp.imshow_field(
np.zeros(focal_grid.size), 
grid=focal_grid, 
cmap='inferno', 
vmin=vmin, 
vmax=vmax,
ax=ax
)
plt.colorbar(im_handle, label='Contrast ($\log_{10}(I/I_{star})$)')

# CRITICAL FIX 1: Initialize the standard title and store the artist object.
# This mirrors the successful setup in your Schrödinger code.
title_artist = ax.set_title("Coronagraphic Image (Separation: 0.0 $\lambda/D$)")

ax.set_xlabel('x / D')
ax.set_ylabel('y / D')


def animate_coronagraph(planet_offset_x):
    
    aperture_scale = 1.5
    grid_size = 256
    local_grid_size = 256
    pupil_grid = hp.make_pupil_grid(grid_size,aperture_scale)
    local_diameter = 1 # meters

    telescope_pupil_generator = hp.make_circular_aperture(local_diameter)

    telescope_pupil = telescope_pupil_generator(pupil_grid)

    # define propagator (pupil to focal)
    focal_grid = hp.make_focal_grid(q=8, num_airy=16)
    prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

    # obtain wavefront at telescope pupil plane for the star
    wavefront_star = hp.Wavefront(telescope_pupil)

    # obtain wavefront at telescope pupil plane for the planet
    local_sqrt_contrast = 1e-5 # Planet-to-star contrast

    # Planet offset in units of lambda/D (Using corrected phase term for stability)
    planet_offset_y = 0
    wavefront_planet = hp.Wavefront(
        local_sqrt_contrast * telescope_pupil 
        * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x / local_diameter) 
        * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y / local_diameter)
    )

    # ... (Coronagraph calculations remain the same) ...
    wavefront_total_intensity = wavefront_star.intensity + wavefront_planet.intensity
    focal_star = prop.forward(wavefront_star)
    focal_planet = prop.forward(wavefront_planet)
    focal_total_intensity = focal_star.intensity + focal_planet.intensity
    sigma_lambda_d = 5
    occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
    occulter_mask = hp.Field(occulter_mask,focal_grid)
    E_focal_total = focal_star.electric_field + focal_planet.electric_field
    E_focal_after_occulter = E_focal_total * occulter_mask
    I_focal_after_occulter = np.abs(E_focal_after_occulter)**2
    prop_no_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask)
    star_occulter_no_lyot = prop_no_lyot.forward(wavefront_star)
    planet_occulter_no_lyot = prop_no_lyot.forward(wavefront_planet)
    total_intensity_occulter_no_lyot = star_occulter_no_lyot.intensity + planet_occulter_no_lyot.intensity
    ratio = 0.8 # Lyot Stop diameter ratio
    lyot_stop_generator = hp.make_circular_aperture(ratio*local_diameter)
    lyot_stop_mask = lyot_stop_generator(pupil_grid)
    prop_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask,lyot_stop_mask)
    star_occulter_lyot = prop_lyot.forward(wavefront_star)
    planet_occulter_lyot = prop_lyot.forward(wavefront_planet)
    total_intensity_occulter_lyot = star_occulter_lyot.intensity + planet_occulter_lyot.intensity
    wavefront_focal_after_occulter_star = prop.forward(star_occulter_lyot)
    wavefront_focal_after_occulter_planet = prop.forward(planet_occulter_lyot)
    E_final_total = wavefront_focal_after_occulter_star.electric_field + wavefront_focal_after_occulter_planet.electric_field
    wavefront_focal_after_occulter_total_intensity = np.abs(E_final_total)**2

    # Normalize and convert to log scale
    I_norm = wavefront_focal_after_occulter_total_intensity / wavefront_focal_after_occulter_total_intensity.max()
    log_I_norm = np.log10(I_norm)

    # Reshape for matplotlib's set_data()
    reshaped_data = log_I_norm.reshape((local_grid_size, local_grid_size))
    
    # Update the plot handle with the new data
    im_handle.set_data(reshaped_data)

    # CRITICAL FIX 2: Update the standard title artist using set_text()
    new_title = f"Intensity After Lyot Stop (Separation: {planet_offset_x:.2f} $\lambda/D$)"
    title_artist.set_text(new_title)

    # CRITICAL FIX 3: Return the image handle AND the standard title artist.
    return im_handle, title_artist

# Create the animation over planet_offset_x variable
ani = FuncAnimation(
fig, 
animate_coronagraph, 
frames=separation_range, 
blit=True, 
interval=10 
)

plt.show()