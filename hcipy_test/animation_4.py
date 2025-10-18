import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Create a figure and an axes
fig, ax = plt.subplots()
line, = ax.plot([], [], 'r-') # Initialize an empty line

# Set axis limits
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-1.1, 1.1)

# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# Animation function: this is called sequentially
def animate(i):
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x + i / 10.0) # Update data for animation
    line.set_data(x, y)
    return line,

# Create the animation
ani = animation.FuncAnimation(fig, animate, init_func=init, frames=200, interval=20, blit=True)

# Save the animation (requires a writer like 'ffmpeg' or 'imagemagick')
# ani.save('sine_wave.gif', writer='imagemagick', fps=30)

plt.show()

import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
# Assuming gaussian_occulter_generator is defined in this imported file
from gaussian_occulter import gaussian_occulter_generator 

# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- GLOBAL CONSTANTS (ONLY those that define the animation range) ---
aperture_scale = 1.5
grid_size = 256
diameter = 1 # meters
sqrt_contrast = 1e-5 # Amplitude factor (E_p/E_s)
q_factor = 8 # Oversampling factor for the focal plane

# --- ANIMATION SETUP ---
separation_range = np.linspace(2, 25, 50) # Separation in lambda/D units
vmin = -10 # Log contrast minimum
vmax = -4  # Log contrast maximum

# Initialize the figure and image plot
fig, ax = plt.subplots(figsize=(8, 7))

# Initialize a dummy focal grid to set up the plot area size initially
focal_grid_size = grid_size * q_factor 
focal_grid_dummy = hp.make_focal_grid(q=8, num_airy=16)

# Create a dummy Field object from the correctly defined grid for plot initialization
dummy_field = hp.Field(np.zeros(focal_grid_dummy.size), focal_grid_dummy)

im_handle = hp.imshow_field(
    dummy_field, 
    cmap='inferno', 
    vmin=vmin, 
    vmax=vmax,
    ax=ax
)
plt.colorbar(im_handle, label='Contrast ($\log_{10}(I/I_{star})$)')
ax.set_title("Intensity After Lyot Stop (Separation: 0.0 $\lambda/D$)")
ax.set_xlabel('x / D')
ax.set_ylabel('y / D')


# ====================================================================
# --- ANIMATION FUNCTION (ALL CALCULATIONS INSIDE) ---
# ====================================================================

def animate_coronagraph(planet_offset_x):
    """
    Performs the entire coronagraphic path for BOTH star and planet from scratch
    for the current planet_offset_x value. This is highly inefficient but useful for debugging.
    """
    # --- SETUP (RECALCULATED EVERY FRAME) ---
    pupil_grid = hp.make_pupil_grid(grid_size, aperture_scale)
    focal_grid_calculated = hp.make_focal_grid(q=8, num_airy=16)
    prop = hp.FraunhoferPropagator(pupil_grid, focal_grid_calculated)
    
    telescope_pupil = hp.make_circular_aperture(diameter)(pupil_grid)
    wavefront_star = hp.Wavefront(telescope_pupil)
    
    # --- MASKS (RECALCULATED EVERY FRAME) ---
    sigma_lambda_d = 5
    occulter_mask = gaussian_occulter_generator(focal_grid_calculated, sigma_lambda_d)
    occulter_mask = hp.Field(occulter_mask, focal_grid_calculated)

    lyot_stop_ratio = 0.8
    lyot_stop_generator = hp.make_circular_aperture(lyot_stop_ratio * diameter)
    lyot_stop_mask = lyot_stop_generator(pupil_grid)
    
    # Define the full Lyot Coronagraph propagator (F1 -> P2 step)
    prop_lyot = hp.LyotCoronagraph(focal_grid_calculated, occulter_mask, lyot_stop_mask)

    # --- PLANET WAVEFRONT (RECALCULATED EVERY FRAME) ---
    planet_offset_y = 0
    wavefront_planet = hp.Wavefront(
        sqrt_contrast * telescope_pupil 
        * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x / diameter) 
        * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y / diameter)
    )

    # ===========================================================
    # --- STAR PROPAGATION (Coronagraphic Path) ---
    # ===========================================================
    star_P2 = prop_lyot.forward(wavefront_star)
    star_F2 = prop.forward(star_P2) # Star field in final detector plane (F2)

    # ===========================================================
    # --- PLANET PROPAGATION (Coronagraphic Path) ---
    # ===========================================================
    planet_P2 = prop_lyot.forward(wavefront_planet)
    planet_F2 = prop.forward(planet_P2)
    
    # ===========================================================
    # --- FINAL RESULT ---
    # ===========================================================

    # Final Coherent Sum and Intensity (F2)
    E_final_total = star_F2.electric_field + planet_F2.electric_field
    I_final_total = np.abs(E_final_total)**2

    # Normalize and plot update
    I_norm = I_final_total / wavefront_star.intensity.max() 
    log_I_norm = np.log10(I_norm)
    
    # Update the plot data (Reshape is required for matplotlib)
    reshaped_data = log_I_norm.reshape((grid_size, grid_size))
    im_handle.set_data(reshaped_data)
    
    # Update the title
    ax.set_title(f"Intensity in Detector Plane (Separation: {planet_offset_x:.2f} $\lambda/D$)")
    
    # Return the updated artists for blitting
    return im_handle, ax.title 

# ====================================================================
# --- RUN ANIMATION ---
# ====================================================================

# Note: This will be slow due to the repeated calculations inside the function.
ani = FuncAnimation(
    fig, 
    animate_coronagraph, 
    frames=separation_range, 
    blit=True, 
    interval=100
)

plt.show()