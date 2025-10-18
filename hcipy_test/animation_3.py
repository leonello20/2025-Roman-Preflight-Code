import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
# Assuming gaussian_occulter_generator is defined in this imported file
from gaussian_occulter import gaussian_occulter_generator 

# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ====================================================================
# --- 1. GLOBAL CONSTANTS & STATIC SETUP (Calculated ONCE) ---
# ====================================================================

aperture_scale = 1.5
grid_size = 256
diameter = 1 # meters
sqrt_contrast = 1e-5 # Amplitude factor (E_p/E_s). If C=1e-10, this is 1e-5.
sigma_lambda_d = 5 # Gaussian width
lyot_stop_ratio = 0.8
q_factor = 8 # Oversampling factor for the focal plane

# Define Grids and Propagators
pupil_grid = hp.make_pupil_grid(grid_size, aperture_scale)

# CORRECTED FOCAL GRID DEFINITION: Calculate the size N = q * N_pupil
focal_grid_size = grid_size * q_factor # 256 * 8 = 2048
focal_grid = hp.make_focal_grid(q=8, num_airy=16) 
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# Define Star Wavefront and Static Masks
telescope_pupil = hp.make_circular_aperture(diameter)(pupil_grid)
wavefront_star = hp.Wavefront(telescope_pupil)

occulter_mask = gaussian_occulter_generator(focal_grid, sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask, focal_grid)

lyot_stop_generator = hp.make_circular_aperture(lyot_stop_ratio * diameter)
lyot_stop_mask = lyot_stop_generator(pupil_grid)

# Define the full Lyot Coronagraph propagator (F1 -> P2 step)
prop_lyot = hp.LyotCoronagraph(focal_grid, occulter_mask, lyot_stop_mask)

# Pre-calculate Star's FINAL Electric Field (The static contrast floor)
star_P2 = prop_lyot.forward(wavefront_star)
star_F2 = prop.forward(star_P2) # Star field in final detector plane (F2)


# --- ANIMATION SETUP ---
separation_range = np.linspace(2, 25, 50) # Separation in lambda/D units
vmin = -10 # Log contrast minimum
vmax = -4  # Log contrast maximum

# Initialize the figure and image plot
fig, ax = plt.subplots(figsize=(8, 7))
# Create a dummy Field object from the correctly defined grid for plot initialization
dummy_field = hp.Field(np.zeros(focal_grid.size), focal_grid)

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
# --- 2. ANIMATION FUNCTION (Only variable steps inside) ---
# ====================================================================

def animate_coronagraph(planet_offset_x):
    """
    Performs the entire coronagraphic path for the planet and updates the final plot.
    """
    
    # 1. Obtain wavefront at telescope pupil plane for the planet (P1)
    planet_offset_y = 0
    wavefront_planet = hp.Wavefront(
        sqrt_contrast * telescope_pupil 
        * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x / diameter) 
        * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y / diameter)
    )

    # 2. Propagate Planet through the Lyot Coronagraph (F1 -> P2, including Occulter and Lyot Stop)
    planet_P2 = prop_lyot.forward(wavefront_planet) 

    # 3. Propagate P2 to the FINAL DETECTOR PLANE (F2)
    planet_F2 = prop.forward(planet_P2)
    
    # 4. Final Coherent Sum and Intensity (F2)
    E_final_total = star_F2.electric_field + planet_F2.electric_field
    I_final_total = np.abs(E_final_total)**2

    # 5. Normalize and plot update
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
# --- 3. RUN ANIMATION ---
# ====================================================================

ani = FuncAnimation(
    fig, 
    animate_coronagraph, 
    frames=separation_range, 
    blit=True, 
    interval=100
)

# To display the animation interactively (usually works in IDEs/scripts):
plt.show()

# To save the animation (uncomment and install ffmpeg if needed):
# ani.save('coronagraph_separation_animation.mp4', writer='ffmpeg')