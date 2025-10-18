import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib.animation import FuncAnimation
from gaussian_occulter import gaussian_occulter_generator

# NOTE: You need the gaussian_occulter_generator accessible.
# Since it's in a separate file, we assume 'from gaussian_occulter import gaussian_occulter_generator' works.
# If not, you'll need to define it here:
# def gaussian_occulter_generator(grid,sigma_lambda_d): ... (paste your function here)

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- GLOBAL PARAMETERS (from your code) ---
aperture_scale = 1.5
grid_size = 256
diameter = 1 # meters
sqrt_contrast = 1e-5 # Amplitude factor (E_p/E_s). If C=1e-10, this is 1e-5.
sigma_lambda_d = 5 # Fixed Gaussian width for this animation

# Define Grids and Propagator
pupil_grid = hp.make_pupil_grid(grid_size, aperture_scale)
focal_grid = hp.make_focal_grid(q=8, num_airy=16)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# Define Masks (Occulter and Lyot Stop are fixed for this animation)
occulter_mask = gaussian_occulter_generator(focal_grid, sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask, focal_grid)

lyot_stop_ratio = 0.8
lyot_stop_generator = hp.make_circular_aperture(lyot_stop_ratio * diameter)
lyot_stop_mask = lyot_stop_generator(pupil_grid)

# Define the full Lyot Coronagraph propagator
prop_lyot = hp.LyotCoronagraph(focal_grid, occulter_mask, lyot_stop_mask)

# Define Star Wavefront (constant)
telescope_pupil = hp.make_circular_aperture(diameter)(pupil_grid)
wavefront_star = hp.Wavefront(telescope_pupil)
# Propagate star through the coronagraph once to get the static starlight floor
star_occulter_lyot = prop_lyot.forward(wavefront_star)

# --- ANIMATION PARAMETERS ---
# Varying planet separation from 2 to 25 lambda/D
separation_range = np.linspace(2, 25, 50) 
vmin = -10 # Log contrast minimum for plotting (adjust based on your sqrt_contrast)
vmax = 0  # Log contrast maximum for plotting

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
ax.set_title("Coronagraphic Image (Separation: 0.0 $\lambda/D$)")
ax.set_xlabel('x / D')
ax.set_ylabel('y / D')

def animate_coronagraph(planet_offset_x):
    """
    Calculates and updates the final focal plane intensity for a given planet separation.
    """
    
    # 1. Update the Planet Wavefront (P1)
    # The phase ramp is based on the current planet_offset_x value
    wavefront_planet = hp.Wavefront(
        sqrt_contrast * telescope_pupil * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x / diameter) * np.exp(2j * np.pi * pupil_grid.y * 0 / diameter) # y-offset is zero
    )
    
    # 2. Propagate Planet through the Coronagraph (Prop_lyot handles F1 -> P2 -> F2)
    planet_occulter_lyot = prop_lyot.forward(wavefront_planet)
    
    # 3. Final Coherent Sum and Intensity in F2 (Detector)
    E_final_total = star_occulter_lyot.electric_field + planet_occulter_lyot.electric_field
    I_final_total = np.abs(E_final_total)**2
    
    # 4. Normalize and convert to log scale for plotting
    # We normalize by the star's peak intensity (which is suppressed) for the contrast floor
    I_norm = I_final_total / wavefront_star.intensity.max() 
    log_I_norm = np.log10(I_norm)

    reshaped_data = log_I_norm.reshape((grid_size, grid_size))
    
    # 5. Update the plot data
    im_handle.set_data(reshaped_data) # Use the 2D reshaped data
    ax.set_title(f"Coronagraphic Image (Separation: {planet_offset_x:.2f} $\lambda/D$)")
    
    return im_handle,

# Create the animation over planet_offset_x variable
ani = FuncAnimation(
    fig, 
    animate_coronagraph, 
    frames=separation_range, # Use the list of separations as frames
    blit=True, 
    interval=100 # milliseconds between frames
)

# NOTE: To see the animation, you need to save it or display it.
# If running in a Jupyter environment, uncomment the first line:
# from IPython.display import HTML
# HTML(ani.to_jshtml()) 
# If running in a script, you can save it:
# ani.save('coronagraph_separation_animation.mp4', writer='ffmpeg')

plt.show() # Call plt.show() to start the interactive plot (usually works in IDEs)