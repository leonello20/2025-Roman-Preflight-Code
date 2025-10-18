import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import warnings

# --- Define Helper Function (Required by your code flow) ---
def gaussian_occulter_generator(grid, sigma_lambda_d):
    """Generates a Gaussian-shaped occulter mask centered at the origin."""
    r_sq = grid.x**2 + grid.y**2
    return np.exp(-r_sq / (2 * sigma_lambda_d**2))
# -----------------------------------------------------------

# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- SYSTEM PARAMETERS ---
aperture_scale = 1.5
grid_size = 256
pupil_grid = hp.make_pupil_grid(grid_size,aperture_scale)
diameter = 1 # meters
sqrt_contrast = 1e-5 
planet_offset_x = 15
planet_offset_y = 0
sigma_lambda_d = 5
ratio = 0.8 # Lyot Stop diameter ratio

telescope_pupil_generator = hp.make_circular_aperture(diameter)
telescope_pupil = telescope_pupil_generator(pupil_grid)

# plot the pupil
hp.imshow_field(telescope_pupil, cmap='gray')
plt.colorbar(label='Amplitude')
plt.title("Telescope Pupil (Before Lens 1)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# define propagator (pupil to focal)
focal_grid = hp.make_focal_grid(q=8, num_airy=16)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# obtain wavefront at telescope pupil plane for the star
wavefront_star = hp.Wavefront(telescope_pupil)

# obtain wavefront at telescope pupil plane for the planet
# Corrected phase tilt calculation (dividing by diameter for dimensional consistency)
wavefront_planet = hp.Wavefront(
    sqrt_contrast * telescope_pupil 
    * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x / diameter) 
    * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y / diameter)
)

# --------------------------------------------------------------------------
# --- NEW: ABERRATION AND DEFORMABLE MIRROR (DM) SECTION ---
# --------------------------------------------------------------------------

# 1. SETUP ZERNIKE BASIS AND CODES
# We will use Zernike modes 4 through 20 (piston=1, tilt=2,3 are typically ignored)
# Zernike modes are indexed from 1 (Piston)
zernike_modes = hp.make_zernike_basis(num_modes=20, D=diameter, grid=pupil_grid, starting_mode=4)

# 2. INTRODUCE A STATIC ABERRATION (THE PROBLEM)
# This simulates a fixed error in the telescope (e.g., poor primary mirror alignment).
# We introduce 10 nm RMS of primary coma (Zernike mode 7, Noll index 7)
coma_amplitude_nm = 10 
c_aberration = np.zeros(zernike_modes.num_modes)
# Zernike mode 4 is the starting mode (Defocus, Noll index 4)
# Primary Coma (Noll 7) is index 4 in our list (modes 4, 5, 6, 7 -> indices 0, 1, 2, 3)
c_aberration[3] = coma_amplitude_nm * 1e-9 # Set 10 nm for Zernike mode 7 (Primary Coma)

# Create the aberration phase map (in radians)
aberration_phase = zernike_modes.linear_combination(c_aberration)

# Apply the aberration to the star wavefront
wavefront_star_aberrated = wavefront_star.add_phase(aberration_phase)


# 3. DEFINE DM CORRECTION (THE SOLUTION)
# The DM is used to counteract the aberration. For a perfect correction, 
# the DM coefficients are the negative of the aberration coefficients.
c_dm = -c_aberration 

# Create the DM phase map (the phase sheet applied by the mirror)
dm_surface_phase = zernike_modes.linear_combination(c_dm)

# Apply the DM correction to the aberrated star wavefront
wavefront_star_corrected = wavefront_star_aberrated.add_phase(dm_surface_phase)

# --- The rest of the simulation uses the corrected wavefront ---

# obtain total wavefront intensity at pupil plane
wavefront_total_intensity = wavefront_star_corrected.intensity + wavefront_planet.intensity

# obtain the wavefront intensity at focal plane for the star
focal_star = prop.forward(wavefront_star_corrected) # Use CORRECTED wavefront here

# obtain the wavefront intensity at focal plane for the planet
focal_planet = prop.forward(wavefront_planet)

# obtain total wavefront intensity at focal plane
focal_total_intensity = focal_star.intensity + focal_planet.intensity

# plot the focal plane intensity (star + planet) (before occulter, after lens 1)
hp.imshow_field(np.log10(focal_total_intensity/focal_total_intensity.max()))
plt.colorbar(label='Contrast ($\log_{10}(I/I_{total})$)')
plt.title("Intensity (Focal Plane, Before Occulter, After Lens 1) - Aberrated/Corrected")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# create the Gaussian occulter mask
sigma_lambda_d = 5
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask,focal_grid)

# plot the focal plane intensity (star + planet) with occulter (focal plane, after lens 1)
E_focal_total = focal_star.electric_field + focal_planet.electric_field

# apply the occulter mask (Field * Field multiplication IS SUPPORTED for Field/Field on the same grid)
E_focal_after_occulter = E_focal_total * occulter_mask

# Calculate the intensity for plotting (Intensity = |E|^2)
I_focal_after_occulter = np.abs(E_focal_after_occulter)**2

# plot the focal plane intensity (star + planet) with occulter (focal plane, after lens 1) (not a log scale)
hp.imshow_field(I_focal_after_occulter/I_focal_after_occulter.max())
plt.colorbar(label='Contrast ($I/I_{total}$)')
plt.title("Intensity (Focal Plane, AFTER Occulter)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# after lens 2 but before Lyot Stop
prop_no_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask)
star_occulter_no_lyot = prop_no_lyot.forward(wavefront_star)
planet_occulter_no_lyot = prop_no_lyot.forward(wavefront_planet)
total_intensity_occulter_no_lyot = star_occulter_no_lyot.intensity + planet_occulter_no_lyot.intensity

# plot the pupil (Lyot) plane intensity (star + planet) with occulter and no Lyot Stop (pupil (Lyot) plane, after lens 2) (not a log scale)
hp.imshow_field(total_intensity_occulter_no_lyot/total_intensity_occulter_no_lyot.max())
plt.colorbar(label='Contrast ($I/I_{total}$)')
plt.title("Pupil (Lyot) Plane Intensity (Occulter, No Lyot Stop) (Lyot Plane, After Lens 2)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# create the occulter mask and Lyot Stop in the Lyot Coronagraph
ratio = 0.8 # Lyot Stop diameter ratio
lyot_stop_generator = hp.make_circular_aperture(ratio*diameter) # percentage of the telescope diameter
lyot_stop_mask = lyot_stop_generator(pupil_grid)

# create the occulter mask and Lyot Stop in the Lyot Coronagraph
prop_lyot = hp.LyotCoronagraph(focal_grid, occulter_mask, lyot_stop_mask)
star_occulter_lyot = prop_lyot.forward(wavefront_star_corrected) # Use CORRECTED wavefront here
planet_occulter_lyot = prop_lyot.forward(wavefront_planet)

# propagate the wavefront to the focal plane (after lens 3)
wavefront_focal_after_occulter_star = prop.forward(star_occulter_lyot)
wavefront_focal_after_occulter_planet = prop.forward(planet_occulter_lyot)
E_final_total = wavefront_focal_after_occulter_star.electric_field + wavefront_focal_after_occulter_planet.electric_field
wavefront_focal_after_occulter_total_intensity = np.abs(E_final_total)**2

# plot the focal plane intensity (star + planet) after Lyot Stop
hp.imshow_field(np.log10(wavefront_focal_after_occulter_total_intensity/wavefront_focal_after_occulter_total_intensity.max()))
plt.colorbar(label='Contrast ($\log_{10}(I/I_{total})$)')
plt.title("Final Intensity AFTER DM Correction (Focal Plane, After Lens 3)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# --------------------------------------------------------------------------
# --- APPENDED SECTION: GENERATE 1D CONTRAST CURVE ---
# --------------------------------------------------------------------------

# 1. Calculate the true normalization factor 
I_star_unocculted = prop.forward(wavefront_star).intensity # Use *UN-ABERRATED* star for normalization
I_star_peak = I_star_unocculted.max()

# 2. Calculate Log Contrast relative to the unocculted star peak
log_contrast = np.log10(wavefront_focal_after_occulter_total_intensity / I_star_peak)

# 3. Extract the slice parameters
grid_dimension = int(np.sqrt(log_contrast.size))
center_index = grid_dimension // 2
log_contrast_2D = log_contrast.reshape((grid_dimension, grid_dimension))
x_focal_2D = focal_grid.x.reshape((grid_dimension, grid_dimension))
contrast_slice = log_contrast_2D[center_index, :]
x_slice = x_focal_2D[center_index, :]

# 4. Filter for the right half (x/D > 0)
x_plot = x_slice[center_index:]
I_plot = contrast_slice[center_index:]

# 5. Plot the 1D Contrast Curve

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x_plot, I_plot, 
        label=f'Contrast Profile (DM Corrected)', 
        linewidth=2, 
        color='#17becf') # Cyan for corrected result

# Highlight the planet location
ax.axvline(x=planet_offset_x, color='r', linestyle='--', alpha=0.6, label=f'Planet at {planet_offset_x:.1f} $\lambda/D$')

# --- Styling ---
ax.set_ylim(-11, -3) 
ax.set_xlim(x_plot.min(), x_plot.max())

ax.set_xlabel('Angular Separation ($x / \lambda D$)', fontsize=14)
ax.set_ylabel('Log Contrast ($\log_{10}(I / I_{star,peak})$)', fontsize=14)
ax.set_title(f"DM-Corrected Contrast Curve ($\sigma = {sigma_lambda_d:.1f} \lambda/D$)", fontsize=16)
ax.grid(True, which="both", ls="--", alpha=0.5)
ax.legend()
plt.tight_layout()

plt.show()