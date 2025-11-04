import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import hcipy as hp
from matplotlib import animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from gaussian_occulter import gaussian_occulter_generator
from contrast_curve import contrast_curve
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

pupil_diameter = 0.019725 # m
gap_size = 90e-6 # m
num_rings = 3
segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)
focal_length = 1 # m

# Parameters for the simulation
num_pix = 1024
wavelength = 638e-9
num_airy = 20
sampling = 4
norm = False

# HCIPy grids and propagator
pupil_grid = hp.make_pupil_grid(dims=num_pix, diameter=pupil_diameter)

focal_grid = hp.make_focal_grid(sampling, num_airy,
                                   pupil_diameter=pupil_diameter,
                                   reference_wavelength=wavelength,
                                   focal_length=focal_length)
focal_grid = focal_grid.shifted(focal_grid.delta / 2)

prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

aper, segments = hp.make_hexagonal_segmented_aperture(num_rings,
                                                         segment_flat_to_flat,
                                                         gap_size,
                                                         starting_ring=1,
                                                         return_segments=True)

aper = hp.evaluate_supersampled(aper, pupil_grid, 1)
segments = hp.evaluate_supersampled(segments, pupil_grid, 1)

plt.title('HCIPy aperture')
hp.imshow_field(aper, cmap='gray')
plt.colorbar()
plt.show()

# Instantiate the segmented mirror
hsm = hp.SegmentedDeformableMirror(segments)

# Make a pupil plane wavefront from aperture
wavefront_star = hp.Wavefront(aper, wavelength=wavelength)

# Apply SM if you want to
wavefront_star = hsm(wavefront_star)

# Contrast
contrast = 1e-4 # Planet-to-star contrast
sqrt_contrast = np.sqrt(contrast)

# Planet offset in units of lambda/D
planet_offset_x = 15
planet_offset_y = 0
planet_offset_x = planet_offset_x/pupil_diameter
planet_offset_y = planet_offset_y/pupil_diameter
wavefront_planet = hp.Wavefront(sqrt_contrast * aper * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x) * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y))

wf = hp.Wavefront(wavefront_star.electric_field + wavefront_planet.electric_field, wavelength=wavelength)

plt.figure(figsize=(8, 8))
plt.title('Wavefront intensity at HCIPy SM')
hp.imshow_field(wf.intensity, cmap='gray')
plt.colorbar()
plt.show()

# Apply SM to pupil plane wf
wf_sm = hsm(wf)

# Propagate from SM to image plane
im_ref_hc = prop(wf_sm)

# Display intensity and phase in image plane
plt.figure(figsize=(8, 8))
plt.suptitle('Image plane after HCIPy SM')

# Get normalization factor for HCIPy reference image
norm_hc = np.max(im_ref_hc.intensity)

hp.imshow_psf(im_ref_hc, normalization='peak')
plt.show()

# Flatten both SMs just to be sure
hsm.flatten()

# Define function from rad of phase to m OPD
def aber_to_opd(aber_rad, wavelength):
    aber_m = aber_rad * wavelength / (2 * np.pi)
    return aber_m

aber_rad = 4.0

print('Aberration: {} rad'.format(aber_rad))
print('Aberration: {} m'.format(aber_to_opd(aber_rad, wavelength)))

# Poking segment 35 and 25
for i in [35, 25]:
    hsm.set_segment_actuators(i, aber_to_opd(aber_rad, wavelength) / 2, 0, 0)

# Display both segmented mirrors in OPD

# HCIPy
plt.figure(figsize=(8,8))
plt.title('OPD for HCIPy SM')
hp.imshow_field(hsm.surface * 2, mask=aper, cmap='RdBu_r', vmin=-5e-7, vmax=5e-7)
plt.colorbar()
plt.show()

### HCIPy
# Apply SM to pupil plane wf
wf_fp_pistoned = hsm(wf)
plt.figure(figsize=(15, 6))
plt.suptitle('Pupil plane after SM for $\phi$ = ' + str(aber_rad) + ' rad')

hp.imshow_field(np.log10(wf_fp_pistoned.intensity / np.max(wf_fp_pistoned.intensity)), cmap='inferno', vmin=-1)
plt.title('HCIPy pistoned pair')
plt.colorbar()
plt.show()

# Propagate from SM to image plane
im_pistoned_hc = prop(wf_fp_pistoned)

### Display intensity of image plane
plt.figure(figsize=(15, 6))
plt.suptitle('Image plane after SM for $\phi$ = ' + str(aber_rad) + ' rad')

hp.imshow_field(np.log10(im_pistoned_hc.intensity / norm_hc), cmap='inferno', vmin=-9)
plt.title('HCIPy pistoned pair')
plt.colorbar()
plt.show()






# Run a full coronagraph simulation with HCIPy on the aberrated wavefront



# --- ANIMATION PARAMETERS ---
sigma_range = np.linspace(0, 1, 21)
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
plt.colorbar(im_handle, label='Contrast ($\log_{10}(I/I_{total})$)')
title = ax.set_title("")
ax.set_xlabel('x / D')
ax.set_ylabel('y / D')
fps = 10 # Frames per second for the output video
filename = 'dm_gaussian_occulter_animation.mp4' # Output filename

def animate_coronagraph_gaussian_sigma(lyot_stop_ratio):
    # create the Gaussian occulter mask
    sigma_lambda_d = 2
    sigma_meter = sigma_lambda_d*wavelength*focal_length/pupil_diameter
    occulter_mask = gaussian_occulter_generator(focal_grid,sigma_meter)
    occulter_mask = hp.Field(occulter_mask,focal_grid)

    # after lens 2 but before Lyot Stop
    prop_no_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask)
    wf_occulter_no_lyot = prop_no_lyot.forward(wf_sm)

    # 2. CRITICAL FIX: Segmented Lyot Stop Definition
    # We re-create the segmented aperture, but slightly undersized to block diffracted light from segment edges/gaps.
    # lyot_stop_ratio = 0.7
    lyot_segment_flat_to_flat = segment_flat_to_flat * lyot_stop_ratio
    lyot_gap_size = gap_size * lyot_stop_ratio # Scale the gaps as well

    lyot_stop_mask, _ = hp.make_hexagonal_segmented_aperture(
        num_rings,
        lyot_segment_flat_to_flat,
        lyot_gap_size,
        starting_ring=1,
        return_segments=True
    )
    # Evaluate the Lyot Stop mask on the pupil grid
    lyot_stop_mask = hp.evaluate_supersampled(lyot_stop_mask, pupil_grid, 1)
    prop_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask,lyot_stop_mask)
    wf_occulter_lyot = prop_lyot.forward(wf_sm)

    # propagate the wavefront to the focal plane (after lens 3)
    wavefront_focal_after_occulter_star = prop.forward(wf_occulter_lyot)

    log_I_norm = np.log10(wavefront_focal_after_occulter_star.intensity / wavefront_focal_after_occulter_star.intensity.max())
    reshaped_data = log_I_norm.reshape(focal_grid.dims)

    # Update the plot handle with the new data
    im_handle.set_data(reshaped_data)

    # Update the title
    title.set_text(r"Coronagraphic Image (Lyot Stop Ratio: " + f"{lyot_stop_ratio:.2f}" + r")")

    # Return the updated artists for blitting
    return im_handle,

# Create the animation over planet_offset_x variable
ani = FuncAnimation(
    fig,
    animate_coronagraph_gaussian_sigma, 
    frames=sigma_range, # Use the list of separations as frames
    blit=False, 
    interval=500 # milliseconds between frames
)
plt.show()

# call the contrast curve function

# contrast_curve(wavefront_star,focal_grid,prop,wavefront_focal_after_occulter_star,planet_offset_x,sigma_lambda_d)