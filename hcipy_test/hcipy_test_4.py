import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import hcipy as hp
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







# create the Gaussian occulter mask
sigma_lambda_d = 5
sigma_meter = sigma_lambda_d*wavelength*focal_length/pupil_diameter
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_meter)
occulter_mask = hp.Field(occulter_mask,focal_grid)

# after lens 2 but before Lyot Stop
prop_no_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask)
wf_occulter_no_lyot = prop_no_lyot.forward(wf_sm)

# plot the pupil (Lyot) plane intensity with occulter and no Lyot Stop (pupil (Lyot) plane, after lens 2) (not a log scale)
hp.imshow_field(wf_occulter_no_lyot.intensity/wf_occulter_no_lyot.intensity.max())
plt.colorbar(label='Contrast ($I/I_{total}$)')
plt.title("Pupil (Lyot) Plane Intensity (Occulter, No Lyot Stop) (Lyot Plane, After Lens 2)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# 2. CRITICAL FIX: Segmented Lyot Stop Definition
# We re-create the segmented aperture, but slightly undersized to block diffracted light from segment edges/gaps.
lyot_stop_ratio = 0.7
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

# create the occulter mask and Lyot Stop in the Lyot Coronagraph
# ratio = 0.8 # Lyot Stop diameter ratio
# lyot_stop_generator = hp.make_circular_aperture(ratio*pupil_diameter) # percentage of the telescope diameter
# lyot_stop_mask = lyot_stop_generator(pupil_grid)
prop_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask,lyot_stop_mask)
wf_occulter_lyot = prop_lyot.forward(wf_sm)

# plot the focal plane intensity (star + planet) with occulter and Lyot Stop (Lyot (pupil) plane, after lens 2)
hp.imshow_field(np.log10(wf_occulter_lyot.intensity/wf_occulter_lyot.intensity.max()))
plt.colorbar(label='Contrast ($\log_{10}(I/I_{toal})$)')
plt.title("Final Coronagraphic Image (Occulter + Lyot Stop) (Lyot Plane, After Lens 2)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# propagate the wavefront to the focal plane (after lens 3)
wavefront_focal_after_occulter_star = prop.forward(wf_occulter_lyot)

# plot the focal plane intensity after Lyot Stop
hp.imshow_field(np.log10(wavefront_focal_after_occulter_star.intensity/wavefront_focal_after_occulter_star.intensity.max()))
plt.colorbar(label='Contrast ($\log_{10}(I/I_{total})$)')
plt.title("Intensity After Lyot Stop (Focal Plane, After Lens 3)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# call the contrast curve function

# contrast_curve(wavefront_star,focal_grid,prop,wavefront_focal_after_occulter_star,planet_offset_x,sigma_lambda_d)



"""

# --- STEP 1: DEFINE THE DARK HOLE MASK (DH) ---
# Define the Dark Hole region in lambda/D
IWA_ld = 3.0 # Inner Working Angle
OWA_ld = 20.0 # Outer Working Angle
y_extent_ld = 20.0 # Vertical extent (+/- 20 lambda/D)

airy_disk_radius_m = wavelength * focal_length / pupil_diameter
wavel_D_m = airy_disk_radius_m

# Convert lambda/D units to meters on the focal grid
x_min_m = -OWA_ld * wavel_D_m
x_max_m = OWA_ld * wavel_D_m
y_min_m = -y_extent_ld * wavel_D_m
y_max_m = y_extent_ld * wavel_D_m

# CORRECTED: Use coordinate comparison to define the rectangular bounding box
dh_mask_rect = (focal_grid.x >= x_min_m) * (focal_grid.x <= x_max_m) * \
               (focal_grid.y >= y_min_m) * (focal_grid.y <= y_max_m)

# Exclude Inner Working Angle (IWA)
focal_grid_r = np.sqrt(focal_grid.x**2 + focal_grid.y**2)
dh_mask_iwa = (focal_grid_r / wavel_D_m > IWA_ld)

# Combine the masks
dh_mask = dh_mask_rect * dh_mask_iwa


# --- PROPAGATE INITIAL WAVEFRONT TO GET INITIAL DH E-FIELD ---

# Propagate initial WF through the coronagraph
wf_occulter_lyot_initial = prop_lyot.forward(wf)
wf_focal_initial = prop.forward(wf_occulter_lyot_initial)

# STEP 2: Extract the E-field vector E_DH from the Dark Hole
E_DH = wf_focal_initial.electric_field[dh_mask]

# print(f"Total number of DM segments: {hsm.num_segments}")
print(f"DM actuators per segment (Piston, Tip, Tilt): 3")
print(f"Total number of DM actuators: {hsm.num_actuators}")
print(f"Number of points in the Dark Hole E-field vector (E_DH): {E_DH.size}")


# --- STEP 3: CALCULATE FIRST G-MATRIX COLUMN (Probe Actuator 1: Piston) ---

probe_segment_index = 1 
probe_piston_m = aber_to_opd(0.1, wavelength) # Small piston poke (0.1 rad, in meters)

# The E-field of the unperturbed star (which includes the speckle-generating aberrations)
E_unperturbed_star = wf_focal_initial.electric_field[dh_mask]

# Perform the poke on the DM (Segment 1 Piston)
hsm.set_segment_actuators(probe_segment_index, probe_piston_m, 0, 0)

# Propagate the perturbed WF through the coronagraph
wf_perturbed = prop_lyot.forward(wf)
wf_focal_perturbed = prop.forward(wf_perturbed)

# Extract the perturbed E-field vector E_DH
E_perturbed_star = wf_focal_perturbed.electric_field[dh_mask]

# Calculate the first G-column
# G_i = (E_perturbed - E_unperturbed) / delta_a_i
G_column_1 = (E_perturbed_star - E_unperturbed_star) / probe_piston_m

print(f"\n--- G-MATRIX CALCULATION STATUS ---")
print(f"Calculated G-matrix column for Segment {probe_segment_index} (Piston mode).")
print(f"Size of G_column_1: {G_column_1.size} (Matches E_DH size)")


# --- PLOTTING FINAL DH SETUP FOR VISUALIZATION ---
plt.figure(figsize=(10, 10))
# Plot the log contrast of the initial E-field
log_I_contrast = np.log10(wf_focal_initial.intensity / norm_hc)

# We are using LogNorm now for better visual range mapping
hp.imshow_field(log_I_contrast, grid=focal_grid, cmap='inferno', vmin=-10, vmax=10)
plt.colorbar(label=r'Contrast ($\log_{10}(I/I_{star})$)')
plt.title("Initial Coronagraphic Image with Dark Hole Defined")
plt.xlabel(r'x ($\lambda/\mathrm{D}$)')
plt.ylabel(r'y ($\lambda/\mathrm{D}$)')

# Overlay the Dark Hole mask outline
plt.contour(
    focal_grid.x.reshape(focal_grid.dims) / wavel_D_m,
    focal_grid.y.reshape(focal_grid.dims) / wavel_D_m,
    dh_mask.reshape(focal_grid.dims),
    levels=[0.5],
    colors='cyan',
    linestyles='dashed'
)

plt.show()

"""


"""
# create the Gaussian occulter mask
sigma_lambda_d = 0.000002
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask,focal_grid)

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
ratio = 5 # Lyot Stop diameter ratio
lyot_stop_generator = hp.make_circular_aperture(ratio*pupil_diameter) # percentage of the telescope diameter
lyot_stop_mask = lyot_stop_generator(pupil_grid)
prop_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask,lyot_stop_mask)
star_occulter_lyot = prop_lyot.forward(wavefront_star)
planet_occulter_lyot = prop_lyot.forward(wavefront_planet)
total_intensity_occulter_lyot = star_occulter_lyot.intensity + planet_occulter_lyot.intensity

# plot the focal plane intensity (star + planet) with occulter and Lyot Stop (Lyot (pupil) plane, after lens 2)
hp.imshow_field(np.log10(total_intensity_occulter_lyot/total_intensity_occulter_lyot.max()))
plt.colorbar(label='Contrast ($\log_{10}(I/I_{toal})$)')
plt.title("Final Coronagraphic Image (Occulter + Lyot Stop) (Lyot Plane, After Lens 2)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# propagate the wavefront to the focal plane (after lens 3)
wavefront_focal_after_occulter_star = prop.forward(star_occulter_lyot)
wavefront_focal_after_occulter_planet = prop.forward(planet_occulter_lyot)
wavefront_focal_after_occulter_total_intensity = wavefront_focal_after_occulter_star.intensity + wavefront_focal_after_occulter_planet.intensity

# plot the focal plane intensity (star + planet) after Lyot Stop
hp.imshow_field(np.log10(wavefront_focal_after_occulter_total_intensity/wavefront_focal_after_occulter_total_intensity.max()))
plt.colorbar(label='Contrast ($\log_{10}(I/I_{total})$)')
plt.title("Intensity After Lyot Stop (Focal Plane, After Lens 3)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# call the contrast curve function

contrast_curve(wavefront_star,focal_grid,prop,wavefront_focal_after_occulter_total_intensity,planet_offset_x,sigma_lambda_d)
"""