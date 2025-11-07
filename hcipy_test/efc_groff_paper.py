import numpy as np
import hcipy as hp
import matplotlib.pyplot as plt

# --- 1. Simulation and Optical Parameters ---
N_segments = 36          # Number of segments (36 segments = 108 modes TTP)
num_modes = 3           # Modes per segment (Piston, Tip, Tilt)
N_actuators = N_segments * num_modes # Total number of modes/actuators
pupil_grid_size = 1024    # Resolution of the pupil plane grid
wavelength = 632.8e-9    # meters
pupil_diameter = 0.01975     # meters
f_number = 30            # Effective f-number for propagation
propagate_distance = f_number * pupil_diameter

# --- 2. Defining Grids ---
pupil_grid = hp.make_pupil_grid(pupil_grid_size, pupil_diameter)
# Focal grid setup for the image plane (x,y)
# We choose 64 lambda/D across the focal grid for good coverage
q = 8  # Sampling factor
num_airy = 64  # Extent of the focal grid in lambda/D
focal_grid = hp.make_focal_grid(q=q, num_airy=num_airy)

# --- 3. Optical Components A(u,v) and Propagation ---
# A(u,v): The Aperture Function (ideal, continuous part of the pupil)
aperture = hp.make_circular_aperture(pupil_diameter)(pupil_grid)
A = aperture.copy() # The A(u,v) term

# Propagation operator C{} (Fourier Transform)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# --- 4. Segmented Mirror Setup (for phi_0) ---
# Create the 36-segment geometry

gap_size = 90e-6 # m
num_rings = 3
segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)
focal_length = 1 # m
wavelength = 638e-9 # m

# --- STEP 1: DEFINE THE MECHANICAL STRUCTURE (Aperture and DM) ---

# Define the segmented aperture mask and segment indices
aper, segments = hp.make_hexagonal_segmented_aperture(num_rings,
                                                         segment_flat_to_flat,
                                                         gap_size,
                                                         starting_ring=1,
                                                         return_segments=True)
aper = hp.evaluate_supersampled(aper, pupil_grid, 1)
segments = hp.evaluate_supersampled(segments, pupil_grid, 1)

# Segment modes are Piston, Tip, Tilt (Zernike 1, 2, 3)
# segment_modes = hp.make_zernike_basis(3, 1, pupil_grid, starting_mode=1)
# Segmented Deformable Mirror (hsm)
# hsm = hp.SegmentedDeformableMirror(segments, segment_modes=segment_modes)
hsm = hp.SegmentedDeformableMirror(segments)
hsm.flatten() # Reset to flat first

# --- 6. Initial DM State (k=0): Creating phi_0(u,v) ---
# We set the DM to a small, random shape to ensure E_im,0 is not zero (phi_0 != 0)
# Random DM commands (actuator heights, in meters)
# Desired Actuator RMS command (in meters)
actuator_rms_meters = 200e-6 # 100 microns RMS

# Create a 3x36 array representing the commands for [Piston, Tip, Tilt] x [36 Segments]
# Shape: (3, 36)
structured_actuators = np.random.uniform(
    low=-actuator_rms_meters, 
    high=actuator_rms_meters, 
    size=(num_modes, N_segments) 
)

plt.figure(figsize=(8,8))
plt.title('OPD for HCIPy SM')
hp.imshow_field(aper, cmap='gray')
plt.colorbar()
plt.show()
# set the DM to this initial random shape
piston_factor = 1000
N_segments_array = np.arange(1, N_segments + 1) # Segment IDs start at 1

for i in N_segments_array:
    piston = structured_actuators[0, i - 1]
    tip = structured_actuators[1, i - 1]
    tilt = structured_actuators[2, i - 1]
    hsm.set_segment_actuators(i - 1, piston/piston_factor, tip, tilt)

# HCIPy
plt.figure(figsize=(8,8))
plt.title('OPD for HCIPy SM')
hp.imshow_field(hsm.surface * 2, mask=aper, cmap='RdBu_r')
plt.colorbar()
plt.show()

# --- 5. Initial Aberration g(u,v) ---
# This models the static phase error g(u,v). We use random Zernikes.
zernike_modes = hp.make_zernike_basis(40, pupil_diameter, pupil_grid)
aberrations = 50e-9 # 50 nm RMS
aberration_coefficients = np.random.uniform(-aberrations, aberrations, 40) # 50nm RMS aberration
aberration_opd = (zernike_modes.linear_combination(aberration_coefficients))
aberration_opd = aberration_opd * aperture # Mask to the pupil

phi_0_opd = hsm.surface # This is H(u,v)

"""
# Calculate the initial total phase field (phi_total = phi_aberration + phi_0)
# Phase is 4*pi*H/lambda for a reflective surface
phi_aberration_total = (aberration_opd + phi_0_opd) * 4 * np.pi / wavelength

# --- 7. Calculate the Pupil Wavefront E_pup,0 ---
# E_pup,0 = A(u,v) * e^(i * phi_total)
# The wavefront contains the complex E-field information.

wavefront = hp.Wavefront(A, wavelength)
initial_phase_screen = Phase(phi_aberration_total)
wavefront_pupil_0 = initial_phase_screen.forward(wavefront)

# --- 8. Calculate the Image Plane E_im,0 ---
# E_im,0 = C{E_pup,0}
wavefront_image_0 = prop.forward(wavefront_pupil_0)
E_im_0 = wavefront_image_0.electric_field

# --- 9. Output Visualization and Summary ---
initial_intensity = wavefront_image_0.intensity
I_max = initial_intensity.max()
I_dark_zone = initial_intensity[np.abs(focal_grid.x) > 4 * (wavelength/pupil_diameter)].mean()
initial_contrast = I_dark_zone / I_max

# Create a small plot of the initial aberrated image
plt.figure(figsize=(6, 5))
plt.title(f"Initial Image (E_im,0, Intensity)")
hp.imshow_field(np.log10(initial_intensity / I_max), cmap='inferno', vmin=-7, vmax=0)
plt.colorbar(label='log10(Normalized Intensity)')
plt.show()

print("--- EFC Initialization Summary ---")
print(f"Total Actuator Modes (N_actuators): {N_actuators}")
print(f"DM Phase Perturbation (phi_0) RMS: {np.std(phi_0_opd) * 1e9:.2f} nm")
print(f"Initial Aberration (g) RMS: {np.std(aberration_opd) * 1e9:.2f} nm")
print(f"Maximum Initial Intensity (Raw): {I_max:.2e}")
print(f"Initial Contrast (Mean DZ): {initial_contrast:.2e}")
"""