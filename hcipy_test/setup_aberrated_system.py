import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def setup_aberrated_system(grid_size, pupil_diameter):
    """
    Sets up the segmented DM, introduces random aberrations, and returns 
    the essential system components.
    """
    # Parameters for the pupil function
    aperture_ratio = 2.0
    gap_size = 90e-6 # m
    num_rings = 3
    segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)
    focal_length = 1 # m

    # Parameters for the simulation
    wavelength = 638e-9
    num_airy = 20
    sampling = 4
    norm = False

    # HCIPy grids and propagator
    pupil_grid = hp.make_pupil_grid(dims=grid_size, diameter=pupil_diameter)
    focal_grid = hp.make_focal_grid(sampling, num_airy, pupil_diameter=pupil_diameter, reference_wavelength=wavelength,focal_length=focal_length)
    focal_grid = focal_grid.shifted(focal_grid.delta / 2)
    prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

    aper, segments = hp.make_hexagonal_segmented_aperture(num_rings, segment_flat_to_flat, gap_size, starting_ring=1, return_segments=True)

    aper = hp.evaluate_supersampled(aper, pupil_grid, 1)
    segments = hp.evaluate_supersampled(segments, pupil_grid, 1)

    plt.title('HCIPy aperture')
    hp.imshow_field(aper, cmap='gray')
    plt.show()

    # Instantiate the segmented mirror
    hsm = hp.SegmentedDeformableMirror(segments)

    # Get the list of all unique segment IDs (e.g., 1 to 19)
    # Get the list of all unique segment IDs (1 to N)
    num_segments = segments.num_modes
    segment_ids = np.arange(1, num_segments + 1)

    # 2. INTRODUCE RANDOM SEGMENTED ABERRATIONS (The Problem)
    random_pistons_m = np.random.uniform(-20e-9, 20e-9, num_segments)
    hsm.flatten()

    # --- FIX: Use direct actuators array (3*N) for setting the DM state ---

    # The DM actuators are a 3*N vector: [Piston_1..N, Tip_1..N, Tilt_1..N]
    actuator_commands_initial = np.zeros(num_segments * 3)

    # Piston commands occupy the first N indices (0 to N-1)
    actuator_commands_initial[:num_segments] = random_pistons_m 

    # Set the initial state of the DM (the problem it must correct)
    hsm.actuators = actuator_commands_initial
        
    print(f"Introduced random piston errors on {num_segments} segments (± 20 nm range).")

    # Plot the total aberration map
    plt.figure(figsize=(8, 8))
    hp.imshow_field(hsm.surface, mask=aper, cmap='RdBu_r', vmin=-20e-9, vmax=20e-9)
    plt.title('Random Segment Piston Aberration Map')
    plt.colorbar(label='Optical Path Difference (m)')
    plt.show()

    return hsm, aper, pupil_grid, wavelength, focal_length

    """
    initial_aberration_map = random_pistons_m.copy() 
    DM_command_history = [initial_aberration_map.copy()]
    contrast_history = []

    # --- Dark Hole Mask Creation ---
    r = np.sqrt(focal_grid.x**2 + focal_grid.y**2) * pupil_diameter / wavelength / focal_length
    theta = np.arctan2(focal_grid.y, focal_grid.x) * 180 / np.pi # degrees

    DH_INNER_RADIUS = 5 # lambda/D
    DH_OUTER_RADIUS = 20 # lambda/D
    DH_ANGLE_START = 45 # degrees
    DH_ANGLE_END = 135 # degrees
    # Filter 1: Radial Annulus
    radial_mask = (r > DH_INNER_RADIUS) * (r < DH_OUTER_RADIUS)

    # Filter 2: Angular Wedge (Convert angles to be between 0 and 360)
    theta_normalized = (theta + 360) % 360
    angular_mask = (theta_normalized >= DH_ANGLE_START) * (theta_normalized <= DH_ANGLE_END)

    dark_hole_mask = hp.Field(radial_mask * angular_mask, focal_grid)

    plt.figure(figsize=(6, 6))
    hp.imshow_field(dark_hole_mask, cmap='gray')
    plt.title('Dark Hole Region Mask (Used for Loss Function)')
    plt.show()

    # Create the ideal unaberrated light in the pupil plane
    wavefront_star_initial = hp.Wavefront(aper, wavelength)

    # Create the aberrated wavefront by passing it through the DM
    # wavefront_aberrated_pupil = hsm.forward(contrast*wavefront_star_initial)
    # Plot the pupil plane intensity
    plt.figure(figsize=(8, 8))
    hp.imshow_field(wavefront_aberrated_pupil.intensity)
    plt.title('Pupil Plane Intensity with Segment Aberrations')
    plt.colorbar(label='Intensity')
    plt.xlabel('x / D')
    plt.ylabel('y / D')
    plt.show()
    """






"""
sigma_lambda_d = 5
ratio = 0.8 # Lyot Stop diameter ratio
contrast = 1e-10 # Planet-to-star contrast
sqrt_contrast = np.sqrt(contrast)
planet_offset_x = 15
planet_offset_y = 5
planet_offset_x = planet_offset_x/pupil_diameter
planet_offset_y = planet_offset_y/pupil_diameter
# --- Coronagraph Setup ---
occulter_mask = gaussian_occulter_generator(focal_grid, sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask, focal_grid)

lyot_stop_generator = hp.make_circular_aperture(ratio * pupil_diameter)
lyot_stop_mask = lyot_stop_generator(pupil_grid)

prop_lyot = hp.LyotCoronagraph(focal_grid, occulter_mask, lyot_stop_mask)
prop_focal = prop

# Normalize relative to the unocculted star peak for true contrast
I_star_unaberrated_peak = prop_focal.forward(hp.Wavefront(aper)).intensity.max()

# Planet Wavefront (Defined on the segmented aperture)
wavefront_planet = hp.Wavefront(
    sqrt_contrast * aper 
    * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x / pupil_diameter + 2j * np.pi * pupil_grid.y * planet_offset_y / pupil_diameter) 
)

# 3. SPECKLE NULLING OPTIMIZATION LOOP
print("Starting iterative dark hole speckle nulling...")
wavefront_star_no_aberration = hp.Wavefront(aper, wavelength)
current_DM_piston_commands = initial_aberration_map.copy()





# Make a pupil plane wavefront from aperture
wf = hp.Wavefront(aper, wavelength)

# Apply SM if you want to
wf = hsm(wf)

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
"""