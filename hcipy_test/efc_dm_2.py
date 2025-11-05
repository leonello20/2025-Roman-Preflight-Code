import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os
import time
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- FFMPEG CONFIGURATION ---
# Manually specify the path to your ffmpeg.exe
# ** YOU MUST UPDATE THIS PATH TO MATCH YOUR SYSTEM **
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
# ----------------------------


# --- MOCKING / HELPER FUNCTIONS ---
def gaussian_occulter_generator(grid, sigma):
    """Generates a Gaussian Occulter mask, calculating r robustly."""
    r = np.sqrt(grid.x**2 + grid.y**2) 
    return 1 - np.exp(-(r**2) / (2 * sigma**2))

def aber_to_opd(aber_rad, wavelength):
    aber_m = aber_rad * wavelength / (2 * np.pi)
    return aber_m

def propagate_coronagraph(wf_input, occulter_mask, lyot_stop_mask, prop, prop_inverse):
    """Performs the full 5-step Lyot Coronagraph propagation manually."""
    wf_focal_initial = prop.forward(wf_input)
    wf_focal_masked = hp.Wavefront(wf_focal_initial.electric_field * occulter_mask)
    wf_lyot_plane = prop_inverse.forward(wf_focal_masked)
    wf_lyot_masked = hp.Wavefront(wf_lyot_plane.electric_field * lyot_stop_mask)
    wf_final_image = prop.forward(wf_lyot_masked)
    return wf_final_image

def build_g_matrix(hsm, dh_mask, wf_initial_flat, probe_amplitude_m, propagate_coronagraph, occulter_mask, lyot_stop_mask, prop, prop_inverse):
    """
    Builds the G-matrix (Influence Matrix) for the EFC algorithm.
    It returns G_matrix and the baseline unperturbed Electric Field vector (E_unperturbed).
    
    NOTE: hsm.actuators is assumed to be a 1D array: [P1, T1, T1, P2, T2, T2, ...]
    """
    # Store the initial 1D flat command vector
    initial_actuators_1d = hsm.actuators.copy() 

    wf_aberrated_baseline = hsm(wf_initial_flat)
    
    wf_focal_unperturbed_star = propagate_coronagraph(wf_aberrated_baseline, occulter_mask, lyot_stop_mask, prop, prop_inverse)
    E_unperturbed = wf_focal_unperturbed_star.electric_field[dh_mask]
    
    # FIX: Correctly derive N_actuators and N_segments from the 1D array size
    N_actuators = initial_actuators_1d.size
    N_segments = int(N_actuators / 3) 
    N_DH_pixels = E_unperturbed.size
    G_matrix = np.zeros((N_DH_pixels, N_actuators), dtype=np.complex128)
    
    print(f"Starting G-matrix calculation: {N_actuators} actuators, {N_DH_pixels} pixels in DH.")
    
    actuator_index = 0
    # FIX: Iterate over segment indices 0 to N_segments-1
    for segment_index in range(N_segments):
        start_index = segment_index * 3 # Start index of [P, T, T] for this segment

        # Iterate over the Piston (0), Tip (1), and Tilt (2) modes
        for mode_name_index in range(3): 
            
            # Create a temporary poked 1D array for this specific poke
            poked_actuators_1d = initial_actuators_1d.copy()
            
            # Poke the specific mode (Piston=0, Tip=1, Tilt=2)
            poke_target_index = start_index + mode_name_index
            poked_actuators_1d[poke_target_index] += probe_amplitude_m
            probe_amplitude_used = probe_amplitude_m

            # Apply the poked command to the DM
            hsm.actuators = poked_actuators_1d
            
            wf_perturbed = hsm(wf_initial_flat) 
            wf_focal_perturbed = propagate_coronagraph(wf_perturbed, occulter_mask, lyot_stop_mask, prop, prop_inverse)
            E_perturbed = wf_focal_perturbed.electric_field[dh_mask]
            
            G_column = (E_perturbed - E_unperturbed) / probe_amplitude_used
            G_matrix[:, actuator_index] = G_column
            actuator_index += 1

    # Restore DM state using the 1D array
    hsm.actuators = initial_actuators_1d.copy()
    
    # Return the *real* Jacobian matrix and the *real* E-field vector
    jacobian_real = np.concatenate((G_matrix.real, G_matrix.imag), axis=0)
    E_vec_real = np.concatenate((E_unperturbed.real, E_unperturbed.imag))

    return jacobian_real, E_vec_real

# --- EFC LOOP FUNCTION (Refactored) ---
def run_efc_segmented(hsm, dh_mask, wf_star_flat, wavefront_planet, jacobian_real, E_vec_initial, num_iterations=10, efc_loop_gain=0.3, rcond=1e-1):
    
    print("Calculating EFC matrix (pseudo-inverse)...")
    efc_matrix = hp.inverse_tikhonov(jacobian_real, rcond=rcond)
    
    # Use the 1D actuators property for EFC math
    current_actuators_1d = hsm.actuators.copy()
    
    # History lists to store data for animation (N+1 points: initial state + N corrections)
    actuators_history = []
    electric_fields_history = []
    images_history = []
    
    # --- 0. CAPTURE INITIAL STATE (Iteration 0) ---
    actuators_history.append(current_actuators_1d.copy())
    
    # 1. Propagate the initial aberrated state for visualization
    wf_star_initial = hsm(wf_star_flat)
    wf_focal_star_initial = propagate_coronagraph(wf_star_initial, occulter_mask, lyot_stop_mask, prop, prop_inverse)
    
    # 2. Store the TOTAL (Star + Planet) field for visualization history
    wf_total_initial = hp.Wavefront(wf_star_initial.electric_field + wavefront_planet.electric_field, wavelength=wavelength)
    wf_focal_total_initial = propagate_coronagraph(wf_total_initial, occulter_mask, lyot_stop_mask, prop, prop_inverse)

    electric_fields_history.append(wf_focal_total_initial.electric_field)
    images_history.append(wf_focal_total_initial.intensity)
    
    # norm_hc = wf_star_corrected.total_power
    norm_ref = img_ref.max()
    initial_contrast = np.mean(wf_focal_star_initial.intensity[dh_mask] / norm_ref)
    print(f"Initial Contrast (Iteration 0): {initial_contrast:.2e}")
    # ------------------------------------------------------------------------

    print(f"Starting EFC loop for {num_iterations} correction steps...")
    
    # EFC loop runs for N steps, producing N new command sets (N+1 total history)
    for i in range(num_iterations):
        print(f"--- Correction Step {i+1} / {num_iterations} ---")
        
        # --- 1. MEASURE E-FIELD from CURRENT DM STATE (Star Only) ---
        # We assume hsm.actuators is set to the state from the previous step (or initial state for i=0)
        wf_star_for_efc = hsm(wf_star_flat)
        wf_focal_star_for_efc = propagate_coronagraph(wf_star_for_efc, occulter_mask, lyot_stop_mask, prop, prop_inverse)
        
        # Extract the complex E-field from the focal plane dark hole
        E_vec_complex = wf_focal_star_for_efc.electric_field[dh_mask]
        
        # Combine real and imaginary parts to form the real vector E_vec_current_star
        E_vec_current_star = np.concatenate((E_vec_complex.real, E_vec_complex.imag))
        
        # --- 2. CALCULATE CORRECTION (delta_a) ---
        # The EFC algorithm calculates the command that drives E_vec_current_star to zero.
        delta_a = efc_matrix.dot(E_vec_current_star)
        
        # --- 3. APPLY CORRECTION and UPDATE STATE ---
        # Apply the correction scaled by the loop gain (efc_loop_gain = 0.5)
        current_actuators_1d -= efc_loop_gain * delta_a
        
        # Store the new actuator commands for the contrast plot history
        actuators_history.append(current_actuators_1d.copy())
        
        # --- 4. PROPAGATE NEW STATE for Visualization/Contrast Logging ---
        hsm.actuators = current_actuators_1d # Apply new commands to DM object
        
        # Propagate the corrected star field (needed for contrast and for mixing with planet)
        wf_star_corrected = hsm(wf_star_flat)
        
        # Total WF for visualization/history (Star + Planet)
        wf_total_corrected = hp.Wavefront(wf_star_corrected.electric_field + wavefront_planet.electric_field, wavelength=wavelength)
        wf_focal_total = propagate_coronagraph(wf_total_corrected, occulter_mask, lyot_stop_mask, prop, prop_inverse)

        # Store history for visualization
        electric_fields_history.append(wf_focal_total.electric_field)
        images_history.append(wf_focal_total.intensity)
        
        # LOGGING: Calculate the Star-Only contrast for immediate feedback
        # NOTE: This uses the star-only field calculated for EFC, which is the previous wf_focal_star_for_efc.
        # Alternatively, we could re-propagate the star-only field using the new DM commands:
        wf_focal_star_corrected_for_log = propagate_coronagraph(wf_star_corrected, occulter_mask, lyot_stop_mask, prop, prop_inverse)
        contrast_at_end_of_step = np.mean(wf_focal_star_corrected_for_log.intensity[dh_mask] / norm_ref)

        print(f"  Star-Only Contrast: {contrast_at_end_of_step:.2e}")
        
    return actuators_history, electric_fields_history, images_history

# --- SIMULATION SETUP ---
pupil_diameter = 0.019725 # m
gap_size = 90e-6 # m
num_rings = 3
segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)
focal_length = 1 # m
wavelength = 638e-9 # m

num_pix = 1024
num_airy = 20
sampling = 4
aber_rad = 6.0 # Aberration phase in radians
probe_amplitude_m = aber_to_opd(0.01, wavelength) / 2
probe_amplitude_m = 10e-9 # 10 nm poke for segmented DM

wavel_D_m = wavelength * focal_length / pupil_diameter
spatial_resolution = wavel_D_m # For animation plotting

# --- GRIDS AND PROPAGATORS ---
pupil_grid = hp.make_pupil_grid(dims=num_pix, diameter=pupil_diameter)
focal_grid = hp.make_focal_grid(sampling, num_airy,
                                        pupil_diameter=pupil_diameter,
                                        reference_wavelength=wavelength,
                                        focal_length=focal_length)
focal_grid = focal_grid.shifted(focal_grid.delta / 2)

prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)
prop_inverse = hp.FraunhoferPropagator(focal_grid, pupil_grid, focal_length)

# --- DM AND ABERRATION (FIXED APERTURE DEFINITION) ---
hex_aperture_func, segments = hp.make_hexagonal_segmented_aperture(num_rings, segment_flat_to_flat, gap_size, starting_ring=1, return_segments=True)
# This is the correct segmented aperture mask
aperture = hp.evaluate_supersampled(hex_aperture_func, pupil_grid, 1)

# R-coordinate fix location 1 (Aperture)
# pupil_r = np.sqrt(pupil_grid.x**2 + pupil_grid.y**2)
# aperture = hp.Field(np.exp(-(pupil_r / (0.5 * pupil_diameter))**30), pupil_grid)

segments = hp.evaluate_supersampled(segments, pupil_grid, 1)
deformable_mirror = hp.SegmentedDeformableMirror(segments) # This is the global DM for animation
deformable_mirror.flatten()

"""
# Introduce the aberration (piston)
opd_m_aberration = aber_to_opd(aber_rad, wavelength)
piston_poke_m_aberration = opd_m_aberration / 2

for i in [35, 25]:
    deformable_mirror.set_segment_actuators(i, piston_poke_m_aberration, 0, 0)
"""
    
# --- CRITICAL FIX: APPLY RANDOM ABERRATION FOR VISIBLE INITIAL CONTRAST ---
aberration_rms_nm = 10.0  # RMS of 100 nm is common for speckle generation
aberration_rms_m = aberration_rms_nm * 1e-9

# Create random piston commands for all segments
# num_segments = len(segments.defined_segments)
num_segments = segments.num_modes
# Random values, centered at zero, with the desired RMS
random_piston = np.random.normal(0, aberration_rms_m, num_segments)

# Apply the random piston to the DM (Piston is every 3rd element in the 1D actuator array)
actuators_1d = deformable_mirror.actuators.copy()
piston_indices = np.arange(0, len(actuators_1d), 3)
actuators_1d[piston_indices] = random_piston

# --- STAR + PLANET WAVEFRONT ---
planet_offset_ld = 15 
planet_offset_x_radians = planet_offset_ld * (wavelength / pupil_diameter)
planet_offset_y_radians = 0.0
contrast = 1e-10 
sqrt_contrast = np.sqrt(contrast) 

wf_star_flat = hp.Wavefront(aperture, wavelength=wavelength)

# Use HCIPy's built-in plane wave function for the planet
wavefront_planet = hp.Wavefront(aperture, wavelength)
wavefront_planet.total_power = contrast
wavefront_planet.electric_field *= np.exp(1j * 2 * np.pi * (pupil_grid.x * np.sin(planet_offset_x_radians) + pupil_grid.y * np.sin(planet_offset_y_radians)) / wavelength)


# --- CORONAGRAPH MASKS SETUP ---
sigma_lambda_d = 0.5
sigma_meter = sigma_lambda_d * wavel_D_m
occulter_mask_values = gaussian_occulter_generator(focal_grid, sigma_meter)
occulter_mask = hp.Field(occulter_mask_values, focal_grid)

lyot_stop_ratio = 0.9
lyot_segment_flat_to_flat = segment_flat_to_flat * lyot_stop_ratio
lyot_gap_size = gap_size * lyot_stop_ratio
lyot_stop_mask, _ = hp.make_hexagonal_segmented_aperture(num_rings, lyot_segment_flat_to_flat, lyot_gap_size, starting_ring=1, return_segments=True)
lyot_stop_mask = hp.evaluate_supersampled(lyot_stop_mask, pupil_grid, 1)

# --- DARK HOLE (DH) SETUP ---
IWA_ld = 3.0 
OWA_ld = 20.0
y_extent_ld = 20.0

# R-coordinate fix location 2 (Dark Hole IWA)
focal_r = np.sqrt(focal_grid.x**2 + focal_grid.y**2)
dh_mask_iwa = (focal_r / wavel_D_m > IWA_ld)

x_min_m = -OWA_ld * wavel_D_m
x_max_m = OWA_ld * wavel_D_m
y_min_m = -y_extent_ld * wavel_D_m
y_max_m = y_extent_ld * wavel_D_m
dh_mask_rect = (focal_grid.x >= x_min_m) * (focal_grid.x <= x_max_m) * (focal_grid.y >= y_min_m) * (focal_grid.y <= y_max_m)

dark_zone = dh_mask_rect * dh_mask_iwa # Use 'dark_zone' to match animation

# --- Reference Normalization ---
im_ref_unaberrated = prop.forward(hp.Wavefront(aperture, wavelength=wavelength))
img_ref = im_ref_unaberrated.intensity # Use 'img_ref' to match animation

# --- 1. CALCULATE G-MATRIX ---
start_time = time.time()
jacobian, E_vec_initial = build_g_matrix(deformable_mirror, dark_zone, wf_star_flat, probe_amplitude_m, 
                                        propagate_coronagraph, occulter_mask, lyot_stop_mask, prop, prop_inverse)
end_time = time.time()
print(f"G-Matrix calculated in {end_time - start_time:.2f} seconds.")

# --- 2. RUN ITERATIVE EFC LOOP ---
actuators, electric_fields, images = run_efc_segmented(
    deformable_mirror, dark_zone, wf_star_flat, wavefront_planet, 
    jacobian, E_vec_initial, num_iterations=10, efc_loop_gain=0.5
)

# --- 3. CONTRAST HISTORY FOR PLOTTING (STAR-ONLY) ---
"""
def calculate_star_only_contrast_history(actuators, wf_star_flat, dh_mask, norm_ref):
    # Calculates the contrast for the contrast plot using the stored actuator commands.
    contrasts = []
    # Create a local copy of the DM to avoid corrupting the main DM state
    local_dm = deformable_mirror.copy() 
    
    for a in actuators:
        local_dm.actuators = np.array(a)
        wf_star = local_dm(wf_star_flat)
        wf_focal_star = propagate_coronagraph(wf_star, occulter_mask, lyot_stop_mask, prop, prop_inverse)
        contrasts.append(np.mean(wf_focal_star.intensity[dh_mask] / norm_ref))
    return contrasts

# FIX: Use the dedicated helper function to get the Star-Only contrast history
average_contrast = calculate_star_only_contrast_history(actuators, wf_star_flat, dark_zone, img_ref.max())
num_iterations = len(actuators) # Should be 11 (0 to 10)
iteration_range = np.linspace(0, num_iterations-1, num_iterations, dtype=int)
"""

# --- 4. RUN ANIMATION ---
fig = plt.figure(figsize=(12, 10))
anim = FFMpegWriter(fps=2, metadata=dict(artist='HCIPy Segmented EFC Loop'))

num_iterations = len(actuators)
average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]
electric_field_norm = mpl.colors.LogNorm(10**-7, 10**(-3), True) # Adjusted norm for EFC
iteration_range = np.linspace(0, num_iterations-1, num_iterations, dtype=int)

def make_animation_1dm(iteration):
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.title('Electric field (log intensity)')
    electric_field = electric_fields[iteration]
    temp_wf = hp.Wavefront(electric_field, wavelength=wavelength)
    hp.imshow_field(np.log10(temp_wf.intensity / img_ref.max()), norm=None, grid_units=spatial_resolution, vmin=-10, vmax=-5)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    plt.subplot(2, 2, 2)
    plt.title('Intensity image (log scale)')
    hp.imshow_field(np.log10(images[iteration] / img_ref.max()), grid_units=spatial_resolution, cmap='inferno', vmin=-12, vmax=-7)
    plt.colorbar()
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    plt.subplot(2, 2, 3)
    # The DM command history is a flat array, so we reshape it back to (N, 3) for the actuators property
    deformable_mirror.actuators = np.array(actuators[iteration])
    plt.title('DM surface in nm (OPD)')
    # Show OPD (surface * 2)
    hp.imshow_field(deformable_mirror.surface * 2 * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-100, vmax=100)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('Average contrast')
    plt.plot(range(iteration + 1), average_contrast[:iteration + 1], 'o-')
    plt.xlim(0, num_iterations)
    plt.yscale('log')
    plt.ylim(1e-11, 1e-6)
    plt.grid(color='0.5')

    plt.suptitle('Iteration %d / %d' % (iteration + 1, num_iterations), fontsize='x-large')
    anim.grab_frame()

# --- Run the animation loop ---
filename = 'segmented_efc_loop_animation.mp4'
print("Setting up animation writer...")
anim.setup(fig, filename, dpi=150)
print("Starting animation rendering...")
for i in iteration_range:
    print(f"Rendering frame {i+1} / {num_iterations}")
    make_animation_1dm(i)
anim.finish()
plt.close(fig) 
print(f"Animation '{filename}' saved successfully.")