import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- REQUIRED MOCK FUNCTION (Gaussian Occulter Profile) ---
def gaussian_occulter_generator(grid, sigma):
    """Returns a Gaussian occulter field for the Lyot Coronagraph."""
    r = grid.as_('polar').r
    # Occulter profile: T(r) = 1 - exp(-r**2 / (2 * sigma**2))
    mask = 1 - np.exp(-r**2 / (2 * sigma**2 + 1e-10))
    return mask

# --- INPUT PARAMETERS ---
pupil_diameter = 7e-3 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m

# SM SYSTEM PARAMETERS
num_segments_across = 6
# Segment diameter based on the overall pupil diameter
segment_diam = pupil_diameter / num_segments_across 
aberration_ptv = 0.2 * wavelength # PTV of initial static aberration (not on segments)

# CORONAGRAPHY PARAMETERS (Gaussian)
gaussian_sigma_lambda_d = 2.0  # Gaussian sigma in lambda/D units
lyot_stop_ratio = 0.8          # Inner diameter of the Lyot stop (Fraction of pupil diameter)

# EFC PARAMETERS
epsilon = 1e-9                  # Epsilon for Jacobian calculation
efc_loop_gain = 0.5
efc_iterations = 50
rcond_tikhonov = 1e-4           # Regularization for inverse matrix

# DARK ZONE PARAMETERS (in units of spatial resolution lambda*f/D)
spatial_resolution = focal_length * wavelength / pupil_diameter
iwa = 3.5 * spatial_resolution  # Inner Working Angle
owa = 15.0 * spatial_resolution # Outer Working Angle
offset = 1.0 * spatial_resolution # Dark zone is off-axis

# --- GRIDS AND PROPAGATORS ---
pupil_grid = hp.make_pupil_grid(256, pupil_diameter * 1.2)
focal_grid = hp.make_focal_grid(q=4, num_airy=32, spatial_resolution=spatial_resolution)

# --- APERTURE AND MASKS ---
# Telescope pupil (defines the overall beam boundary)
aperture = hp.circular_aperture(pupil_diameter)(pupil_grid)

# Lyot Stop: Circular stop in the Lyot plane
lyot_stop_mask_gen = hp.make_circular_aperture(lyot_stop_ratio * pupil_diameter)
lyot_stop_mask = lyot_stop_mask_gen(pupil_grid)

# Gaussian Occulter
occulter_mask_field = gaussian_occulter_generator(focal_grid, gaussian_sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask_field, focal_grid)

# Dark Zone Definition
dark_zone = hp.circular_aperture(2 * owa)(focal_grid)
dark_zone -= hp.circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset # Only half the field is the dark zone
dark_zone = dark_zone.astype(bool)

# --- SEGMENTED DEFORMABLE MIRROR SETUP (MANUAL MODE BASIS GENERATION) ---

# 1. Create a Hexagonal Grid defining the segment centers
segment_grid = hp.make_hexagonal_grid(segment_diam, num_segments_across)

# 2. Define the shape of a single segment (Hexagon, function generator)
segment_shape_generator = hp.regular_polygon_aperture(
    num_sides=6, 
    circum_diameter=segment_diam
)

# 3. Manually create the influence function (Field) for each segment (piston mode)
sm_influence_functions_list = []
for i in range(segment_grid.size):
    center = segment_grid[i]
    
    # Shift the segment shape to the center position
    segment_position_mask = segment_shape_generator(
        pupil_grid.shifted(-center)
    )
    
    # We must also mask out segments that fall outside the main circular aperture
    segment_influence = segment_position_mask * aperture
    
    # Only include the influence function if the segment is actually inside the pupil
    if np.any(segment_influence):
        sm_influence_functions_list.append(segment_influence)

# 4. Create the ModeBasis from the list of segment influence functions
# This is the 'segments' input expected by SegmentedDeformableMirror.
segments_mode_basis = hp.ModeBasis(sm_influence_functions_list)

# 5. Initialize the CORRECT class: SegmentedDeformableMirror
sm = hp.SegmentedDeformableMirror(segments_mode_basis) 
num_modes = len(sm.influence_functions) # The number of modes is the number of active segments

print(f"Segmented Mirror initialized with {num_modes} active segments.")

# --- OPTICAL COMPONENTS ---
# Static Aberration (applied once)
tip_tilt = hp.make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, remove_modes=tip_tilt, exponent=-3)

# Coronagraphic Propagation (Gaussian Occulter Lyot Coronagraph)
coronagraph = hp.LyotCoronagraph(focal_grid, occulter_mask, lyot_stop_mask)

# --- CRITICAL FIX: REFERENCE INTENSITY CALCULATION ---
# 1. Propagate a flat wavefront *without* the coronagraph
ref_wf = hp.Wavefront(aperture, wavelength)
ref_image = hp.FraunhoferPropagator(pupil_grid, focal_grid)(ref_wf).intensity

# 2. img_ref must be a single float: the peak intensity of the reference PSF
img_ref_scalar = ref_image.max()
print(f"Reference Peak Intensity (img_ref): {img_ref_scalar:.2e}")

# --- EFC SIMULATION CORE FUNCTIONS ---

def get_image(actuators=None, include_aberration=True):
    """Propagates the wavefront through the segmented mirror and coronagraph."""
    if actuators is not None:
        sm.actuators = actuators

    wf = hp.Wavefront(aperture, wavelength)
    
    # 1. Apply Segmented Mirror phase map
    wf = sm(wf)
    
    # 2. Apply static aberration
    if include_aberration:
        wf = aberration(wf)

    # 3. Propagate through the Lyot Coronagraph system
    img = coronagraph(wf)

    return img

img_ref = hp.Wavefront(aperture, wavelength).power # Total power for normalization

def get_jacobian_matrix(get_image, dark_zone, num_modes):
    """Calculates the Jacobian matrix (G) using mode probing."""
    print(f"Calculating Jacobian using {num_modes} modes...")
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2)
    
    # The influence functions of the Segmented Mirror are the modes we probe
    sm_modes = sm.influence_functions

    for i, mode in enumerate(sm_modes):
        # Calculate the complex electric field response
        response = 0
        for amp in amps:
            actuator_probe = np.zeros(num_modes)
            actuator_probe[i] = amp
            
            # Note: We probe without static aberration to get the system response
            response += amp * get_image(actuator_probe, include_aberration=False).electric_field 
            
        response /= np.var(amps)
        
        # Extract response in the dark zone and concatenate real/imag parts
        response_dz = response[dark_zone]
        responses.append(np.concatenate((response_dz.real, response_dz.imag)))

    # Reset SM surface and actuators to zero for EFC start
    sm.actuators = np.zeros(num_modes)

    jacobian = np.array(responses).T
    print("Jacobian calculation complete.")
    return jacobian

jacobian = get_jacobian_matrix(get_image, dark_zone, num_modes)

def run_efc(get_image, dark_zone, num_modes, jacobian, rcond=rcond_tikhonov):
    """Runs the Electric Field Conjugation (EFC) loop."""
    efc_matrix = hp.inverse_tikhonov(jacobian, rcond)

    current_actuators = np.zeros(num_modes)
    actuators = []
    electric_fields = []
    images = []

    print(f"Starting EFC loop for {efc_iterations} iterations...")
    for i in range(efc_iterations):
        img = get_image(current_actuators)

        electric_field = img.electric_field
        image = img.intensity

        actuators.append(current_actuators.copy())
        electric_fields.append(electric_field)
        images.append(image)

        # Extract electric field in the dark zone (x)
        x = np.concatenate((electric_field[dark_zone].real, electric_field[dark_zone].imag))
        
        # Calculate the required actuator change (y = G_dagger * x)
        y = efc_matrix.dot(x)

        # Update actuators
        current_actuators -= efc_loop_gain * y
        
        if (i+1) % 10 == 0:
            # Contrast is the average intensity in the dark zone relative to the input intensity
            print(f"  Iteration {i+1}: Average Contrast = {np.mean(image[dark_zone] / img_ref_scalar):.2e}")

    print("EFC loop finished.")
    return actuators, electric_fields, images

# --- RUN EFC AND PRE-CALCULATE ALL FRAMES ---
actuators, electric_fields, images = run_efc(get_image, dark_zone, num_modes, jacobian)

# --- ANIMATION SETUP ---
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe' # <<-- Check this path!

fig = plt.figure(figsize=(12, 10))
filename = 'sm_efc_gaussian_animation_final_manual.mp4'
fps_rate = 5
dpi_quality = 150
anim = FFMpegWriter(fps=fps_rate, metadata=dict(artist='HCIPy SM EFC Simulation'))
num_iterations = len(actuators)
average_contrast = [np.mean(image[dark_zone] / img_ref_scalar) for image in images]
electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.5), True)
iteration_range = np.linspace(0, num_iterations - 1, num_iterations, dtype=int)

def make_animation_sm_efc(iteration):
    """Drawing function for a single frame of the SM EFC loop."""
    plt.clf() 

    # 1. Electric field subplot
    plt.subplot(2, 2, 1)
    plt.title('Electric field')
    electric_field = electric_fields[iteration] / np.sqrt(img_ref_scalar) 
    hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    # 2. Intensity image subplot
    plt.subplot(2, 2, 2)
    plt.title('Intensity image (Log Contrast)')
    hp.imshow_field(np.log10(images[iteration] / img_ref_scalar), grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5)
    plt.colorbar()
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    # 3. SM Piston Map subplot
    plt.subplot(2, 2, 3)
    sm.actuators = actuators[iteration] 
    plt.title('SM Piston Map in nm')
    hp.imshow_field(sm.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-10, vmax=10)
    plt.colorbar()

    # 4. Average contrast plot
    plt.subplot(2, 2, 4)
    plt.title('Average Contrast in Dark Zone')
    plt.plot(range(iteration + 1), average_contrast[:iteration + 1], 'o-')
    plt.xlim(0, num_iterations)
    plt.yscale('log')
    plt.ylim(1e-11, 1e-5)
    plt.grid(color='0.5', linestyle='--')
    plt.xlabel("EFC Iteration")
    plt.ylabel("Average Contrast")

    plt.suptitle('SM EFC Iteration %d / %d' % (iteration + 1, num_iterations), fontsize='x-large')


# --- VIDEO SAVING LOOP ---
print("\n--- Video Saving ---")
try:
    anim.setup(fig, filename, dpi=dpi_quality) 

    print(f"Starting animation rendering ({num_iterations} frames)...")
    for i in iteration_range:
        make_animation_sm_efc(i) 
        anim.grab_frame()     
        
except Exception as e:
    print("\n--- CRITICAL ERROR DURING SAVING ---")
    print(f"Error: {e}")
    print(f"If this is a FileNotFoundError, please ensure 'C:\\ffmpeg\\bin\\ffmpeg.exe' is the correct path.")
    
finally:
    anim.finish()
    print(f"\nAnimation process finished. Output file saved to '{filename}'.")
    plt.show()