import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from gaussian_occulter import gaussian_occulter_generator
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Input parameters
pupil_diameter = 7e-3 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m

num_actuators_across = 32
actuator_spacing = 1.05 / 32 * pupil_diameter
aberration_ptv = 0.02 * wavelength # m

epsilon = 1e-9

spatial_resolution = focal_length * wavelength / pupil_diameter
iwa = 2 * spatial_resolution
owa = 12 * spatial_resolution
offset = 1 * spatial_resolution

efc_loop_gain = 0.5

# Create grids
pupil_grid = hp.make_pupil_grid(128, pupil_diameter * 1.2)
focal_grid = hp.make_focal_grid(2, 16, spatial_resolution=spatial_resolution)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

# TARGETED PISTON ABERRATION FOR THIS TEST
single_piston_ptv = 0.05 * wavelength # PTV of initial static aberration (on segment 0)

# SM parameters

gap_size = 90e-6 # m
num_rings = 3
segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)

# Create aperture and dark zone
aper, segments = hp.make_hexagonal_segmented_aperture(num_rings,
                                                         segment_flat_to_flat,
                                                         gap_size,
                                                         starting_ring=1,
                                                         return_segments=True)

aper = hp.evaluate_supersampled(aper, pupil_grid, 1)
segments = hp.evaluate_supersampled(segments, pupil_grid, 1)

dark_zone = hp.make_circular_aperture(2 * owa)(focal_grid)
dark_zone -= hp.make_circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# Create optical elements
sigma_lambda_d = 5
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)

ratio = 0.8
lyot_stop_generator = hp.make_circular_aperture(ratio*pupil_diameter) # percentage of the telescope diameter
lyot_stop_mask = lyot_stop_generator(pupil_grid)
Lyot_Coronagraph = hp.LyotCoronagraph(focal_grid,occulter_mask,lyot_stop_mask)

deformable_mirror = hp.DeformableMirror(segments)
num_modes = len(deformable_mirror.influence_functions)

target_segment_index = 5 # Center segment
initial_aberration_actuators = np.zeros(num_modes)
# Actuator value is surface height in meters
initial_aberration_actuators[target_segment_index] = single_piston_ptv / 2

def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        deformable_mirror.actuators = actuators

    wf = hp.Wavefront(aper, wavelength)
    img = prop(Lyot_Coronagraph(deformable_mirror(wf)))
    return img

img_ref = prop(hp.Wavefront(aper, wavelength)).intensity

def get_jacobian_matrix(get_image, dark_zone, num_modes):
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2)

    for i, mode in enumerate(np.eye(num_modes)):
        response = 0

        for amp in amps:
            response += amp * get_image(mode * amp, include_aberration=False).electric_field

        response /= np.var(amps)
        response = response[dark_zone]

        responses.append(np.concatenate((response.real, response.imag)))

    jacobian = np.array(responses).T
    return jacobian

jacobian = get_jacobian_matrix(get_image, dark_zone, num_modes)

rcond_tikhonov = 1e-2
def run_efc(get_image, dark_zone, num_modes, jacobian, initial_aberration_actuators, rcond):
    # Calculate EFC matrix
    efc_matrix = hp.inverse_tikhonov(jacobian, rcond)

    # Run EFC loop
    current_actuators = np.zeros(num_modes)

    actuators = []
    electric_fields = []
    images = []

    for i in range(50):
        img = get_image(current_actuators)

        electric_field = img.electric_field
        image = img.intensity

        actuators.append(current_actuators.copy())
        electric_fields.append(electric_field)
        images.append(image)

        x = np.concatenate((electric_field[dark_zone].real, electric_field[dark_zone].imag))
        y = efc_matrix.dot(x)

        current_actuators -= efc_loop_gain * y

    return actuators, electric_fields, images

actuators, electric_fields, images = run_efc(get_image, dark_zone, num_modes, jacobian, initial_aberration_actuators, rcond_tikhonov)

# --- ANIMATION SETUP ---
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe' # <<-- Check this path!

fig = plt.figure(figsize=(12, 10))
filename = 'sm_efc_gaussian_animation_final_manual.mp4'
fps_rate = 5
dpi_quality = 150
anim = FFMpegWriter(fps=fps_rate, metadata=dict(artist='HCIPy SM EFC Simulation'))
num_iterations = len(actuators)
average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]
electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.5), True)
iteration_range = np.linspace(0, num_iterations - 1, num_iterations, dtype=int)

def make_animation_sm_efc(iteration):
    """Drawing function for a single frame of the SM EFC loop."""
    plt.clf() 

    # 1. Electric field subplot
    plt.subplot(2, 2, 1)
    plt.title('Electric field')
    electric_field = electric_fields[iteration] / np.sqrt(img_ref) 
    hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    # 2. Intensity image subplot
    plt.subplot(2, 2, 2)
    plt.title('Intensity image (Log Contrast)')
    hp.imshow_field(np.log10(images[iteration] / img_ref), grid_units=spatial_resolution, cmap='inferno', vmin=-6, vmax=-4)
    plt.colorbar()
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    # 3. SM Piston Map subplot
    plt.subplot(2, 2, 3)
    deformable_mirror.actuators = actuators[iteration] 
    plt.title('SM Piston Map in nm')
    hp.imshow_field(deformable_mirror.surface * 1e9, grid_units=pupil_diameter, mask=aper, cmap='RdBu', vmin=-10, vmax=10)
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