import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import os
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- INPUT PARAMETERS ---
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

# --- GRIDS AND PROPAGATORS ---
pupil_grid = hp.make_pupil_grid(128, pupil_diameter * 1.2)
focal_grid = hp.make_focal_grid(2, 16, spatial_resolution=spatial_resolution)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

# --- APERTURE AND DARK ZONE ---
aperture = hp.Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

dark_zone = hp.circular_aperture(2 * owa)(focal_grid)
dark_zone -= hp.circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# --- OPTICAL ELEMENTS ---
coronagraph = hp.PerfectCoronagraph(aperture, order=6)

tip_tilt = hp.make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, remove_modes=tip_tilt, exponent=-3)

influence_functions = hp.make_xinetics_influence_functions(pupil_grid, num_actuators_across, actuator_spacing)
deformable_mirror = hp.DeformableMirror(influence_functions)

# --- EFC FUNCTIONS ---

def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        deformable_mirror.actuators = actuators

    wf = hp.Wavefront(aperture, wavelength)
    if include_aberration:
        wf = aberration(wf)

    # Propagate through DM and Coronagraph to final image plane
    img = prop(coronagraph(deformable_mirror(wf)))
    return img

img_ref = prop(hp.Wavefront(aperture, wavelength)).intensity

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

jacobian = get_jacobian_matrix(get_image, dark_zone, len(influence_functions))

def run_efc(get_image, dark_zone, num_modes, jacobian, rcond=1e-2):
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

# --- RUN THE EFC SIMULATION ---
actuators, electric_fields, images = run_efc(get_image, dark_zone, len(influence_functions), jacobian)

# --- ANIMATION SETUP ---
# CRITICAL FIX: The FFmpeg path must be set in rcParams for your matplotlib version
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

fig = plt.figure(figsize=(12, 10))
filename = 'efc_loop_animation.mp4' # Output filename
fps_rate = 5
dpi_quality = 150 # High DPI for good video quality

anim = FFMpegWriter(fps=fps_rate, metadata=dict(artist='HCIPy EFC Loop'))
num_iterations = len(actuators)
average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]
electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.5), True)
iteration_range = np.linspace(0, num_iterations - 1, num_iterations, dtype=int)


def make_animation_1dm(iteration):
    """Drawing function for a single frame of the EFC loop."""
    plt.clf() # Clear the figure for the new frame

    # 1. Electric field subplot
    plt.subplot(2, 2, 1)
    plt.title('Electric field')
    electric_field = electric_fields[iteration] / np.sqrt(img_ref.max())
    hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    # 2. Intensity image subplot
    plt.subplot(2, 2, 2)
    plt.title('Intensity image')
    hp.imshow_field(np.log10(images[iteration] / img_ref.max()), grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5)
    plt.colorbar()
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    # 3. DM surface subplot
    plt.subplot(2, 2, 3)
    deformable_mirror.actuators = actuators[iteration]
    plt.title('DM surface in nm')
    hp.imshow_field(deformable_mirror.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5)
    plt.colorbar()

    # 4. Average contrast plot
    plt.subplot(2, 2, 4)
    plt.title('Average contrast')
    plt.plot(range(iteration), average_contrast[:iteration], 'o-')
    plt.xlim(0, num_iterations)
    plt.yscale('log')
    plt.ylim(1e-11, 1e-5)
    plt.grid(color='0.5')

    plt.suptitle('Iteration %d / %d' % (iteration + 1, num_iterations), fontsize='x-large')

# --- VIDEO SAVING LOOP (HCI PY STYLE) ---
print("Setting up animation writer...")
try:
    # 1. Start the animation writer process
    anim.setup(fig, filename, dpi=dpi_quality) 

    print("Starting animation rendering and saving...")
    # 2. Loop through all the pre-calculated images and "grab" a frame
    for i in iteration_range:
        if (i+1) % 10 == 0 or i == 0 or i == num_iterations - 1:
            print(f"Rendering frame {i+1} / {num_iterations}")
        
        make_animation_1dm(i) # Draw the frame
        anim.grab_frame()     # Capture the drawn frame

except Exception as e:
    # This catches the FileNotFoundError (if path is wrong) and other save errors
    print("\n--- CRITICAL ERROR DURING SAVING ---")
    print(f"Error: {e}")
    print(f"If the error is FileNotFoundError, please ensure 'C:\\ffmpeg\\bin\\ffmpeg.exe' is the correct path to FFmpeg.")

finally:
    # 3. CRITICAL: This is required to finalize the video file.
    anim.finish()
    print(f"\nAnimation process finished. Output file should be at '{filename}'.")
    # Only show the static plot after saving is complete
    plt.show()