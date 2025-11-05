import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import os

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

# Create aperture and dark zone
aperture = hp.Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

dark_zone = hp.circular_aperture(2 * owa)(focal_grid)
dark_zone -= hp.circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# Create optical elements
coronagraph = hp.PerfectCoronagraph(aperture, order=6)

tip_tilt = hp.make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, remove_modes=tip_tilt, exponent=-3)

influence_functions = hp.make_xinetics_influence_functions(pupil_grid, num_actuators_across, actuator_spacing)
deformable_mirror = hp.DeformableMirror(influence_functions)

def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        deformable_mirror.actuators = actuators

    wf = hp.Wavefront(aperture, wavelength)
    if include_aberration:
        wf = aberration(wf)

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

actuators, electric_fields, images = run_efc(get_image, dark_zone, len(influence_functions), jacobian)



# Make animation

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
fig = plt.figure(figsize=(12, 10))
anim = FFMpegWriter(fps=5,metadata=dict(artist='HCIPy EFC Loop'))

num_iterations = len(actuators)
average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]

electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.5), True)

iteration_range = np.linspace(0, num_iterations-1, num_iterations, dtype=int)

def make_animation_1dm(iteration):
    plt.clf()

    plt.subplot(2, 2, 1)
    plt.title('Electric field')
    electric_field = electric_fields[iteration] / np.sqrt(img_ref.max())
    hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    plt.subplot(2, 2, 2)
    plt.title('Intensity image')
    hp.imshow_field(np.log10(images[iteration] / img_ref.max()), grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5)
    plt.colorbar()
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

    plt.subplot(2, 2, 3)
    deformable_mirror.actuators = actuators[iteration]
    plt.title('DM surface in nm')
    hp.imshow_field(deformable_mirror.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5)
    plt.colorbar()

    plt.subplot(2, 2, 4)
    plt.title('Average contrast')
    plt.plot(range(iteration), average_contrast[:iteration], 'o-')
    plt.xlim(0, num_iterations)
    plt.yscale('log')
    plt.ylim(1e-11, 1e-5)
    plt.grid(color='0.5')

    plt.suptitle('Iteration %d / %d' % (iteration + 1, num_iterations), fontsize='x-large')

    anim.grab_frame()

    # This finalizes the video file.
    # anim.finish()
    # plt.close(fig) # Close the figure object
    # print("Animation 'video.mp4' saved successfully.")

    # return anim

filename = 'efc_loop_animation.mp4' # Output filename
print("Setting up animation writer...")
anim.setup(fig, filename, dpi=150) # dpi for better quality

print("Starting animation rendering...")
# 2. A simple 'for' loop that calls your drawing function
for i in iteration_range:
    if (i+1) % 10 == 0 or i == 0:
        print(f"Rendering frame {i+1} / {num_iterations}")
    make_animation_1dm(i) # Call your function to draw frame 'i'

# Create the animation over planet_offset_x variable
# ani = FuncAnimation(
#     fig,
#     make_animation_1dm, 
#     frames=iteration_range, # Use the list of separations as frames
#     blit=False, 
#     interval=1000 # milliseconds between frames
# )

# print("Setting up animation writer...")

"""
def make_animation_1dm(actuators, electric_fields, images, dark_zone):
    fig = plt.figure(figsize=(12, 10))
    anim = FFMpegWriter('video.mp4')

    num_iterations = len(actuators)
    average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]

    electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.5), True)

    print("Setting up animation writer...")
    anim.setup(fig, 'video.mp4', dpi=150) # dpi for better quality
    plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'

    for i in range(num_iterations):
        plt.clf()

        plt.subplot(2, 2, 1)
        plt.title('Electric field')
        electric_field = electric_fields[i] / np.sqrt(img_ref.max())
        hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution)
        hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

        plt.subplot(2, 2, 2)
        plt.title('Intensity image')
        hp.imshow_field(np.log10(images[i] / img_ref.max()), grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5)
        plt.colorbar()
        hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

        plt.subplot(2, 2, 3)
        deformable_mirror.actuators = actuators[i]
        plt.title('DM surface in nm')
        hp.imshow_field(deformable_mirror.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5)
        plt.colorbar()

        plt.subplot(2, 2, 4)
        plt.title('Average contrast')
        plt.plot(range(i), average_contrast[:i], 'o-')
        plt.xlim(0, num_iterations)
        plt.yscale('log')
        plt.ylim(1e-11, 1e-5)
        plt.grid(color='0.5')

        plt.suptitle('Iteration %d / %d' % (i + 1, num_iterations), fontsize='x-large')

        anim.grab_frame()

    # This finalizes the video file.
    anim.finish()
    plt.close(fig) # Close the figure object
    print("Animation 'video.mp4' saved successfully.")

    return anim

make_animation_1dm(actuators, electric_fields, images, dark_zone)
"""