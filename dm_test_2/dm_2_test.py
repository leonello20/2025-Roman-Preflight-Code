"""
import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from fig2img import fig2img
from gaussian_occulter import gaussian_occulter_generator
import os
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Input parameters
pupil_diameter = 1 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m

num_actuators_across = 32
actuator_spacing = 1.05 / num_actuators_across * pupil_diameter
aberration_ptv = 0.02 * wavelength # m

epsilon = 1e-9
DM_STROKE_LIMIT = 5e-7 # meters

spatial_resolution = focal_length * wavelength / pupil_diameter
iwa = 6 * spatial_resolution
owa = 12 * spatial_resolution
offset = 1 * spatial_resolution

efc_loop_gain = 0.05

# --- NEW: Dual Rcond Values ---
# DM1 is typically used for initial field correction, DM2 for final dark hole cleanup.
rcond_dm1 = 0.5 # Tikhonov parameter for DM1
rcond_dm2 = 0.5 # Tikhonov parameter for DM2

# Create grids
grid_size = 256
q = 8
num_airy = 16
pupil_grid = hp.make_pupil_grid(grid_size, pupil_diameter * 1.2)
focal_grid = hp.make_focal_grid(q, num_airy, spatial_resolution=spatial_resolution)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

# --- Define the inverse propagator for returning to the second pupil plane ---
prop_inv = prop.backward

# Create aperture and dark zone
aperture = hp.Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

# Create dark zone
dark_zone = hp.make_circular_aperture(2 * owa)(focal_grid)
dark_zone -= hp.make_circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# Create optical elements
sigma_lambda_d = 5
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask,focal_grid)

# Create the Lyot Stop
ratio = 0.7 # Lyot Stop diameter ratio
lyot_stop_generator = hp.make_circular_aperture(ratio*pupil_diameter)
lyot_stop_mask = lyot_stop_generator(pupil_grid)

tip_tilt = hp.make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, exponent=-3)

# DM setup
influence_functions = hp.make_xinetics_influence_functions(pupil_grid, num_actuators_across, actuator_spacing)
deformable_mirror_1 = hp.DeformableMirror(influence_functions)
deformable_mirror_2 = hp.DeformableMirror(influence_functions)

# --- CORRECTED get_image FUNCTION (Manual Propagation Path) ---
def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        deformable_mirror_1.actuators = actuators[:len(influence_functions)]
        deformable_mirror_2.actuators = actuators[len(influence_functions):]
    
    wf = hp.Wavefront(aperture, wavelength)
    if include_aberration:
        wf = aberration(wf)

    # --- NEW OPTICAL PATH IMPLEMENTATION ---
    
    # 1. DM1 correction
    wf_dm_1 = deformable_mirror_1(wf)

    # 2. Propagate to Lyot Plane before DM and Lyot Stop
    lyot_coronagraph = hp.LyotCoronagraph(pupil_grid, occulter_mask, lyot_stop=None)
    wf_lyot_no_mask = lyot_coronagraph(wf_dm_1)

    # 3. Apply DM2 in Lyot Plane
    wf_dm_2 = deformable_mirror_2(wf_lyot_no_mask)

    # 4. Apply Lyot Stop Mask (PP2)
    wf_lyot_electric_field = wf_dm_2.electric_field * lyot_stop_mask

    # 5. Propagate to Final Focal Plane
    img = prop(hp.Wavefront(wf_lyot_electric_field, wavelength))

    return img
# --- END get_image FUNCTION ---


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

    # J has shape (2*N_dark_zone, 2*N_actuators)
    jacobian = np.array(responses).T
    return jacobian

jacobian = get_jacobian_matrix(get_image, dark_zone, 2 * len(influence_functions))

# --- NEW FUNCTION FOR PARTITIONED REGULARIZATION ---
def inverse_tikhonov_partitioned(jacobian, num_modes_dm1, rcond_dm1, rcond_dm2):
    num_modes_total = jacobian.shape[1]
    
    # Calculate J^H * J
    J_H_J = jacobian.T.conj() @ jacobian
    
    # Create the block-diagonal regularization matrix Lambda^2
    lambda_matrix = np.zeros((num_modes_total, num_modes_total))
    
    # DM1 block (top-left)
    np.fill_diagonal(lambda_matrix[:num_modes_dm1, :num_modes_dm1], rcond_dm1**2)
    
    # DM2 block (bottom-right)
    np.fill_diagonal(lambda_matrix[num_modes_dm1:, num_modes_dm1:], rcond_dm2**2)

    # Calculate (J^H * J + Lambda^2)^-1
    try:
        M_inv = np.linalg.inv(J_H_J + lambda_matrix)
    except np.linalg.LinAlgError:
        print("Error: Matrix inversion failed. Try increasing rcond_dm1 or rcond_dm2.")
        return np.zeros((num_modes_total, jacobian.shape[0]))

    # Calculate M_EFC = (J^H * J + Lambda^2)^-1 * J^H
    EFC_matrix = M_inv @ jacobian.T.conj()
    return EFC_matrix


def run_efc(get_image, dark_zone, num_modes, jacobian, rcond_dm1, rcond_dm2):
    num_modes_dm1 = len(influence_functions)
    
    # --- CHANGE HERE: Use the custom partitioned inverse function ---
    efc_matrix = inverse_tikhonov_partitioned(jacobian, num_modes_dm1, rcond_dm1, rcond_dm2)

    # Run EFC loop
    current_actuators = np.zeros(num_modes)

    actuators = []
    electric_fields = []
    images = []

    NUM_ITERATIONS = 100
    
    for i in range(NUM_ITERATIONS):
        img = get_image(current_actuators)

        electric_field = img.electric_field
        image = img.intensity

        actuators.append(current_actuators.copy())
        electric_fields.append(electric_field)
        images.append(image)

        # Map complex focal plane error (x) to real actuator change (y)
        x = np.concatenate((electric_field[dark_zone].real, electric_field[dark_zone].imag))
        y = efc_matrix.dot(x)

        current_actuators -= efc_loop_gain * y
        
        # ENFORCE LIMIT: Uses the DM_STROKE_LIMIT variable
        current_actuators = np.clip(current_actuators, -DM_STROKE_LIMIT, DM_STROKE_LIMIT)

    return actuators, electric_fields, images, NUM_ITERATIONS

# Get the results and the actual number of iterations used, passing both rconds
actuators, electric_fields, images, num_iterations = run_efc(get_image, dark_zone, 2 * len(influence_functions), jacobian, rcond_dm1, rcond_dm2)


# --- Animation Rendering Block ---

# Using a raw string for the ffmpeg path
plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
# Use a figure size and dpi that is friendly to memory 
fig = plt.figure(figsize=(9, 7)) 

anim = FFMpegWriter(fps=5, metadata=dict(artist='HCIPy EFC Loop'))

average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]

electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.0), True)

iteration_range = np.linspace(0, num_iterations-1, num_iterations, dtype=int)

def make_animation_1dm(iteration):
    deformable_mirror_1.actuators = actuators[iteration][:len(influence_functions)]
    deformable_mirror_2.actuators = actuators[iteration][len(influence_functions):]
    # Clear the entire figure before drawing new subplots
    fig.clf()
    
    # Re-establish subplot layout
    
    # 2. Intensity Image
    ax2 = fig.add_subplot(2, 2, 1)
    ax2.set_title('Focal Plane Intensity Image')
    ax2.set_xlabel('x/D')
    ax2.set_ylabel('y/D')
    log_intensity = np.log10(images[iteration] / img_ref.max())
    img = hp.imshow_field(log_intensity, grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5, ax=ax2)
    plt.colorbar(img, ax=ax2)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white', ax=ax2)

    # 5. Average Contrast
    ax5 = fig.add_subplot(2, 2, 2)
    ax5.set_title('Average contrast')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Average Contrast ($log_{10}(I/I_{total})$)')
    ax5.plot(range(iteration + 1), average_contrast[:iteration + 1], 'o-')
    ax5.set_xlim(0, num_iterations)
    ax5.set_yscale('log')
    ax5.set_ylim(1e-11, 1e-5)
    ax5.grid(color='0.5')

    # 3. DM1 Surface (Using .format())
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title(r'DM1 surface ($\lambda_{{DM1}}={rcond_dm1:.3f}$)'.format(rcond_dm1=rcond_dm1))
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    dm_img = hp.imshow_field(deformable_mirror_1.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5, ax=ax3)
    plt.colorbar(dm_img, ax=ax3, label='DM1 Surface (nm)')

    # 4. DM2 Surface (Using .format())
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title(r'DM2 surface ($\lambda_{{DM2}}={rcond_dm2:.3f}$)'.format(rcond_dm2=rcond_dm2))
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    dm_img = hp.imshow_field(deformable_mirror_2.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5, ax=ax4)
    plt.colorbar(dm_img, ax=ax4, label='DM2 Surface (nm)')

    # Supertitle (Using standard formatting for the animation loop)
    fig.suptitle('Iteration %d / %d (Custom Path, Dual Rcond)' % (iteration + 1, num_iterations), fontsize='x-large')
    # Adjust layout to prevent overlap
    fig.tight_layout()

filename = 'efc_loop_animation_dm_1_dm_2_dual_rcond.mp4' # Output filename
print("Setting up animation writer...")
anim.setup(fig, filename, dpi=50)

print("Starting animation rendering...")

# 2. Manual Loop for grabbing frames
for i in iteration_range:
    if (i+1) % 5 == 0 or i == 0:
        print(f"Rendering frame {i+1} / {num_iterations}")
    make_animation_1dm(i) 
    anim.grab_frame() # Grab frame inside the loop after drawing

# Finalize the video file
anim.finish()
print(f"Animation '{filename}' saved successfully after {num_iterations} frames.")
plt.close(fig)

# Intensity image for the final iteration

final_iteration = num_iterations - 1

# Plot the last iteration

# Intensity Image
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title('Intensity image for last iteration (Custom Path)')
plt.xlabel('x/D')
plt.ylabel('y/D')
log_intensity = np.log10(images[final_iteration] / img_ref.max())
hp.imshow_field(log_intensity, grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5)
plt.colorbar(label='Contrast ($log_{10}(I/I_{total})$)')
hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

# Average Contrast
plt.subplot(1, 2, 2)
plt.title('Average Dark Zone Contrast')
plt.xlabel('Iteration')
plt.ylabel('Average Contrast ($log_{10}(I/I_{total})$)')
plt.plot(range(num_iterations), average_contrast, 'o-')
plt.xlim(0, num_iterations)
plt.yscale('log')
plt.ylim(1e-11, 1e-5)
plt.grid(color='0.5')
# --- Using .format() for suptitle ---
plt.suptitle(r'Final Results after {num_iterations} Iterations (Custom Path, $\lambda_{{DM1}}={rcond_dm1:.3f}$, $\lambda_{{DM2}}={rcond_dm2:.3f}$)'
             .format(num_iterations=num_iterations, rcond_dm1=rcond_dm1, rcond_dm2=rcond_dm2), 
             fontsize='x-large')
# --- END UPDATED LINE ---

fig_intensity = plt.gcf()
img_intensity = fig2img(fig_intensity)
# Raw string for file path
img_intensity.save(r'C:\Users\leone\OneDrive\Documents\GitHub\2025-Roman-Preflight-Code\Images\dm_1_dm_2_final_intensity_dual_rcond_fixed.png')
plt.show()

# DM 1 Surface
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
# Raw string for title (Updated to use .format())
plt.title(r'DM1 surface ($\lambda_{{DM1}}={rcond_dm1:.3f}$)'.format(rcond_dm1=rcond_dm1))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
deformable_mirror_1.actuators = actuators[final_iteration][:len(influence_functions)]
hp.imshow_field(deformable_mirror_1.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5)
plt.colorbar(label='DM1 Surface (nm)')

# DM 2 Surface
plt.subplot(1, 2, 2)
# Raw string for title (Updated to use .format())
plt.title(r'DM2 surface ($\lambda_{{DM2}}={rcond_dm2:.3f}$)'.format(rcond_dm2=rcond_dm2))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
deformable_mirror_2.actuators = actuators[final_iteration][len(influence_functions):]
hp.imshow_field(deformable_mirror_2.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5)
plt.colorbar(label='DM2 Surface (nm)')

# Save the DM surfaces
fig_dm = plt.gcf()
img_dm = fig2img(fig_dm)
# Raw string for file path
img_dm.save(r'C:\Users\leone\OneDrive\Documents\GitHub\2025-Roman-Preflight-Code\Images\final_dm_surfaces_dm_1_dm_2_dual_rcond_fixed.png')
plt.show()

# Print the initial and final contrast values
print("Contrast for first iteration:", average_contrast[0])
print("Contrast for last iteration:", average_contrast[-1])
"""

import hcipy as hp
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from fig2img import fig2img
from gaussian_occulter import gaussian_occulter_generator
import os
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Input parameters
pupil_diameter = 1 # m
wavelength = 700e-9 # m
focal_length = 500e-3 # m

num_actuators_across = 32
actuator_spacing = 1.05 / num_actuators_across * pupil_diameter
aberration_ptv = 0.02 * wavelength # m

epsilon = 1e-9
DM_STROKE_LIMIT = 5e-7 # meters

spatial_resolution = focal_length * wavelength / pupil_diameter
iwa = 6 * spatial_resolution
owa = 12 * spatial_resolution
offset = 1 * spatial_resolution

efc_loop_gain = 0.05

# Create grids
grid_size = 256
q = 8
num_airy = 16
pupil_grid = hp.make_pupil_grid(grid_size, pupil_diameter * 1.2)
focal_grid = hp.make_focal_grid(q, num_airy, spatial_resolution=spatial_resolution)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

# Create aperture and dark zone
aperture = hp.Field(np.exp(-(pupil_grid.as_('polar').r / (0.5 * pupil_diameter))**30), pupil_grid)

# Create dark zone
dark_zone = hp.make_circular_aperture(2 * owa)(focal_grid)
dark_zone -= hp.make_circular_aperture(2 * iwa)(focal_grid)
dark_zone *= focal_grid.x > offset
dark_zone = dark_zone.astype(bool)

# Create optical elements
sigma_lambda_d = 5
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask,focal_grid)

# create the occulter mask and Lyot Stop in the Lyot Coronagraph
ratio = 0.7 # Lyot Stop diameter ratio
lyot_stop_generator = hp.make_circular_aperture(ratio*pupil_diameter) # percentage of the telescope diameter
lyot_stop_mask = lyot_stop_generator(pupil_grid)

tip_tilt = hp.make_zernike_basis(3, pupil_diameter, pupil_grid, starting_mode=2)
# aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, remove_modes=tip_tilt, exponent=-3)
aberration = hp.SurfaceAberration(pupil_grid, aberration_ptv, pupil_diameter, exponent=-3)

# This uses the 32x32 continuous DM model, matching your goal
influence_functions = hp.make_xinetics_influence_functions(pupil_grid, num_actuators_across, actuator_spacing)
deformable_mirror_1 = hp.DeformableMirror(influence_functions)
deformable_mirror_2 = hp.DeformableMirror(influence_functions)

def get_image(actuators=None, include_aberration=True):
    if actuators is not None:
        deformable_mirror_1.actuators = actuators[:len(influence_functions)]
        deformable_mirror_2.actuators = actuators[len(influence_functions):]
    
    wf = hp.Wavefront(aperture, wavelength)
    if include_aberration:
        wf = aberration(wf)

    # --- NEW OPTICAL PATH IMPLEMENTATION ---
    
    # 1. DM1 correction
    wf_dm_1 = deformable_mirror_1(wf)

    # 2. Propagate to Lyot Plane before DM and Lyot Stop
    lyot_coronagraph = hp.LyotCoronagraph(pupil_grid, occulter_mask, lyot_stop=None)
    wf_lyot_no_mask = lyot_coronagraph(wf_dm_1)

    # 3. Apply DM2 in Lyot Plane
    wf_dm_2 = deformable_mirror_2(wf_lyot_no_mask)

    # 4. Apply Lyot Stop Mask (PP2)
    wf_lyot_electric_field = wf_dm_2.electric_field * lyot_stop_mask

    # 5. Propagate to Final Focal Plane
    img = prop(hp.Wavefront(wf_lyot_electric_field, wavelength))

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

jacobian = get_jacobian_matrix(get_image, dark_zone, 2 * len(influence_functions))
rcond_dm1 = 0.025
rcond_dm2 = 0.005

def run_efc(get_image, dark_zone, num_modes, jacobian, rcond_dm1, rcond_dm2):
    # --- CHANGE HERE: Using inverse_tikhonov_double_dm ---
    num_modes_dm1 = len(influence_functions)
    efc_matrix = hp.inverse_tikhonov(jacobian, num_modes_dm1, rcond_dm1, rcond_dm2)

    # Run EFC loop
    current_actuators = np.zeros(num_modes)

    actuators = []
    electric_fields = []
    images = []

    # Keeping iterations low for stability
    NUM_ITERATIONS = 1000
    
    for i in range(NUM_ITERATIONS):
        img = get_image(current_actuators)

        electric_field = img.electric_field
        image = img.intensity

        actuators.append(current_actuators.copy())
        electric_fields.append(electric_field)
        images.append(image)

        x = np.concatenate((electric_field[dark_zone].real, electric_field[dark_zone].imag))
        y = efc_matrix.dot(x)

        current_actuators -= efc_loop_gain * y
        
        # ENFORCE LIMIT: Uses the DM_STROKE_LIMIT variable
        current_actuators = np.clip(current_actuators, -DM_STROKE_LIMIT, DM_STROKE_LIMIT)

    return actuators, electric_fields, images, NUM_ITERATIONS

# Get the results and the actual number of iterations used, passing both rconds
actuators, electric_fields, images, num_iterations = run_efc(get_image, dark_zone, 2 * len(influence_functions), jacobian, rcond_dm1, rcond_dm2)





# --- Animation Rendering Block ---

plt.rcParams['animation.ffmpeg_path'] = r'C:\ffmpeg\bin\ffmpeg.exe'
# Use a figure size and dpi that is friendly to memory 
fig = plt.figure(figsize=(9, 7)) 

anim = FFMpegWriter(fps=5, metadata=dict(artist='HCIPy EFC Loop'))

average_contrast = [np.mean(image[dark_zone] / img_ref.max()) for image in images]

electric_field_norm = mpl.colors.LogNorm(10**-5, 10**(-2.0), True)

iteration_range = np.linspace(0, num_iterations-1, num_iterations, dtype=int)

def make_animation_1dm(iteration):
    deformable_mirror_1.actuators = actuators[iteration][:len(influence_functions)]
    deformable_mirror_2.actuators = actuators[iteration][len(influence_functions):]
    # Clear the entire figure before drawing new subplots
    fig.clf()
    
    # Re-establish subplot layout
    
    # 1. Electric Field
    # ax1 = fig.add_subplot(2, 2, 1)
    # ax1.set_title('Focal Plane Electric Field')
    # ax1.set_xlabel('x/D')
    # ax1.set_ylabel('y/D')
    # electric_field = electric_fields[iteration] / np.sqrt(img_ref.max())
    # hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution, ax=ax1)
    # hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white', ax=ax1)

    # 2. Intensity Image
    ax2 = fig.add_subplot(2, 2, 1)
    ax2.set_title('Focal Plane Intensity Image')
    ax2.set_xlabel('x/D')
    ax2.set_ylabel('y/D')
    log_intensity = np.log10(images[iteration] / img_ref.max())
    img = hp.imshow_field(log_intensity, grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5, ax=ax2)
    plt.colorbar(img, ax=ax2)
    hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white', ax=ax2)

    # 5. Average Contrast
    ax5 = fig.add_subplot(2, 2, 2)
    ax5.set_title('Average contrast')
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Average Contrast ($log_{10}(I/I_{total})$)')
    ax5.plot(range(iteration + 1), average_contrast[:iteration + 1], 'o-')
    ax5.set_xlim(0, num_iterations)
    ax5.set_yscale('log')
    ax5.set_ylim(1e-11, 1e-5)
    ax5.grid(color='0.5')

    # 3. DM1 Surface
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title('DM1 surface ($\\lambda_{DM1}=%.3f$)' % rcond_dm1)
    ax3.set_xlabel('x (m)')
    ax3.set_ylabel('y (m)')
    dm_img = hp.imshow_field(deformable_mirror_1.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5, ax=ax3)
    plt.colorbar(dm_img, ax=ax3, label='DM1 Surface (nm)')

    # 4. DM2 Surface
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title('DM2 surface ($\\lambda_{DM2}=%.3f$)' % rcond_dm2)
    ax4.set_xlabel('x (m)')
    ax4.set_ylabel('y (m)')
    dm_img = hp.imshow_field(deformable_mirror_2.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu', vmin=-5, vmax=5, ax=ax4)
    plt.colorbar(dm_img, ax=ax4, label='DM2 Surface (nm)')

    # Supertitle
    fig.suptitle('Iteration %d / %d' % (iteration + 1, num_iterations), fontsize='x-large')
    # Adjust layout to prevent overlap
    fig.tight_layout()

filename = 'efc_loop_animation_dm_1_dm_2_dual_rcond.mp4' # Output filename
print("Setting up animation writer...")
anim.setup(fig, filename, dpi=50)

print("Starting animation rendering...")

# 2. Manual Loop for grabbing frames
for i in iteration_range:
    if (i+1) % 5 == 0 or i == 0:
        print(f"Rendering frame {i+1} / {num_iterations}")
    make_animation_1dm(i) 
    anim.grab_frame() # Grab frame inside the loop after drawing

# Finalize the video file
anim.finish()
print(f"Animation '{filename}' saved successfully after {num_iterations} frames.")
plt.close(fig)

# Intensity image for the final iteration

final_iteration = num_iterations - 1

# Plot the last iteration

# 1. Electric Field
# plt.figure(figsize=(10, 10))
# plt.subplot(2, 2, 1)
# plt.title('Electric field for last iteration')
# electric_field = electric_fields[final_iteration] / np.sqrt(img_ref.max())
# hp.imshow_field(electric_field, norm=electric_field_norm, grid_units=spatial_resolution)
# hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

# Intensity Image
plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.title('Intensity image for last iteration')
plt.xlabel('x/D')
plt.ylabel('y/D')
log_intensity = np.log10(images[final_iteration] / img_ref.max())
hp.imshow_field(log_intensity, grid_units=spatial_resolution, cmap='inferno', vmin=-10, vmax=-5)
plt.colorbar(label='Contrast ($log_{10}(I/I_{total})$)')
hp.contour_field(dark_zone, grid_units=spatial_resolution, levels=[0.5], colors='white')

# Average Contrast
plt.subplot(1, 2, 2)
plt.title('Average Dark Zone Contrast')
plt.xlabel('Iteration')
plt.ylabel('Average Contrast ($log_{10}(I/I_{total})$)')
plt.plot(range(num_iterations), average_contrast, 'o-')
plt.xlim(0, num_iterations)
plt.yscale('log')
plt.ylim(1e-11, 1e-5)
plt.grid(color='0.5')
plt.suptitle('Final Results after {num_iterations} Iterations (Custom Path, $\\lambda_{{DM1}}={rcond_dm1:.3f}$, $\\lambda_{{DM2}}={rcond_dm2:.3f}$)'.format(num_iterations=num_iterations, rcond_dm1=rcond_dm1, rcond_dm2=rcond_dm2), fontsize='x-large')

# Save the intensity image
fig_intensity = plt.gcf()
img_intensity = fig2img(fig_intensity)
img_intensity.save('C:/Users/leone/OneDrive/Documents/GitHub/2025-Roman-Preflight-Code/dm_test_2/Images_DM/dm_1_dm_2_dual_rcond_final_intensity.png')
plt.show()

# DM 1 Surface
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('DM1 surface (Custom Path, $\\lambda_{DM1}=%.3f$)' % rcond_dm1)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
deformable_mirror_1.actuators = actuators[final_iteration][:len(influence_functions)]
hp.imshow_field(deformable_mirror_1.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu')
plt.colorbar(label='DM1 Surface (nm)')

# DM 2 Surface
plt.subplot(1, 2, 2)
plt.title('DM2 surface (Custom Path, $\\lambda_{DM2}=%.3f$)' % rcond_dm2)
plt.xlabel('x (m)')
plt.ylabel('y (m)')
deformable_mirror_2.actuators = actuators[final_iteration][len(influence_functions):]
hp.imshow_field(deformable_mirror_2.surface * 1e9, grid_units=pupil_diameter, mask=aperture, cmap='RdBu')
plt.colorbar(label='DM2 Surface (nm)')

# Save the DM surfaces
fig_dm = plt.gcf()
img_dm = fig2img(fig_dm)
img_dm.save('C:/Users/leone/OneDrive/Documents/GitHub/2025-Roman-Preflight-Code/dm_test_2/Images_DM/final_dm_surfaces_dm_1_dm_2_dual_rcond.png')
plt.show()

# Print the initial and final contrast values
print("Contrast for first iteration:", average_contrast[0])
print("Contrast for last iteration:", average_contrast[-1])