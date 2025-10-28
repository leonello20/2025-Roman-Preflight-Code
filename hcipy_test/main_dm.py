import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import warnings
from gaussian_occulter import gaussian_occulter_generator
from contrast_curve import contrast_curve
from setup_aberrated_system import setup_aberrated_system
from create_combined_aberration_wavefront import create_combined_aberration_wavefront
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

pupil_diameter = 0.019725 # meters
grid_size = 1024
hsm, aper, pupil_grid, wavelength, focal_length = setup_aberrated_system(grid_size, pupil_diameter)

# 2. Define planet parameters
PLANET_CONTRAST = 1e-10 # Star is 1, Planet is 1e-10 dimmer
PLANET_SEPARATION_X = 10 # lambda/D
PLANET_SEPARATION_Y = 0 # lambda/D

# 3. Create the combined wavefront
wavefront_to_propagate = create_combined_aberration_wavefront(
    hsm,
    aper,
    pupil_diameter,
    pupil_grid,
    wavelength,
    focal_length,
    planet_contrast=PLANET_CONTRAST,
    separation_lambda_d_x=PLANET_SEPARATION_X,
    separaton_lambda_d_y=PLANET_SEPARATION_Y
)

print("\nCombined aberrated wavefront created and ready for coronagraph propagation.")

# define propagator (pupil to focal)
focal_grid = hp.make_focal_grid(q=8, num_airy=20)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# obtain the wavefront at focal plane for the combined aberrated wavefront
focal_combined = prop.forward(wavefront_to_propagate)

# plot the focal plane intensity (star + planet) (before occulter, after lens 1)
hp.imshow_psf((focal_combined.intensity/focal_combined.intensity.max()),normalization='peak')
plt.colorbar(label='Contrast ($\log_{10}(I/I_{total})$)')
plt.title("Intensity (Focal Plane, Before Occulter, After Lens 1)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# create the Gaussian occulter mask

"""
# define propagator (pupil to focal)
focal_grid = hp.make_focal_grid(q=8, num_airy=20)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# obtain wavefront at telescope pupil plane for the star
wavefront_star = hp.Wavefront(telescope_pupil)

contrast = 1e-14 # Planet-to-star contrast
sqrt_contrast = np.sqrt(contrast) # Planet-to-star contrast (note: sqrt because we are working with the electric field)

# Planet offset in units of lambda/D
planet_offset_x = 15
planet_offset_y = 0
planet_offset_x = planet_offset_x/diameter
planet_offset_y = planet_offset_y/diameter
wavefront_planet = hp.Wavefront(sqrt_contrast * telescope_pupil * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x) * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y))

# obtain total wavefront intensity at pupil plane
wavefront_total_intensity = wavefront_star.intensity + wavefront_planet.intensity

# obtain the wavefront intensity at focal plane for the star
focal_star = prop.forward(wavefront_star)

# obtain the wavefront intensity at focal plane for the planet
focal_planet = prop.forward(wavefront_planet)

# obtain total wavefront intensity at focal plane
focal_total_intensity = focal_star.intensity + focal_planet.intensity

# plot the focal plane intensity (star + planet) (before occulter, after lens 1)
hp.imshow_field(np.log10(focal_total_intensity/focal_total_intensity.max()))
plt.colorbar(label='Contrast ($\log_{10}(I/I_{total})$)')
plt.title("Intensity (Focal Plane, Before Occulter, After Lens 1)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# create the Gaussian occulter mask
sigma_lambda_d = 5
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask,focal_grid)

# plot the focal plane intensity (star + planet) with occulter (focal plane, after lens 1)
E_focal_total = focal_star.electric_field + focal_planet.electric_field

# apply the occulter mask (Field * Field multiplication IS SUPPORTED for Field/Field on the same grid)
E_focal_after_occulter = E_focal_total * occulter_mask

# Calculate the intensity for plotting (Intensity = |E|^2)
I_focal_after_occulter = np.abs(E_focal_after_occulter)**2

# plot the focal plane intensity (star + planet) with occulter (focal plane, after lens 1) (not a log scale)
hp.imshow_field(I_focal_after_occulter/I_focal_after_occulter.max())
plt.colorbar(label='Contrast ($I/I_{total}$)')
plt.title("Intensity (Focal Plane, AFTER Occulter)")
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

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
ratio = 0.7 # Lyot Stop diameter ratio
lyot_stop_generator = hp.make_circular_aperture(ratio*diameter) # percentage of the telescope diameter
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