import hcipy as hp
import numpy as np
import matplotlib.pyplot as plt
import warnings
# Suppress RuntimeWarnings globally
warnings.filterwarnings("ignore", category=RuntimeWarning)

aperture_scale = 1.5
grid_size = 256
pupil_grid = hp.make_pupil_grid(grid_size,aperture_scale)
diameter = 1 # meters

telescope_pupil_generator = hp.make_circular_aperture(diameter)

telescope_pupil = telescope_pupil_generator(pupil_grid)

# plot the pupil
im = hp.imshow_field(telescope_pupil, cmap='gray')
plt.colorbar()
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# define propagator (pupil to focal)
focal_grid = hp.make_focal_grid(q=8, num_airy=16)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

# obtain wavefront at telescope pupil and focal planes for the star
wavefront_star = hp.Wavefront(telescope_pupil)
focal_star = prop.forward(wavefront_star)

# obtain wavefront at telescope pupil and focal planes for the planet
contrast = 1e-2  # Planet-to-star contrast
# Planet offset in units of lambda/D
planet_offset_x = 12
planet_offset_y = 0
wavefront_planet = hp.Wavefront(contrast * telescope_pupil * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x) * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y))
focal_planet = prop.forward(wavefront_planet)

# obtain total wavefront intensity at pupil plane
wavefront_total_intensity = wavefront_star.intensity + wavefront_planet.intensity

# obtain total wavefront intensity at focal plane
focal_total_intensity = focal_star.intensity + focal_planet.intensity

# plot the focal plane intensity (star + planet)
hp.imshow_field(np.log10(focal_total_intensity/focal_total_intensity.max()))
plt.colorbar()
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# apply Gaussian occulter in focal plane (Lyot Coronagraph)
def gaussian_occulter_generator(grid,sigma_lambda_d):
    # 1. Use the existing grid.r Field for all calculations.
    # This guarantees that the intermediate result (r_term) is an hcipy Field.
    r_term = (np.sqrt(grid.x**2 + grid.y**2) / sigma_lambda_d)
        
    # 2. Perform the mathematical operations on the hcipy Field (r_term).
    # This keeps the result as an hcipy Field object.
    transmission_field = 1.0 - np.exp(-0.5 * r_term**2)
    return transmission_field

# create the occulter mask
sigma_lambda_d = 12
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask,focal_grid)
prop_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask)
focal_star_occulter = prop_lyot.forward(wavefront_star)
focal_planet_occulter = prop_lyot.forward(wavefront_planet)
focal_total_occulter_intensity = focal_star_occulter.intensity + focal_planet_occulter.intensity

# plot the focal plane intensity (star + planet) with occulter
hp.imshow_field(np.log10(focal_total_occulter_intensity/focal_total_occulter_intensity.max()))
plt.colorbar()
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# propagate the wavefront back to the pupil plane
wavefront_pupil_after_occulter_star = prop.backward(focal_star_occulter)
wavefront_pupil_after_occulter_planet = prop.backward(focal_planet_occulter)
wavefront_pupil_after_occulter_total_intensity = wavefront_pupil_after_occulter_star.intensity + wavefront_pupil_after_occulter_planet.intensity

# plot the pupil plane intensity (star + planet) after occulter
hp.imshow_field(np.log10(wavefront_pupil_after_occulter_total_intensity/wavefront_pupil_after_occulter_total_intensity.max()))
plt.colorbar()
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()


"""
# propagate the wavefront to the focal plane
focal_grid = hp.make_focal_grid(q=8, num_airy=16)
prop = hp.FraunhoferPropagator(pupil_grid, focal_grid)

focal_image = prop.forward(wavefront)
hp.imshow_field(np.log10(focal_image.intensity/focal_image.intensity.max()), vmin=-5)
plt.xlabel('Focal place distance [$\lambda/D$]')
plt.ylabel('Focal plane distance [$\lambda/D$]')
plt.colorbar()
plt.show()

# create the focal plane mask occulter for the coronagraph
def gaussian_occulter_generator(grid,sigma_lambda_d):
        
    # 1. Use the existing grid.r Field for all calculations.
    #    This guarantees that the intermediate result (r_term) is an hcipy Field.
    r_term = (np.sqrt(grid.x**2 + grid.y**2) / sigma_lambda_d)
        
    # 2. Perform the mathematical operations on the hcipy Field (r_term).
    #    This keeps the result as an hcipy Field object.
    transmission_field = 1.0 - np.exp(-0.5 * r_term**2)
    return transmission_field

# create the occulter mask
sigma_lambda_d = 20
occulter_mask = gaussian_occulter_generator(focal_grid,sigma_lambda_d)
occulter_mask = hp.Field(occulter_mask,focal_grid)
prop_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask)

focal_image_occulter = prop_lyot.forward(wavefront)

# plot the focal image with occulter
hp.imshow_field(focal_image_occulter.intensity/focal_image.intensity.max())
plt.xlabel('Focal place distance [$\lambda/D$]')
plt.ylabel('Focal plane distance [$\lambda/D$]')
plt.colorbar()
plt.show()

# Planet offset in units of lambda/D
planet_offset_x = 4
planet_offset_y = 0

# wf_star is the initial on-axis wavefront for the star
wf_star = hp.Wavefront(telescope_pupil)

wf_planet = hp.Wavefront(telescope_pupil * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x) * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y))
focal_plane = prop.forward(wf_planet)

# hp.imshow_field(wf_planet.intensity)
contrast = 1.0
hp.imshow_field((np.log10(focal_image.intensity) + contrast*np.log10((focal_plane.intensity))), vmin=-5)

print(np.min(((focal_image.intensity) + contrast*(focal_plane.intensity))))
print(np.max(((focal_image.intensity) + contrast*(focal_plane.intensity))))
plt.colorbar()
plt.xlabel('Focal place distance [$\lambda/D$]')
plt.ylabel('Focal plane distance [$\lambda/D$]')
plt.show()
"""