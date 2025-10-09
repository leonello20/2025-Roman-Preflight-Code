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
contrast = 1e-2 # Planet-to-star contrast
# Planet offset in units of lambda/D
planet_offset_x = 15
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
def eigth_order_mask(grid,l,m,a,epsilon):
    x = grid.x
    x = np.sqrt(grid.x**2 + grid.y**2)
    transmission_field = 0
    if (x.any() == 0):
        transmission_field = 0
    else:
        transmission_field = a*(((l-m)/l - (np.sinc(np.pi*x*epsilon/l)**l) + (m/l)*(np.sinc(np.pi*x*epsilon/m)**m)))**2
    return transmission_field

# create the Lyot Stop mask
ratio = 1.0 # Lyot Stop diameter ratio
lyot_stop_generator = hp.make_circular_aperture(ratio*diameter) # percentage of the telescope diameter
lyot_stop_mask = lyot_stop_generator(pupil_grid)

# create the occulter mask (8th order mask)
l = 3
m = 1
a = 2
epsilon = 0.1
occulter_mask = eigth_order_mask(focal_grid,l,m,a,epsilon)
occulter_mask = hp.Field(occulter_mask,focal_grid)
prop_lyot = hp.LyotCoronagraph(focal_grid,occulter_mask,lyot_stop_mask)
focal_star_occulter_lyot = prop_lyot.forward(wavefront_star)
focal_planet_occulter_lyot = prop_lyot.forward(wavefront_planet)
focal_total_occulter_intensity_lyot = focal_star_occulter_lyot.intensity + focal_planet_occulter_lyot.intensity

# plot the focal plane intensity (star + planet) with occulter
hp.imshow_field(np.log10(focal_total_occulter_intensity_lyot/focal_total_occulter_intensity_lyot.max()))
plt.title("Final Coronagraphic Image (Occulter + Lyot Stop)")
plt.colorbar(label='Contrast ($\log_{10}(I/I_{star})$)')
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()

# propagate the wavefront to the focal plane
wavefront_focal_after_occulter_star = prop.forward(focal_star_occulter_lyot)
wavefront_focal_after_occulter_planet = prop.forward(focal_planet_occulter_lyot)
wavefront_focal_after_occulter_total_intensity = wavefront_focal_after_occulter_star.intensity + wavefront_focal_after_occulter_planet.intensity

# plot the focal plane intensity (star + planet) after occulter
hp.imshow_field(np.log10(wavefront_focal_after_occulter_total_intensity/wavefront_focal_after_occulter_total_intensity.max()))
plt.colorbar()
plt.xlabel('x / D')
plt.ylabel('y / D')
plt.show()