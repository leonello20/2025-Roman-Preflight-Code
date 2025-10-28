import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import hcipy as hp

pupil_diameter = 0.019725 # m
gap_size = 90e-6 # m
num_rings = 3
segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)
focal_length = 1 # m

# Parameters for the simulation
num_pix = 1024
wavelength = 638e-9
num_airy = 20
sampling = 4
norm = False

# HCIPy grids and propagator
pupil_grid = hp.make_pupil_grid(dims=num_pix, diameter=pupil_diameter)

focal_grid = hp.make_focal_grid(sampling, num_airy,
                                   pupil_diameter=pupil_diameter,
                                   reference_wavelength=wavelength,
                                   focal_length=focal_length)
focal_grid = focal_grid.shifted(focal_grid.delta / 2)

prop = hp.FraunhoferPropagator(pupil_grid, focal_grid, focal_length)

aper, segments = hp.make_hexagonal_segmented_aperture(num_rings,
                                                         segment_flat_to_flat,
                                                         gap_size,
                                                         starting_ring=1,
                                                         return_segments=True)

aper = hp.evaluate_supersampled(aper, pupil_grid, 1)
segments = hp.evaluate_supersampled(segments, pupil_grid, 1)

plt.title('HCIPy aperture')
hp.imshow_field(aper, cmap='gray')

# Instantiate the segmented mirror
hsm = hp.SegmentedDeformableMirror(segments)

# Make a pupil plane wavefront from aperture
wf_star = hp.Wavefront(aper, wavelength)

# Apply SM if you want to
wf_star = hsm(wf_star)

# Planet offset in units of lambda/D
sqrt_contrast = 1
planet_offset_x = 15
planet_offset_y = 0
planet_offset_x = planet_offset_x/pupil_diameter
planet_offset_y = planet_offset_y/pupil_diameter
wavefront_planet = hp.Wavefront(sqrt_contrast * aper * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x) * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y))

wf = hp.Wavefront(wf_star.electric_field + wavefront_planet.electric_field, wavelength)

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

# Flatten both SMs just to be sure
hsm.flatten()

# Define function from rad of phase to m OPD
def aber_to_opd(aber_rad, wavelength):
    aber_m = aber_rad * wavelength / (2 * np.pi)
    return aber_m

aber_rad = 4.0

print('Aberration: {} rad'.format(aber_rad))
print('Aberration: {} m'.format(aber_to_opd(aber_rad, wavelength)))

# Poking segment 35 and 25
for i in [35, 25]:
    hsm.set_segment_actuators(i, aber_to_opd(aber_rad, wavelength) / 2, 0, 0)

# Display both segmented mirrors in OPD

# HCIPy
plt.figure(figsize=(8,8))
plt.title('OPD for HCIPy SM')
hp.imshow_field(hsm.surface * 2, mask=aper, cmap='RdBu_r', vmin=-5e-7, vmax=5e-7)
plt.colorbar()
plt.show()

### HCIPy
# Apply SM to pupil plane wf
wf_fp_pistoned = hsm(wf)

# Propagate from SM to image plane
im_pistoned_hc = prop(wf_fp_pistoned)

### Display intensity of image plane
plt.figure(figsize=(15, 6))
plt.suptitle('Image plane after SM for $\phi$ = ' + str(aber_rad) + ' rad')

hp.imshow_field(np.log10(im_pistoned_hc.intensity / norm_hc), cmap='inferno', vmin=-9)
plt.title('HCIPy pistoned pair')
plt.colorbar()
plt.show()