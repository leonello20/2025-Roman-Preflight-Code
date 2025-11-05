import numpy as np
import matplotlib.pyplot as plt
import hcipy as hp
import warnings
# Suppress RuntimeWarnings globally (cleaner output)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- SYSTEM PARAMETERS ---
pupil_diameter = 0.019725 # m
gap_size = 90e-6 # m
num_rings = 3
segment_flat_to_flat = (pupil_diameter - (2 * num_rings + 1) * gap_size) / (2 * num_rings + 1)
focal_length = 1 # m
wavelength = 638e-9 # m

# --- GRIDS ---
num_pix = 1024
pupil_grid = hp.make_pupil_grid(dims=num_pix, diameter=pupil_diameter)


# --- STEP 1: DEFINE THE MECHANICAL STRUCTURE (Aperture and DM) ---

# Define the segmented aperture mask and segment indices
aper, segments = hp.make_hexagonal_segmented_aperture(num_rings,
                                                         segment_flat_to_flat,
                                                         gap_size,
                                                         starting_ring=1,
                                                         return_segments=True)

aper = hp.evaluate_supersampled(aper, pupil_grid, 1)
segments = hp.evaluate_supersampled(segments, pupil_grid, 1)

# Instantiate the Segmented Deformable Mirror (hsm)
hsm = hp.SegmentedDeformableMirror(segments)
hsm.flatten() # Ensure it starts flat


# --- DM Helper Function ---
def aber_to_opd(aber_rad, wavelength):
    """Converts phase aberration in radians to OPD in meters."""
    aber_m = aber_rad * wavelength / (2 * np.pi)
    return aber_m

# --- STEP 2: INTRODUCE THE ABERRATION ---

aber_rad = 4.0 # Aberration phase in radians
opd_m = aber_to_opd(aber_rad, wavelength)
piston_poke_m = opd_m / 2 # The mirror OPD is half the total wavefront OPD

print(f"Applying piston of {piston_poke_m * 1e9:.2f} nm to segments 25 and 35.")

# Poke segments 35 and 25
for i in [35, 25]:
    hsm.set_segment_actuators(i, piston_poke_m, 0, 0)


# --- VISUALIZATION OF DM SURFACE (OPD) ---
plt.figure(figsize=(8,8))
plt.title('DM Surface OPD (Aberration)')
# The OPD is hsm.surface * 2 because the light travels to and from the mirror
hp.imshow_field(hsm.surface * 2, mask=aper, cmap='RdBu_r', 
                vmin=-opd_m, vmax=opd_m) 
plt.colorbar(label='Aberration OPD (m)')
plt.show()


# --- STEP 3: APPLY THE DM STATE TO A WAVEFRONT ---

# Create an initial flat wavefront (just the aperture)
planet_offset_x = 15
planet_offset_y = 0
planet_offset_x = planet_offset_x/pupil_diameter
planet_offset_y = planet_offset_y/pupil_diameter
contrast = 1e-10 # Planet-to-star contrast
sqrt_contrast = np.sqrt(contrast) # Planet-to-star contrast (note: sqrt because we are working with the electric field)
wavefront_planet = hp.Wavefront(sqrt_contrast * aper * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x) * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y))
wavefront_star = hp.Wavefront(aper, wavelength=wavelength)
wf_total = hp.Wavefront(wavefront_star.electric_field + wavefront_planet.electric_field, wavelength=wavelength)

# Apply the DM state (hsm) to the wavefront. 
# This adds the 4.0 rad phase error.
wf_aberrated = hsm(wf_total)


# --- VISUALIZATION OF ABERRATED WAVEFRONT INTENSITY AND PHASE ---
plt.figure(figsize=(15, 6))
plt.suptitle(r'Aberrated Wavefront in Pupil Plane ($\phi = 4.0$ rad)')

# 1. Intensity (should still look like the aperture)
plt.subplot(1, 2, 1)
hp.imshow_field(wf_aberrated.intensity, cmap='gray')
plt.title('Intensity')
plt.colorbar()

# 2. Phase (should show the two poked segments clearly)
plt.subplot(1, 2, 2)
# We plot the phase mod 2*pi to clearly show the piston jumps
hp.imshow_field(wf_aberrated.phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
plt.title('Phase (radians)')
plt.colorbar()

plt.show()