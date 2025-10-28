import numpy as np
import hcipy as hp
import matplotlib.pyplot as plt

def create_combined_aberration_wavefront(hsm, aper, pupil_diameter, pupil_grid, wavelength, focal_length, planet_contrast, separation_lambda_d_x, separaton_lambda_d_y):
    """
    Creates the star and planet wavefronts, applies the DM aberration to both,
    and returns the combined aberrated wavefront in the pupil plane.
    """
    # --- STAR WAVEFRONT ---
    # Star is an on-axis point source
    star_wavefront_initial = hp.Wavefront(aper, wavelength)
    # Apply DM aberration to the star's wavefront
    wavefront_star_aberrated = hsm.forward(star_wavefront_initial)
    wavefront_star_aberrated = hsm(star_wavefront_initial)

    # --- PLANET WAVEFRONT ---
    # Planet is an off-axis point source. 
    # Its light intensity is reduced by the contrast factor (amplitude scaled by sqrt(contrast)).
    # separation_m converts lambda/D to physical meters on the focal grid.
    separation_m = separation_lambda_d_x * wavelength * focal_length / pupil_diameter

    planet_position = np.array([separation_m, 0]) # Planet at 10 lambda/D on the x-axis
    
    # hp.make_point_source creates a wavefront with the correct tilt for the off-axis position
    # planet_source_wavefront = hp.make_point_source(pupil_grid, wavelength, planet_position)

    # Planet offset in units of lambda/D
    planet_offset_x = separation_lambda_d_x/pupil_diameter
    planet_offset_y = separaton_lambda_d_y/pupil_diameter
    planet_wavefront_initial = hp.Wavefront(np.sqrt(planet_contrast) * aper * np.exp(2j * np.pi * pupil_grid.x * planet_offset_x) * np.exp(2j * np.pi * pupil_grid.y * planet_offset_y))

    # Scale amplitude by sqrt(contrast)
    # planet_wavefront_initial = np.sqrt(planet_contrast) * planet_source_wavefront * aper
    
    # Apply the SAME DM aberration to the planet's wavefront (since they pass through the same optics)
    wavefront_planet_aberrated = hsm.forward(planet_wavefront_initial)
    wavefront_planet_aberrated = hsm(planet_wavefront_initial)

    # --- COMBINED ABERRATED WAVEFRONT (Coherent Sum) ---
    wavefront_combined_aberrated = wavefront_star_aberrated.electric_field + wavefront_planet_aberrated.electric_field
    wavefront_combined_aberrated = hp.Wavefront(wavefront_combined_aberrated, wavelength)
    
    # Plot the pupil plane intensity to verify the uniform amplitude (as requested)
    plt.figure(figsize=(8, 8))
    hp.imshow_field(wavefront_combined_aberrated.intensity)
    plt.title('Pupil Plane Intensity (Combined Star + Planet)')
    plt.colorbar(label='Intensity')
    plt.xlabel('x / D')
    plt.ylabel('y / D')
    plt.show()
    
    return wavefront_combined_aberrated