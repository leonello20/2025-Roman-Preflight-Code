# main.py
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
import proper
# from pixel_to_lambda_D import pixel_to_lambda_D
# from contrast_curve import contrast_curve
from hlc_occulter import hlc_occulter
# from contrast_2D import contrast_2D

wavelength = 0.54625000e-6  # Wavelength in meters (500 nm)
grid_size = 311 # Grid size for the simulation
occulter_type = 'GAUSSIAN'  # Type of occulter to use
diam_telescope = 2.363  # Telescope diameter in meters
scale_occulter = 0.1  # Scale factor for the occulter sampling (if needed)
scale_no_occulter = 0 # ensure that the no occulter case is not scaled, so we can compare directly to the occulter case
occrad = 1.0  # Occulter radius in lambda/D units
f_lens = 24 * diam_telescope
beam_ratio = 1

pupil = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\pupil.fits"
dm1 = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\hlc_dm1.fits"
dm2 = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\hlc_dm2.fits"
fpm_real = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\hlc_fpm_trans_0.54625000um_real.fits"
fpm_imag = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\hlc_fpm_trans_0.54625000um_imag.fits"
lyot_stop = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\lyot.fits"

(unaberrated_star_wfo, sampling) = hlc_occulter(wavelength, diam_telescope, scale_no_occulter, grid_size, beam_ratio, f_lens, pupil, fpm_real, fpm_imag, dm1, dm2, lyot_stop)
(final_wavefront_coronagraph, sampling) = hlc_occulter(wavelength, diam_telescope, scale_occulter, grid_size, beam_ratio, f_lens, pupil, fpm_real, fpm_imag, dm1, dm2, lyot_stop)

# Ratio of max amplitude with occulter to max amplitude without occulter (should be << 1 if the occulter is working)

wf_max_ratio = np.max(proper.prop_get_amplitude(final_wavefront_coronagraph)) / np.max(proper.prop_get_amplitude(unaberrated_star_wfo))
print(f"Ratio of max amplitude with occulter to max amplitude without occulter: {wf_max_ratio:.2e}")


"""
# lambda/D separation for the central horizontal cross-section
lambda_D_separation = pixel_to_lambda_D(final_wavefront_coronagraph, diam_telescope, grid_size)
# Calculate contrast curve
contrast = contrast_curve(unaberrated_star_wfo, final_wavefront_coronagraph, grid_size)

# Plot Contrast Curve

plt.figure(figsize=(12, 8))
plt.title("Contrast Curve")
plt.xlabel("Angular Separation ($\\lambda/D$)")
plt.ylabel("Contrast")
plt.semilogy(lambda_D_separation, contrast)
plt.show()

# 2D Contrast Map
contrast_2D(unaberrated_star_wfo, final_wavefront_coronagraph, grid_size)
"""