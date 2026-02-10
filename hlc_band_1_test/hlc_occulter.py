import proper
import numpy as np
import matplotlib.pylab as plt
import astropy.io.fits as fits
from coronagraph import coronagraph

def hlc_occulter(wavelength, diam, grid_size, beam_ratio, f_lens, pupil, fpm_real, fpm_imag, dm1, dm2, lyot_stop):
    pupil_sampling = diam / (grid_size * beam_ratio)
    # pupil_sampling = diam/grid_size

    # 1. Initialize the wavefront at the entrance pupil
    wfo = proper.prop_begin(diam, wavelength, grid_size, beam_ratio)
    # wfo.wfarr = np.fft.fftshift(wfo.wfarr)
    # proper.prop_circular_aperture(wfo, diam)
    # proper.prop_define_entrance(wfo)

    # 2. Read the pupil map and apply it to the wavefront
    shift = 1.5*diam
    pupil_map = proper.prop_readmap(wfo, pupil, shift, shift, SAMPLING=pupil_sampling)
    pupil = fits.getdata(pupil)
    pupil = proper.prop_shift_center(pupil)
    pupil = proper.prop_shift_center(pupil)
    # pupil = proper.prop_shift_center(pupil)
    proper.prop_multiply(wfo, pupil_map)

    amplitude = proper.prop_get_amplitude(wfo)
    # amplitude = np.fft.fftshift(amplitude)

    # 3. Plot the entrance pupil
    plt.figure(figsize=(12,8))
    plt.imshow(amplitude, origin = "lower", cmap = plt.cm.gray)
    plt.title("Entrance Pupil", fontsize = 18)
    plt.show()

    dm1 = fits.getdata(dm1)
    dm2 = fits.getdata(dm2)
    proper.prop_dm(wfo, dm1, 0.0, 0.0, 0.2, n_actuators=48)
    proper.prop_propagate(wfo, 1.0, "DM2")
    proper.prop_dm(wfo, dm2, 0.0, 0.0, 0.2, n_actuators=48)

    # 4. Plot the pupil plane after DMs
    plt.figure(figsize=(12,8))
    amplitude_after_dms = proper.prop_get_amplitude(wfo)
    amplitude_after_dms = np.fft.fftshift(amplitude_after_dms)
    plt.imshow(amplitude_after_dms, origin = "lower", cmap = plt.cm.gray)
    plt.title("Pupil Plane - After DMs", fontsize = 18)
    plt.show()

    # 5. Propagate to the focal plane (F1) to see the effect of DMs before the occulter
    proper.prop_lens(wfo, f_lens, "Focal Plane after DMs")
    proper.prop_propagate(wfo, f_lens, "Focal Plane after DMs")
    plt.figure(figsize=(12,8))
    plt.imshow((proper.prop_get_amplitude(wfo))**(1/4), origin = "lower", cmap = plt.cm.gray)
    plt.title("Focal Plane - After DMs", fontsize = 18)
    plt.show()

    # 6. Multiply by the occulter
    shift = 0.002
    fpm_s = proper.prop_get_sampling(wfo)
    fpm_real_map = proper.prop_readmap(wfo, fpm_real, shift, shift, SAMPLING=fpm_s)
    fpm_imag_map = proper.prop_readmap(wfo, fpm_imag, shift, shift, SAMPLING=fpm_s)
    # fpm_real_map = proper.prop_readmap(wfo, fpm_real, SAMPLING=proper.prop_get_sampling(wfo))
    # fpm_imag_map = proper.prop_readmap(wfo, fpm_imag, SAMPLING=proper.prop_get_sampling(wfo))
    fpm_complex = fpm_real_map + 1j * fpm_imag_map
    proper.prop_multiply(wfo, fpm_complex)

    # 7. Plot the wavefront after the occulter
    plt.figure(figsize=(12,8))
    plt.imshow((proper.prop_get_amplitude(wfo))**(1/4), origin = "lower", cmap = plt.cm.gray)
    plt.title("Focal Plane 1 - After Occulter", fontsize = 18)
    plt.show()

    # 8. Propagate to Lyot Plane
    proper.prop_propagate(wfo, f_lens, "Lyot Plane - with Lyot Stop")
    proper.prop_lens(wfo, f_lens, "Lyot Plane - no Lyot Stop")
    proper.prop_propagate(wfo, f_lens, "Lyot Plane - with Lyot Stop")
    # proper.prop_lens(wfo, f_lens, "Lyot Plane - no Lyot Stop")
    # proper.prop_propagate(wfo, f_lens, "Lyot Plane - with Lyot Stop")
    # proper.prop_lens(wfo, f_lens, "Lyot Plane - no Lyot Stop")
    # proper.prop_propagate(wfo, f_lens, "Lyot Plane - with Lyot Stop")

    # 9. Plot the wavefront at the Lyot Plane before Lyot Stop
    plt.figure(figsize=(12,8))
    amplitude_before_lyot = proper.prop_get_amplitude(wfo)
    amplitude_before_lyot = np.fft.fftshift(amplitude_before_lyot)
    plt.imshow(amplitude_before_lyot, origin = "lower", cmap = plt.cm.gray)
    plt.title("Lyot Plane - Before Lyot Stop", fontsize = 18)
    plt.show()

    # 10. Apply Lyot Stop using Lyot Stop file
    lyot_map = proper.prop_readmap(wfo, lyot_stop, SAMPLING=proper.prop_get_sampling(wfo))
    proper.prop_multiply(wfo, lyot_map)

    # 11. Plot the wavefront at the Lyot Plane after Lyot Stop
    plt.figure(figsize=(12,8))
    amplitude_after_lyot = proper.prop_get_amplitude(wfo)
    amplitude_after_lyot = np.fft.fftshift(amplitude_after_lyot)
    plt.imshow(amplitude_after_lyot, origin = "lower", cmap = plt.cm.gray)
    plt.title("Lyot Plane - After Lyot Stop", fontsize = 18)
    plt.show()

    # 12. Propagate to the final image plane (detector)
    # proper.prop_lens(wfo, f_lens, "Detector")
    proper.prop_propagate(wfo, f_lens, "Detector")
    proper.prop_lens(wfo, f_lens, "Detector")
    proper.prop_propagate(wfo, f_lens, "Detector")
    amplitude_detector = proper.prop_get_amplitude(wfo)
    # amplitude_detector = np.fft.fftshift(amplitude_detector)
    plt.figure(figsize=(12,8))
    plt.imshow((amplitude_detector)**(1/4), origin = "lower", cmap = plt.cm.gray)
    plt.title("Final Image Plane (Detector)", fontsize = 18)
    plt.show()

    # 13. End the PROPER simulation
    sampling = proper.prop_get_sampling(wfo)
    return (wfo, sampling)