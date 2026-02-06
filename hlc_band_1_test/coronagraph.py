# coronagraph.py
import proper
import numpy as np
import matplotlib.pylab as plt

def coronagraph(wfo, f_lens, fpm_real, fpm_imag, dm1, dm2):
    # 1. Propogate to Focal Plane 1 (F1 - Occulter Plane)
    proper.prop_lens(wfo, f_lens, "F1 - no occulter")
    proper.prop_propagate(wfo, f_lens, "F1 - occulter plane")
    
    # 2. Plotting before occulter
    plt.figure(figsize=(12,8))
    plt.imshow(proper.prop_get_amplitude(wfo), origin = "lower", cmap = plt.cm.gray)
    plt.show()
    lamda = proper.prop_get_wavelength(wfo)

    # 3. Multiply by the occulter
    fpm_real_map = proper.prop_readmap(wfo, fpm_real, SAMPLING=proper.prop_get_sampling(wfo))
    fpm_imag_map = proper.prop_readmap(wfo, fpm_imag, SAMPLING=proper.prop_get_sampling(wfo))
    print(proper.prop_get_sampling(wfo))
    fpm_complex = fpm_real_map + 1j * fpm_imag_map
    proper.prop_multiply(wfo, fpm_complex)

    # 4. Plot the wavefront after the occulter
    plt.figure(figsize=(12,8))
    plt.imshow(proper.prop_get_amplitude(wfo), origin = "lower", cmap = plt.cm.gray)
    plt.show()

    return wfo

"""
# def coronagraph(wfo, f_lens, occulter_type, diam, offset_x=0.0, offset_y=0.0, zernike_coeffs=None, is_plotting=False):

def coronagraph(wfo, f_lens, occulter_type, diam, occrad):
    print(f"\n--- Entering coronagraph() ---")
    print(f"DEBUG (coronagraph init): Initial sampling: {proper.prop_get_sampling(wfo):.2e} m/pixel")
    print(f"DEBUG (coronagraph init): f_lens: {f_lens:.2f} meters")
    print(f"DEBUG (coronagraph init): diam: {diam:.2f} meters")

    # --- 1. Propagate from Entrance Pupil (EPP) to Focal Plane 1 (F1 - Occulter Plane) ---
    # This combination correctly takes us from a pupil plane to a focal plane.
    proper.prop_lens(wfo, f_lens, "coronagraph imaging lens") # Lens 1
    print(f"DEBUG: After Lens 1: {proper.prop_get_sampling(wfo):.2e} m/pixel")
    proper.prop_propagate(wfo, f_lens, "occulter") # Propagate f_lens to reach focal plane
    # EXPECTED Sampling at F1 for beam_ratio=1.0: (wavelength * f_lens) / diam = (0.5e-6 * 2.4) / 0.1 = 1.2e-5 m/pixel
    print(f"DEBUG: Sampling at occulter plane (F1): {proper.prop_get_sampling(wfo):.2e} m/pixel (Expected ~1.2e-5)")

    lamda = proper.prop_get_wavelength(wfo)

    occrad_rad = occrad * lamda / diam
    dx_m = proper.prop_get_sampling(wfo)
    dx_rad = proper.prop_get_sampling_radians(wfo)
    occrad_m = occrad_rad * dx_m / dx_rad

    plt.figure(figsize=(12,8))

    # Apply the chosen occulter type
    if occulter_type == "GAUSSIAN":
        r = proper.prop_radius(wfo)
        h = occrad_m
        gauss_spot = 1 - np.exp(-0.5 * (r/h)**2)
        proper.prop_multiply(wfo, gauss_spot)
        plt.suptitle(f"Gaussian spot (h={occrad} lamda/D)", fontsize = 18)
    elif occulter_type == "SOLID":
        proper.prop_circular_obscuration(wfo, occrad_m)
        plt.suptitle("Solid spot", fontsize = 18)
    elif occulter_type == "8TH_ORDER":
        print("Warning: prop_8th_order_mask may not be directly available in standard PROPER. Using circular obscuration as placeholder.")
        proper.prop_circular_obscuration(wfo, occrad_m)
        plt.suptitle("8th order band limited spot (using placeholder)", fontsize = 18)

    plt.subplot(1,2,1)
    plt.imshow(np.sqrt(proper.prop_get_amplitude(wfo)), origin = "lower", cmap = plt.cm.gray)
    plt.text(proper.prop_get_gridsize(wfo) * 0.78, proper.prop_get_gridsize(wfo) * 0.05, "After Occulter", color = "w", horizontalalignment='right')


    # --- 2. Propagate from Focal Plane 1 (F1) to Pupil Plane 2 (PP2 - Lyot Stop Plane) ---
    # From F1, propagate f_lens to Lens 2 (L2)
    proper.prop_propagate(wfo, f_lens, "Propagate from F1 to L2")
    print(f"DEBUG: After Propagate F1 to L2: {proper.prop_get_sampling(wfo):.2e} m/pixel") # Sampling should get coarser as beam expands

    # Apply Lens 2 (L2)
    proper.prop_lens(wfo, f_lens, "pupil reimaging lens (L2)")
    print(f"DEBUG: After Lens 2: {proper.prop_get_sampling(wfo):.2e} m/pixel") # Sampling should be at the lens, not changed yet.

    # Propagate from L2 to Lyot Stop plane (PP2). This is another f_lens.
    proper.prop_propagate(wfo, f_lens, "Propagate L2 to Lyot Stop (PP2)")
    # EXPECTED Sampling at PP2: Should be back to initial pupil-like sampling (diam / grid_size = 0.1 / 512 = 1.95e-4 m/pixel)
    print(f"DEBUG: Sampling at Lyot Stop plane (PP2): {proper.prop_get_sampling(wfo):.2e} m/pixel (Expected pupil-like)")
    plt.subplot(1,2,2)
    plt.imshow(proper.prop_get_amplitude(wfo)**0.2, origin = "lower", cmap = plt.cm.gray)
    plt.text(proper.prop_get_gridsize(wfo), proper.prop_get_gridsize(wfo), "Before Lyot Stop", color = "w", horizontalalignment='right')
    plt.show()

    if occulter_type == "GAUSSIAN":
        proper.prop_circular_aperture(wfo, 0.80, NORM = True)
    elif occulter_type == "SOLID":
        proper.prop_circular_aperture(wfo, 0.84, NORM = True)
    elif occulter_type == "8TH_ORDER":
        proper.prop_circular_aperture(wfo, 0.50, NORM = True)
    
    # --- 3. Propagate from Pupil Plane 2 (PP2) to Final Focal Plane (F2 - Image Plane) ---
    proper.prop_propagate(wfo, f_lens, "Propagate PP2 to L3")
    print(f"DEBUG: After Propagate PP2 to L3: {proper.prop_get_sampling(wfo):.2e} m/pixel")

    proper.prop_lens(wfo, f_lens, "final imaging lens (L3)")
    print(f"DEBUG: After Lens 3: {proper.prop_get_sampling(wfo):.2e} m/pixel")

    proper.prop_propagate(wfo, f_lens, "final focus (F2)")
    # EXPECTED Final Sampling at F2: Should be same as F1 (1.2e-5 m/pixel)
    print(f"DEBUG: Final sampling in coronagraph (F2): {proper.prop_get_sampling(wfo):.2e} m/pixel (Expected ~1.2e-5)")

    print(f"--- Exiting coronagraph() ---")
    return
"""