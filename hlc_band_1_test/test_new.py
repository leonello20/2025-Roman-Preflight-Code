import numpy as np
import astropy.io.fits as fits
import proper
import os

def run_hlc_normalized(dm1_path, dm2_path, output_path="hlc_normalized_result.fits"):
    """
    Runs HLC simulation and normalizes the output to the peak of an unobstructed PSF.
    """
    
    # 1. Load DM data
    dm1_data = fits.getdata(dm1_path)
    dm2_data = fits.getdata(dm2_path)
    
    wavelength = 0.575e-6
    grid_size = 1024
    beam_ratio = 0.2

    def hlc_prescription(wavelength, grid_size, use_coronagraph=True):
        wfo = proper.prop_begin(1.0, wavelength, grid_size, beam_ratio)
        proper.prop_circular_aperture(wfo, 1.0)
        proper.prop_define_entrance(wfo)
        
        if use_coronagraph:
            # Apply DMs
            proper.prop_dm(wfo, dm1_data, 0.0, 0.0, 0.2, n_actuators=48)
            proper.prop_propagate(wfo, 1.0, "DM2")
            proper.prop_dm(wfo, dm2_data, 0.0, 0.0, 0.2, n_actuators=48)
            
            # Note: In a real HLC, you'd add the Focal Plane Mask (FPM) here
            # For now, we are simulating the "Pre-FPM" or "Post-DM" wavefront structure
            proper.prop_propagate(wfo, 1.0, "Detector")
        else:
            # Reference run: No DMs, just straight propagation to see peak intensity
            proper.prop_propagate(wfo, 2.0, "Detector")

        (wavefront, sampling) = proper.prop_end(wfo)
        return wavefront

    # --- Step 1: Get Reference Peak ---
    print("Running reference (unobstructed) PSF...")
    ref_wavefront = hlc_prescription(wavelength, grid_size, use_coronagraph=False)
    ref_intensity = np.abs(ref_wavefront)**2
    peak_val = np.max(ref_intensity)
    print(f"Reference peak intensity: {peak_val:.2e}")

    # --- Step 2: Get Simulation Result ---
    print("Running simulation with DMs...")
    wave_array = hlc_prescription(wavelength, grid_size, use_coronagraph=True)
    intensity = np.abs(wave_array)**2
    
    # --- Step 3: Normalize to Contrast ---
    # This turns values into "Contrast" (e.g., 10^-9 means 1 billionth of the star's peak)
    contrast_map = intensity / peak_val
    log_contrast = np.log10(contrast_map + 1e-15)

    # Save
    hdul = fits.HDUList([
        fits.PrimaryHDU(data=contrast_map.astype(np.float32)),
        fits.ImageHDU(data=log_contrast.astype(np.float32), name="LOG_CONTRAST")
    ])
    
    hdul.writeto(output_path, overwrite=True)
    print(f"Saved. Max contrast in dark hole area: {np.min(contrast_map):.2e}")


if __name__ == "__main__":
    dm1_fits = r"C:\Users\leone\OneDrive\Documents\GitHub\2025-Roman-Preflight-Code\roman_preflight_proper_public_v2.0.1_python\roman_preflight_proper\preflight_data\hlc_20190210b\hlc_dm1.fits"
    dm2_fits = r"C:\Users\leone\OneDrive\Documents\GitHub\2025-Roman-Preflight-Code\roman_preflight_proper_public_v2.0.1_python\roman_preflight_proper\preflight_data\hlc_20190210b\hlc_dm2.fits"
    run_hlc_normalized(dm1_fits, dm2_fits)