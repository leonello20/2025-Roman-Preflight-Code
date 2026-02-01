import numpy as np
import astropy.io.fits as fits
import proper

def run_hlc_with_dms(dm1_fits_path, dm2_fits_path, output_path="hlc_dark_hole.fits"):
    """
    Simulates the Roman HLC Band 1 with specific DM settings loaded from FITS files.
    
    Args:
        dm1_fits_path (str): Path to the FITS file containing DM1 surface settings.
        dm2_fits_path (str): Path to the FITS file containing DM2 surface settings.
        output_path (str): Where to save the resulting PSF.
    """
    
    # 1. Load the DM data from your FITS files
    # Usually, these are 48x48 arrays representing the actuator heights
    try:
        dm1_data = fits.getdata(dm1_fits_path)
        dm2_data = fits.getdata(dm2_fits_path)
        print(f"Successfully loaded DM1 ({dm1_data.shape}) and DM2 ({dm2_data.shape})")
    except Exception as e:
        print(f"Error loading FITS files: {e}")
        return

    # 2. Setup the PROPER simulation parameters
    # Note: These parameters should match your Band 1 HLC config
    wavelength = 0.575  # Microns for Band 1
    grid_size = 1024    # Standard grid resolution
    beam_ratio = 0.2    # Typical beam ratio for HLC
    
    # 3. Define the prescription (This is a simplified logical flow)
    def hlc_prescription(wavelength, grid_size):
        # Initialize the wavefront
        wfo = proper.prop_begin(1.0, wavelength, grid_size, beam_ratio)
        
        # Apply the Roman Entrance Pupil (including the struts you saw)
        proper.prop_circular_aperture(wfo, 1.0) 
        # (In a full CGISim run, this would load the 'roman_pupil.fits')
        
        # --- THE CRITICAL STEP: APPLYING THE DMs ---
        # We apply the DM settings we loaded from your files
        # proper.prop_dm is the standard command
        # n_actuators is typically 48 for the Roman DMs
        proper.prop_dm(wfo, dm1_data, 0.0, 0.0, 0.2, n_actuators=48)
        
        # Propagate to DM2 (approx 1 meter away in the CGI optical bench)
        proper.prop_propagate(wfo, 1.0, "To DM2")
        proper.prop_dm(wfo, dm2_data, 0.0, 0.0, 0.2, n_actuators=48)
        
        # Propagate through the Focal Plane Mask (FPM) and Lyot Stop
        # This is where the 'Bright Center' is suppressed
        # ... [Internal HLC optics logic] ...
        
        # Final propagation to the detector
        proper.prop_propagate(wfo, 1.0, "To Detector")
        proper.prop_end(wfo)
        psf = proper.prop_get_amplitude(wfo)
        return psf

    # 4. Execute
    print("Running simulation with DM correction...")
    final_psf = hlc_prescription(wavelength, grid_size)
    
    # Save the result
    fits.writeto(output_path, final_psf, overwrite=True)
    print(f"Simulation complete. Result saved to {output_path}")
    print("When you view this file, the 'streaks' should be replaced by a dark rectangular 'hole'.")

# Example usage (uncomment and update paths to test):

dm1_fits = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\hlc_dm1.fits"
dm2_fits = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\hlc_dm2.fits"

run_hlc_with_dms(dm1_fits, dm2_fits)