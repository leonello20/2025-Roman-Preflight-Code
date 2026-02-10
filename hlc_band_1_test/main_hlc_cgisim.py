# main_hlc_cgisim.py
import numpy as np
import cgisim
import matplotlib.pyplot as plt
from astropy.io import fits

def run_hlc_simulation():
    # 1. Initialize Deformable Mirrors (DMs)
    dm1_path = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\hlc_dm1.fits"
    dm2_path = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\preflight_data\\hlc_20190210b\\hlc_dm2.fits"
    dm1_path = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\examples\\hlc_ni_2e-9_dm1_v.fits"
    dm2_path = "C:\\Users\\leone\\OneDrive\\Documents\\GitHub\\2025-Roman-Preflight-Code\\roman_preflight_proper_public_v2.0.1_python\\roman_preflight_proper\\examples\\hlc_ni_2e-9_dm2_v.fits"
    dm1 = fits.getdata(dm1_path)
    dm2 = fits.getdata(dm2_path)

    # 2. Configure Parameters for HLC Band 1
    params = {
        'use_dm1': 1,
        'dm1_m': dm1, 
        'use_dm2': 1, 
        'dm2_m': dm2, 
        'source_x_offset_mas': 0.0,
        'use_errors': 0,
        'use_fpm': 1
    }
    print("Starting HLC Band 1 Simulation...")
    
    # 3. Execute the simulation
    # Star: A0V (standard Vega-like), Magnitude 2.0.
    image, counts = cgisim.rcgisim(
        'excam', 
        'hlc', 
        '1', 
        10, 
        params, 
        star_spectrum='a0v', 
        star_vmag=7.0
    )

    print("Simulation Complete.")
    return image, counts

def analyze_results(image, counts):
    # 1. Visualization
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log10(image + 1e-15), cmap='magma')
    plt.colorbar(label='Log10 Intensity')
    plt.title('HLC Band 1 - Raw Simulation Output (Uncorrected)')
    plt.show()

    # 2. Statistics
    print(f"Total Photons (Counts): {np.sum(counts):.2e}")
    print(f"Peak Pixel Value: {np.max(image):.2e}")
    print(f"Mean Intensity: {np.mean(image):.2e}")

    # 3. Save to FITS
    hdu = fits.PrimaryHDU(image)
    hdu.header['MODE'] = 'HLC Band 1'
    hdu.header['OFFSET'] = '0.0 mas'
    hdu.writeto('hlc_band1_result.fits', overwrite=True)
    print("Saved result to hlc_band1_result.fits")

if __name__ == "__main__":
    img, cnts = run_hlc_simulation()
    try:
        analyze_results(img, cnts)
    except NameError:
        print("Ensure 'img' and 'cnts' from the simulation are available.")

"""
import numpy as np
import cgisim
import matplotlib.pyplot as plt
from astropy.io import fits
import os

def run_hlc_streaks_sim():
    # Base path for the HLC config
    base_path = r"C:/Users/leone/OneDrive/Documents/GitHub/2025-Roman-Preflight-Code/roman_preflight_proper_public_v2.0.1_python/roman_preflight_proper/preflight_data/hlc_20190210b"
    
    # Load DMs to ensure the FPM is working in its designed 'dark hole' state
    # This contrast allows the faint diffraction streaks to become visible
    dm1_m = fits.getdata(os.path.join(base_path, "hlc_dm1.fits"))
    dm2_m = fits.getdata(os.path.join(base_path, "hlc_dm2.fits"))

    params = {
        'use_dm1': 1,
        'dm1_m': dm1_m,
        'use_dm2': 1,
        'dm2_m': dm2_m,
        'source_x_offset_mas': 0.0,  # Center it to see the radial symmetry of streaks
        'use_errors': 1,           # CRITICAL: Adds the struts/errors that cause streaks
    }

    print("Simulating HLC Band 1 with high spectral sampling...")
    
    # Using 21 samples over the 10% bandpass (Band 1)
    # This creates the 'smooth' radial streaks away from the center
    image, counts = cgisim.rcgisim(
        'excam', 
        'hlc', 
        '1', 
        10, # Higher sampling = smoother, sharper streaks
        params, 
        star_spectrum='a0v', 
        star_vmag=2.0
    )

    return image

def plot_streaks(image):
    plt.figure(figsize=(12, 10))
    
    # Log scale is essential. 
    # The 'sharper away from center' look comes from the log-stretching 
    # of the Airy rings and spider diffraction.
    norm_img = image / np.max(image)
    plt.imshow(np.log10(norm_img + 1e-9), origin='lower', cmap='inferno')
    
    plt.colorbar(label='Log10 Normalized Intensity')
    plt.title('HLC Diffraction Spikes & Speckle Elongation\n(Sharpness increases with radial distance)')
    plt.show()

if __name__ == "__main__":
    img = run_hlc_streaks_sim()
    plot_streaks(img)
"""


"""
import numpy as np
import cgisim
import matplotlib.pyplot as plt
import os

def run_and_plot_hlc():
    # 1. Setup Parameters
    # We use 'save_field' to tell the PROPER model to write the field to disk
    # Since the internal 'state' dictionary can be finicky in the wrapper,
    # we will read the file it creates, which is the most reliable way.
    
    output_dir = os.getcwd()
    lyot_file = os.path.join(output_dir, "field_lyot.fits")
    
    # Clean up old files if they exist to ensure we are seeing new data
    if os.path.exists(lyot_file):
        os.remove(lyot_file)

    params = {
        'use_errors': 1,
        'save_field': 'lyot',  # This triggers the FITS save at the Lyot plane
    }

    print("Running simulation (this takes ~1 minute)...")
    
    # Run in E-field mode to get the complex data at the detector
    # Mode 'excam_efield' returns (fields, sampling)
    efields, sampling = cgisim.rcgisim(
        'excam_efield', 
        'hlc', 
        '1', 
        0, 
        params, 
        star_spectrum='a0v', 
        star_vmag=2.0
    )

    print(f"Simulation finished. Final E-field shape: {efields.shape}")

    # 2. Create Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: Final Intensity at Detector (Log Scale) ---
    # We take the first wavelength [0] and calculate intensity (abs squared)
    intensity_final = np.abs(efields[0])**2
    
    im1 = axes[0].imshow(np.log10(intensity_final + 1e-15), origin='lower', cmap='magma')
    axes[0].set_title("Final Detector Intensity (Log10)")
    fig.colorbar(im1, ax=axes[0], label="Log10 Intensity")

    # --- Plot 2: Final Phase at Detector ---
    phase_final = np.angle(efields[0])
    
    im2 = axes[1].imshow(phase_final, origin='lower', cmap='twilight')
    axes[1].set_title("Final Detector Phase (Radians)")
    fig.colorbar(im2, ax=axes[1], label="Phase")

    plt.tight_layout()
    print("Displaying plots...")
    plt.show()

    # 3. Check for the Lyot file
    # If save_field worked, it wrote a file named field_lyot.fits
    if os.path.exists("field_lyot.fits"):
        print("Success! field_lyot.fits was created.")
    else:
        print("Note: field_lyot.fits was not found in the current directory.")

if __name__ == "__main__":
    run_and_plot_hlc()
"""