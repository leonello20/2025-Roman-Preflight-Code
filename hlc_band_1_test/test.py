import numpy as np
import cgisim  # Ensure cgisim is installed and in your path
import matplotlib.pyplot as plt
from astropy.io import fits

def run_hlc_simulation():
    """
    Sets up and runs the HLC Band 1 simulation based on arXiv:2309.16012.
    """
    # 1. Initialize Deformable Mirrors (DMs)
    # HLC uses two DMs to 'dig' the dark hole. 
    # For a flat-wavefront start, we initialize with zeros (48x48 actuators).
    dm1 = np.zeros((48, 48))
    dm2 = np.zeros((48, 48))

    # 2. Configure Parameters for HLC Band 1
    # 'excam' refers to the imaging camera.
    # 'hlc' is the Hybrid Lyot Coronagraph mode.
    # '1' corresponds to Band 1 (Center: 575nm, BW: 10%).
    # 'polaxis': 10 runs all polarizations for high fidelity.
    params = {
        'use_dm1': 1, 
        'dm1_m': dm1, 
        'use_dm2': 1, 
        'dm2_m': dm2, 
        'source_x_offset_mas': 3.0,  # 3.0 mas offset as requested
        'use_errors': 1,             # Include as-built optical errors
    }
    print("Starting HLC Band 1 Simulation...")
    
    # 3. Execute the simulation
    # 10 iterations (or samples) across the 10% bandpass.
    # Star: A0V (standard Vega-like), Magnitude 2.0.
    image, counts = cgisim.rcgisim(
        'excam', 
        'hlc', 
        '1', 
        10, 
        params, 
        star_spectrum='a0v', 
        star_vmag=2.0
    )

    print("Simulation Complete.")
    return image, counts

def analyze_results(image, counts):
    """
    Processes the raw output from CGISim to visualize the PSF
    and estimate the contrast level.
    """
    # 1. Visualization
    # Using log scale because coronagraphic contrast spans many orders of magnitude
    plt.figure(figsize=(10, 8))
    # We add a small epsilon to avoid log(0)
    plt.imshow(np.log10(image + 1e-15), cmap='magma')
    plt.colorbar(label='Log10 Intensity')
    plt.title('HLC Band 1 - Raw Simulation Output (Uncorrected)')
    plt.show()

    # 2. Statistics
    print(f"Total Photons (Counts): {np.sum(counts):.2e}")
    print(f"Peak Pixel Value: {np.max(image):.2e}")
    print(f"Mean Intensity: {np.mean(image):.2e}")

    # 3. Save to FITS for project records
    # This allows you to open the result in DS9 or Glue
    hdu = fits.PrimaryHDU(image)
    hdu.header['MODE'] = 'HLC Band 1'
    hdu.header['OFFSET'] = '3.0 mas'
    hdu.writeto('hlc_band1_result.fits', overwrite=True)
    print("Saved result to hlc_band1_result.fits")

if __name__ == "__main__":
    img, cnts = run_hlc_simulation()
    # You can now use your preferred plotting tool to see the resulting PSF/Dark Hole.
    try:
        analyze_results(img, cnts)
    except NameError:
        print("Please ensure 'img' and 'cnts' from the simulation are available.")