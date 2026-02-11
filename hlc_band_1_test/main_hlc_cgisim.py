#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import cgisim as cgisim
import proper
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import numpy as np
from astropy.io import fits
from roman_preflight_proper import trim
import roman_preflight_proper

def testsim_hlc_iterations():
    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1'
    polaxis = -10       # compute images for mean X+Y polarization (don't compute at each polarization)

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_flat_wfe_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_flat_wfe_dm2_v.fits' )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    print( "Computing A0V star coronagraphic field, flattened WFE" )
    a0_flattened, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0 )

    #dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_3e-8_dm1_v.fits' )
    #dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_3e-8_dm2_v.fits' )
    #params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    #print( "Computing A0V star coronagraphic field, worst-contrast dark hole" )
    #a0_worst, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
    #    star_spectrum='a0v', star_vmag=2.0 )

    #dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_5e-9_dm1_v.fits' )
    #dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_5e-9_dm2_v.fits' )
    #params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    #print( "Computing A0V star coronagraphic field, mild-contrast dark hole" )
    #a0_mild, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
    #    star_spectrum='a0v', star_vmag=2.0 )

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm2_v.fits' )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    print( "Computing A0V star coronagraphic field, best-contrast dark hole" )
    a0_best, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0 )

    # omit FPM to get unocculted PSF to compute NI

    print( "Computing A0V star unocculted PSF" )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':0}
    a0_psf, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0 )
    max_psf = np.max(a0_psf)

    #ni_worst = a0_worst / max_psf
    #ni_mild = a0_mild / max_psf
    ni_best = a0_best / max_psf

    #fig, ax = plt.subplots( nrows=2, ncols=2, figsize=(9,9) )

    #im = ax[0,0].imshow( trim(ni_flattened,100), norm=LogNorm(vmin=1e-10,vmax=1e-5), cmap=plt.get_cmap('jet'))
    #ax[0,0].set_title('Flattened WFE')
    #fig.colorbar( im, ax=ax[0,0], shrink=0.5 ) 

    #im = ax[0,1].imshow( trim(ni_worst,100), norm=LogNorm(vmin=1e-10,vmax=1e-5), cmap=plt.get_cmap('jet'))
    #ax[0,1].set_title('Early iteration')
    #fig.colorbar( im, ax=ax[0,1], shrink=0.5 ) 

    #im = ax[1,0].imshow( trim(ni_mild,100), norm=LogNorm(vmin=1e-10,vmax=1e-5), cmap=plt.get_cmap('jet'))
    #ax[1,0].set_title('Intermediate iteration')
    #fig.colorbar( im, ax=ax[1,0], shrink=0.5 ) 

    #im = ax[1,1].imshow( trim(ni_best,100), norm=LogNorm(vmin=1e-10,vmax=1e-5), cmap=plt.get_cmap('jet'))
    #ax[1,1].set_title('Final iteration')
    #fig.colorbar( im, ax=ax[1,1], shrink=0.5 ) 

    #plt.show()

    # Save image to fits file
    
    hdu = fits.PrimaryHDU(ni_best)
    hdu.header['MODE'] = 'HLC Band 1'
    hdu.header['ITERATION'] = 'Final'
    hdu.writeto('hlc_band1_results.fits', overwrite=True)

    # Save image (all parts of subplot) to png file for quick viewing
    #fig.savefig('hlc_iterations.png', dpi=300)

if __name__ == '__main__':
    testsim_hlc_iterations()




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