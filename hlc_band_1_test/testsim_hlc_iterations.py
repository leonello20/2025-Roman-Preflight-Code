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

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_3e-8_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_3e-8_dm2_v.fits' )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    print( "Computing A0V star coronagraphic field, worst-contrast dark hole" )
    a0_worst, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0 )

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_5e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_5e-9_dm2_v.fits' )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    print( "Computing A0V star coronagraphic field, mild-contrast dark hole" )
    a0_mild, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0 )

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

    ni_flattened = a0_flattened / max_psf
    ni_worst = a0_worst / max_psf
    ni_mild = a0_mild / max_psf
    ni_best = a0_best / max_psf

    fig, ax = plt.subplots( nrows=2, ncols=2, figsize=(9,9) )

    im = ax[0,0].imshow( trim(ni_flattened,100), norm=LogNorm(vmin=1e-10,vmax=1e-5), cmap=plt.get_cmap('jet'))
    ax[0,0].set_title('Flattened WFE')
    fig.colorbar( im, ax=ax[0,0], shrink=0.5 ) 

    im = ax[0,1].imshow( trim(ni_worst,100), norm=LogNorm(vmin=1e-10,vmax=1e-5), cmap=plt.get_cmap('jet'))
    ax[0,1].set_title('Early iteration')
    fig.colorbar( im, ax=ax[0,1], shrink=0.5 ) 

    im = ax[1,0].imshow( trim(ni_mild,100), norm=LogNorm(vmin=1e-10,vmax=1e-5), cmap=plt.get_cmap('jet'))
    ax[1,0].set_title('Intermediate iteration')
    fig.colorbar( im, ax=ax[1,0], shrink=0.5 ) 

    im = ax[1,1].imshow( trim(ni_best,100), norm=LogNorm(vmin=1e-10,vmax=1e-5), cmap=plt.get_cmap('jet'))
    ax[1,1].set_title('Final iteration')
    fig.colorbar( im, ax=ax[1,1], shrink=0.5 ) 

    plt.show()

    # Save image to fits file
    
    hdu = fits.PrimaryHDU(ni_best)
    hdu.header['MODE'] = 'HLC Band 1'
    hdu.header['ITERATION'] = 'Final'
    hdu.writeto('hlc_band1_final.fits', overwrite=True)

    # Save image (all parts of subplot) to png file for quick viewing
    fig.savefig('hlc_iterations.png', dpi=300)

if __name__ == '__main__':
    testsim_hlc_iterations()