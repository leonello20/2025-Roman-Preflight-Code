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
from roman_preflight_proper import trim
import roman_preflight_proper

def testsim_spc_wide():
    cgi_mode = 'excam'
    cor_type = 'spc-wide'
    bandpass = '4'

    # compute unaberrated field, no WFC.  Full prescription has a little defocus, so put in correction.

    print( "Computing unaberrated field..." )
    polaxis = 0       
    params = {'use_errors':0}
    unaberrated, unaberrated_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_wide_unaberrated.fits' )

    # compute aberrated field, no WFC

    print( "Computing aberrated field without WFC..." )
    polaxis = -10       # compute images for mean X+Y polarization (don't compute at each polarization)
    aberrated, aberrated_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis,  
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_wide_aberrated.fits' )
   
    # compute aberrated field with flattening
 
    print( "Computing aberrated field with flattening..." )
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-wide_flat_wfe_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-wide_flat_wfe_dm2_v.fits' )
    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    flattened, flattened_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_wide_flattened.fits' )

    # compute aberrated field with EFC correction

    print( "Computing aberrated field with EFC-derived corrections..." )
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-wide_ni_3e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-wide_ni_3e-9_dm2_v.fits' )
    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    corrected, corrected_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_wide_corrected.fits' )

    # omit FPM to get unocculted PSF to compute NI

    print( "Computing unocculted PSF" )
    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':0}
    psf, psf_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_wide_psf.fits' )
    max_psf = np.max(psf)

    # compute NI

    ni_noab = unaberrated / max_psf 
    ni_ab = aberrated / max_psf
    ni_flat = flattened / max_psf
    ni_corrected = corrected / max_psf

    fig, ax = plt.subplots( nrows=2, ncols=2, figsize=(9,9) )

    im = ax[0,0].imshow( trim(ni_noab,200), norm=LogNorm(vmin=1e-10,vmax=1e-4), cmap=plt.get_cmap('jet'))
    ax[0,0].set_title('Unaberrated Norm Intensity')
    fig.colorbar( im, ax=ax[0,0], shrink=0.5 ) 

    im = ax[0,1].imshow( trim(ni_ab,200), norm=LogNorm(vmin=1e-10,vmax=1e-4), cmap=plt.get_cmap('jet'))
    ax[0,1].set_title('Aberrated Norm Intensity')
    fig.colorbar( im, ax=ax[0,1], shrink=0.5 ) 

    im = ax[1,0].imshow( trim(ni_flat,200), norm=LogNorm(vmin=1e-10,vmax=1e-4), cmap=plt.get_cmap('jet'))
    ax[1,0].set_title('Flattened Norm Intensity')
    fig.colorbar( im, ax=ax[1,0], shrink=0.5 ) 

    im = ax[1,1].imshow( trim(ni_corrected,200), norm=LogNorm(vmin=1e-10,vmax=1e-4), cmap=plt.get_cmap('jet'))
    ax[1,1].set_title('EFC-corrected Norm Intensity')
    fig.colorbar( im, ax=ax[1,1], shrink=0.5 ) 

    plt.show()

if __name__ == '__main__':
    testsim_spc_wide()
