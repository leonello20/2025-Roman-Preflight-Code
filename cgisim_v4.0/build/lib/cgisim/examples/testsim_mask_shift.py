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
import scipy
from roman_preflight_proper import trim
import roman_preflight_proper

################################################################################
def azav( image, xc=-1, yc=-1 ):

    s = image.shape

    if xc == -1:
        xcenter = s[0] // 2
    else:
        xcenter = xc

    if yc == -1:
        ycenter = s[1] // 2
    else:
        ycenter = yc
 
    x, y = np.ogrid[0:s[0],0:s[1]]
    r = np.hypot( x-xcenter, y-ycenter ).astype( int )
    av = scipy.ndimage.mean( image, labels=r, index=np.arange(0, r.max()+1) )

    return av 

################################################################################
def azstdev( image, xc=-1, yc=-1 ):

    s = image.shape

    if xc == -1:
        xcenter = s[0] // 2
    else:
        xcenter = xc

    if yc == -1:
        ycenter = s[1] // 2
    else:
        ycenter = yc
 
    x, y = np.ogrid[0:s[0],0:s[1]]
    r = np.hypot( x-xcenter, y-ycenter ).astype( int )
    av = scipy.ndimage.standard_deviation( image, labels=r, index=np.arange(0, r.max()+1) )

    return av 

################################################################################
def testsim_mask_shift():
    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1'
    polaxis = -10       # compute images for mean X+Y polarization (don't compute at each polarization)

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm2_v.fits' )

    # save intermediate results to save file testfile.fits

    print( "Computing A0V star coronagraphic field with unshifted Lyot stop" )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    a0_sim, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, star_spectrum='a0v', star_vmag=2.0 )
    a0_sim = trim( a0_sim, 50 )

    print( "Computing A0V star coronagraphic field with Lyot stop shifted by 2 microns" )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'lyot_x_shift_m':2.0e-6}
    a0_sim_shifted, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, star_spectrum='a0v', star_vmag=2.0 )
    a0_sim_shifted = trim( a0_sim_shifted, 50 )

    # omit FPM to get unocculted PSF to compute NI

    print( "Computing A0V star unocculted PSF" )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':0}
    a0_psf, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, star_spectrum='a0v', star_vmag=2.0 )

    # compute NI for 10% bandpass

    max_psf = np.max(a0_psf)
    ni = a0_sim / max_psf
    ni_shifted = a0_sim_shifted / max_psf

    av_ni = azav( ni )
    stdev_diff = azstdev( ni-ni_shifted )
    pixscale = 0.39  # sampling in lam/D @ 575 nm
    r = np.arange( 0, av_ni.size ) * pixscale

    fig, ax = plt.subplots( nrows=1, ncols=2, figsize=(9,4) )

    im = ax[0].imshow( ni, norm=LogNorm(vmin=1e-10,vmax=1e-7), cmap=plt.get_cmap('jet'))
    ax[0].set_title('Norm Intensity (no shift)')
    ax[0].set_xlabel('pixels')
    ax[0].set_ylabel('pixels')
    fig.colorbar( im, ax=ax[0], shrink=0.5 ) 

    ax[1].plot( r, av_ni, 'b', r, stdev_diff, 'r' )
    ax[1].set_yscale('log')
    ax[1].set_xlim( 3, 9 )
    ax[1].set_ylim( 1e-10, 1e-8 )
    ax[1].set_xlabel( 'lam/D' )
    ax[1].set_ylabel( 'NI' )
    ax[1].set_title( 'Unshifted NI (blue), Std Dev diff (red)' )

    fig.suptitle( "Lyot stop shifted by 2 microns" )

    plt.show()
 
if __name__ == '__main__':
    testsim_mask_shift()
