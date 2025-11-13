#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  8 Oct 2020


import cgisim as cgisim
import proper
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import numpy as np
from roman_preflight_proper import trim
import roman_preflight_proper

def testsim_lowfs_emccd():

    cgi_mode = 'lowfs'
    bandpass = 'lowfs'
    polaxis = -10       # compute for mean X+Y polarization (10), but don't include incoherent contributions (minus)

    # compute LOWFS image for HLC

    cor_type = 'hlc'
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm2_v.fits' )

    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    hlc_ref, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_save_file='hlc_lowfs_image.fits' )

    # compute image on EMCCD with gain=100 and 0.5 ms exposure

    a0_ccd_1, a0_ccd_counts_1 = cgisim.rcgisim( star_spectrum='a0v', star_vmag=2.0, 
        input_save_file='hlc_lowfs_image.fits', ccd={'gain':100.0,'exptime':0.0005} )

    # compute 20 frames and coadd

    a0_ccd_20 = 0.0 * a0_ccd_1
    for i in range(20):
        a0_ccd, a0_ccd_counts = cgisim.rcgisim( star_spectrum='a0v', star_vmag=2.0, 
            input_save_file='hlc_lowfs_image.fits', ccd={'gain':100.0,'exptime':0.0005} )
        a0_ccd_20 += a0_ccd

    # display images

    fig, ax = plt.subplots( nrows=1, ncols=3, figsize=(9,3) )

    im = ax[0].imshow( hlc_ref, cmap=plt.get_cmap('gray'))
    ax[0].set_title('HLC LOWFS image, no EMCCD')

    im = ax[1].imshow( a0_ccd_1, cmap=plt.get_cmap('gray'))
    ax[1].set_title('One EMCCD frame')

    im = ax[2].imshow( a0_ccd_20, cmap=plt.get_cmap('gray'))
    ax[2].set_title('20 frames co-added')

    plt.show()

if __name__ == '__main__':
    testsim_lowfs_emccd()
