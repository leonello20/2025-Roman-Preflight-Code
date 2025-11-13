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

def testsim_spc_excam():
    cgi_mode = 'excam'
    cor_type = 'spc-spec_band3'
    bandpass = '3'
    polaxis = -10       # compute images for mean X+Y polarization (don't compute at each polarization)

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-spec_ni_1e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-spec_ni_1e-9_dm2_v.fits' )

    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}

    # save intermediate results to save file spctestfile.fits

    print( "Computing A0V star coronagraphic field" )
    a0_sim, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0, output_save_file='spc_testfile.fits', output_file='spc_a0_sim.fits' )

    # omit FPM to get unocculted PSF to compute NI

    print( "Computing A0V star unocculted PSF" )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':0}
    a0_psf, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_a0_psf_sim.fits' )

    ni = a0_sim / np.max(a0_psf)

    # read in intermediate save file and apply K5V spectrum with V=2
 
    print( "Computing K5V coronagraphic field from save file" ) 
    k5_sim, k5_counts = cgisim.rcgisim( star_spectrum='k5v', star_vmag=2.0, input_save_file='spc_testfile.fits', output_file='spc_k5_sim.fits' )

    # read in intermediate save file and apply K5V spectrum with V=2, add CCD with gain=1000 and 30 sec exposure

    print( "Computing K5V coronagraphic field from save file, with CCD noise" ) 
    k5_sim_ccd, k5_counts_ccd = cgisim.rcgisim( star_spectrum='k5v', star_vmag=2.0, input_save_file='spc_testfile.fits', 
        ccd={'gain':1000.0,'exptime':30.0}, output_file='spc_k5_sim_ccd.fits' )

    fig, ax = plt.subplots( nrows=2, ncols=2, figsize=(9,9) )

    im = ax[0,0].imshow( trim(ni,100), norm=LogNorm(vmin=1e-10,vmax=1e-7), cmap=plt.get_cmap('jet'))
    ax[0,0].set_title('A0V Norm Intensity')
    fig.colorbar( im, ax=ax[0,0], shrink=0.5 ) 

    im = ax[0,1].imshow( trim(a0_psf,100)**0.2, cmap='gray' )
    ax[0,1].set_title('A0V Unocculted Intensity')

    im = ax[1,0].imshow( trim(k5_sim,100)**0.5, cmap='gray' )
    ax[1,0].set_title('K5V Intensity')

    im = ax[1,1].imshow( trim(k5_sim_ccd,100), cmap='gray' )
    ax[1,1].set_title('K5V CCD Intensity')

    plt.show()

if __name__ == '__main__':
    testsim_spc_excam()
