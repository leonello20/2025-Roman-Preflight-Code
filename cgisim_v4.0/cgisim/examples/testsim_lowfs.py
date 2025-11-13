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

def testsim_lowfs():

    cgi_mode = 'lowfs'
    bandpass = 'lowfs'
    polaxis = -10       # compute for mean X+Y polarization (10), but don't include incoherent contributions (minus)

    # compute reference & defocus LOWFS images for SPC spectral 

    cor_type = 'spc-spec_band3'
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-spec_ni_1e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-spec_ni_1e-9_dm2_v.fits' )

    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    spc_spec_ref, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_spec_lowfs_ref.fits' )

    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'zindex':[4], 'zval_m':[0.1e-9]}
    spc_spec_z4, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_spec_lowfs_z4.fits' )

    # compute reference  & defocus LOWFS images for SPC wide FOV

    cor_type = 'spc-wide'
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-wide_ni_3e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-wide_ni_3e-9_dm2_v.fits' )

    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    spc_wide_ref, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_wide_lowfs_ref.fits' )

    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'zindex':[4], 'zval_m':[0.1e-9]}
    spc_wide_z4, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_file='spc_wide_lowfs_z4.fits' )

    # compute reference  & defocus LOWFS images for HLC

    cor_type = 'hlc'
    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/hlc_ni_2e-9_dm2_v.fits' )

    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2}
    hlc_ref, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_file='hlc_lowfs_ref.fits' )

    params = {'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'zindex':[4], 'zval_m':[0.1e-9]}
    hlc_z4, counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params,  
        star_spectrum='a0v', star_vmag=2.0, output_file='hlc_lowfs_z4.fits' )

    # display images

    fig, ax = plt.subplots( nrows=3, ncols=2, figsize=(12,9) )

    im = ax[0,0].imshow( spc_wide_ref, cmap=plt.get_cmap('gray'))
    ax[0,0].set_title('SPC Wide Reference')
    fig.colorbar( im, ax=ax[0,0], shrink=0.5 ) 

    im = ax[0,1].imshow( spc_wide_z4-spc_wide_ref, cmap=plt.get_cmap('gray'))
    ax[0,1].set_title('SPC Wide Z4 - Reference')
    fig.colorbar( im, ax=ax[0,1], shrink=0.5 ) 

    im = ax[1,0].imshow( hlc_ref, cmap=plt.get_cmap('gray'))
    ax[1,0].set_title('HLC Reference')
    fig.colorbar( im, ax=ax[1,0], shrink=0.5 ) 

    im = ax[1,1].imshow( hlc_z4-hlc_ref, cmap=plt.get_cmap('gray'))
    ax[1,1].set_title('HLC Z4 - Reference')
    fig.colorbar( im, ax=ax[1,1], shrink=0.5 ) 

    im = ax[2,0].imshow( spc_spec_ref, cmap=plt.get_cmap('gray'))
    ax[2,0].set_title('SPC SPEC Reference')
    fig.colorbar( im, ax=ax[2,0], shrink=0.5 ) 

    im = ax[2,1].imshow( spc_spec_z4-spc_spec_ref, cmap=plt.get_cmap('gray'))
    ax[2,1].set_title('SPC SPEC Z4 - Reference')
    fig.colorbar( im, ax=ax[2,1], shrink=0.5 ) 

    plt.show()

if __name__ == '__main__':
    testsim_lowfs()
