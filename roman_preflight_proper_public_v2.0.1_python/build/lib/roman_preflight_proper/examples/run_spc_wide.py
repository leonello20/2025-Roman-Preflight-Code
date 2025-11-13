#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

import proper
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import roman_preflight_proper
from roman_preflight_proper import trim

def run_spc_wide():

    nlam = 7
    lam0 = 0.825
    bandwidth = 0.114
    minlam = lam0 * (1 - bandwidth/2)
    maxlam = lam0 * (1 + bandwidth/2)
    lam_array = np.linspace( minlam, maxlam, nlam )

    n = 512                 # output image dimension (must be power of 2)
    final_sampling = 0.1    # output sampling in lam0/D


    print( "Computing unaberrated coronagraphic field using compact model" )

    (fields, sampling) = proper.prop_run_multi('roman_preflight_compact', lam_array, n, QUIET=True, \
        PASSVALUE={'cor_type':'spc-wide','final_sampling_lam0':final_sampling} )
    images = np.abs(fields)**2
    image_noab = np.sum( images, 0 ) / nlam


    print( "Computing aberrated coronagraphic field using DM solution for flat WFE" )

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_flat_wfe_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_flat_wfe_dm2_v.fits' )
    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, n, QUIET=True, \
        PASSVALUE={'cor_type':'spc-wide', 'use_errors':1, 'polaxis':10, 
        'final_sampling_lam0':final_sampling, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} )
    images = np.abs(fields)**2
    image_flat = np.sum( images, 0 ) / nlam


    print( "Computing aberrated coronagraphic field using dark hole DM solution" )

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_ni_3e-9_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_ni_3e-9_dm2_v.fits' )
    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, n, QUIET=True, \
        PASSVALUE={'cor_type':'spc-wide', 'use_errors':1, 'polaxis':10, 
        'final_sampling_lam0':final_sampling, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} )
    images = np.abs(fields)**2
    image = np.sum( images, 0 ) / nlam


    print( "Computing 10 lam/D offset source" )

    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, n, QUIET=True, \
        PASSVALUE={'source_x_offset':10.0, 'cor_type':'spc-wide', 'use_errors':1, 'polaxis':10, 
        'final_sampling_lam0':final_sampling, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} )
    psfs = np.abs(fields)**2
    psf = np.sum( psfs, 0 ) / nlam
    max_psf = np.max(psf)

    ni_noab = image_noab / max_psf
    ni_flat = image_flat / max_psf
    ni = image / max_psf

    fig, ax  = plt.subplots( nrows=1, ncols=3, figsize=(10,3) )

    im = ax[0].imshow(ni_noab, norm=LogNorm(vmin=1e-9,vmax=1e-5), cmap=plt.get_cmap('jet'))
    circ_in = Circle((n/2,n/2),5.6/final_sampling,edgecolor='white', facecolor='none')
    circ_out = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[0].add_patch(circ_in)
    ax[0].add_patch(circ_out)
    ax[0].set_title('Unaberrated')
    fig.colorbar(im, ax=ax[0], shrink=0.5) 

    im = ax[1].imshow(ni_flat, norm=LogNorm(vmin=1e-9,vmax=1e-5), cmap=plt.get_cmap('jet'))
    circ_in = Circle((n/2,n/2),5.6/final_sampling,edgecolor='white', facecolor='none')
    circ_out = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[1].add_patch(circ_in)
    ax[1].add_patch(circ_out)
    ax[1].set_title('Flattened WFE')
    fig.colorbar(im, ax=ax[1], shrink=0.5) 

    im = ax[2].imshow(ni, norm=LogNorm(vmin=1e-9,vmax=1e-5), cmap=plt.get_cmap('jet'))
    circ_in = Circle((n/2,n/2),5.6/final_sampling,edgecolor='white', facecolor='none')
    circ_out = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[2].add_patch(circ_in)
    ax[2].add_patch(circ_out)
    ax[2].set_title('3e-9 Dark hole')
    fig.colorbar(im, ax=ax[2], shrink=0.5) 

    plt.show()
    
if __name__ == '__main__':
    run_spc_wide()
