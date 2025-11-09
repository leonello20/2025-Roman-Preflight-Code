#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

import proper
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import roman_preflight_proper
from roman_preflight_proper import trim

def run_flatten():

    nlam = 5
    lam0 = 0.825
    bandwidth = 0.1
    minlam = lam0 * (1 - bandwidth/2)
    maxlam = lam0 * (1 + bandwidth/2)
    lam_array = np.linspace( minlam, maxlam, nlam )

    n = 512
    final_sampling = 0.1

    print( "Computing field before flattening, 40V DM bias" )

    optval = {'cor_type':'spc-wide', 'final_sampling_lam0':final_sampling, 'use_errors':1, 'polaxis':10, \
              'use_dm1':1, 'dm1_v':np.full((48,48),40.), 'use_dm2':1, 'dm2_v':np.full((48,48),40.), 'source_x_offset':0.}
    fields, sampling = proper.prop_run_multi( 'roman_preflight', lam_array, n, QUIET=True, \
        PASSVALUE=optval )
    images = np.abs(fields)**2
    image_before = np.sum(images,0) / nlam

    print( "Computing field after flattening" )

    optval['dm1_v'] = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_flat_wfe_dm1_v.fits' )
    optval['dm2_v'] = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/spc-wide_flat_wfe_dm2_v.fits' )
    fields, sampling = proper.prop_run_multi( 'roman_preflight', lam_array, n, QUIET=True, PASSVALUE=optval )
    images = np.abs(fields)**2
    image_after = np.sum(images,0) / nlam

    print( "Computing 9 lam/D offset source to compute NI..." )
    optval['source_x_offset'] = 9
    fields, sampling = proper.prop_run_multi( 'roman_preflight', lam_array, n, QUIET=True, PASSVALUE=optval )
    images = np.abs(fields)**2
    psf = np.sum(images,0) / nlam
    max_psf = np.max(psf)

    ni_before = image_before / max_psf
    ni_after = image_after / max_psf

    fig, ax = plt.subplots( nrows=1, ncols=2, figsize=(7,3) )

    im = ax[0].imshow(ni_before, norm=LogNorm(vmin=1e-7,vmax=1e-2), cmap=plt.get_cmap('jet'))
    circ = Circle((n/2,n/2),5.8/final_sampling,edgecolor='white', facecolor='none')
    ax[0].add_patch(circ)
    circ = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[0].add_patch(circ)
    ax[0].set_title('Before flattening')
    fig.colorbar( im, ax=ax[0], shrink=0.5 ) 

    im = ax[1].imshow(ni_after, norm=LogNorm(vmin=1e-7,vmax=1e-2), cmap=plt.get_cmap('jet'))
    circ = Circle((n/2,n/2),5.8/final_sampling,edgecolor='white', facecolor='none')
    ax[1].add_patch(circ)
    circ = Circle((n/2,n/2),20.4/final_sampling,edgecolor='white', facecolor='none')
    ax[1].add_patch(circ)
    ax[1].set_title('After flattening')
    fig.colorbar( im, ax=ax[1], shrink=0.5 ) 

    plt.show()
    
if __name__ == '__main__':
    run_flatten()
