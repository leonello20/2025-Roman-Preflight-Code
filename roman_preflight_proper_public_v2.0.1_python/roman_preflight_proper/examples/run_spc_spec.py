#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

import proper
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
import roman_preflight_proper
from roman_preflight_proper import trim

def run_spc_spec():

    nlam = 7
    bandwidth = 0.17
    lam0 = 0.73
    minlam = lam0 * (1 - bandwidth/2)
    maxlam = lam0 * (1 + bandwidth/2)
    lam_array = np.linspace( minlam, maxlam, nlam )

    npsf = 256              # output image dimension (must be power of 2)
    final_sampling = 0.1    # output sampling in lam0/D

    polaxis_array = [-2, -1, 1, 2]
    npol = 4
    cases = ['2e-8', '4e-9', '1e-9']
    ncase = 3
    images = np.zeros((ncase,npsf,npsf))

    # compute dark hole for each mean contrast solution

    for icase in range(ncase):
        print( "Computing "+cases[icase]+" contrast image" )
        rootname = 'spc-spec_ni_' + cases[icase]
        dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm1_v.fits' )
        dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/'+rootname+'_dm2_v.fits' )
        optval = {'cor_type':'spc-spec', 'use_errors':1, 'polaxis':10, 'final_sampling_lam0':final_sampling, \
                  'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2} 
        # compute broadband image for each polarization component
        image = 0
        for polaxis in polaxis_array:
            print("  Computing polarization component "+str(polaxis))
            optval['polaxis'] = polaxis
            # compute all wavelengths at once for this polarization component
            (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, npsf, QUIET=True, PASSVALUE=optval)
            image += (np.sum(np.abs(fields)**2,0) / nlam)
        images[icase,:,:] = image / nlam

    # compute unocculted PSF for mean polarization (polaxis=10) for last contrast case

    print( "Computing unocculted PSF for mean polarization" )

    optval = {'cor_type':'spc-spec', 'use_errors':1, 'polaxis':10, 'final_sampling_lam0':final_sampling, \
              'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':0} 
    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam_array, npsf, QUIET=True, PASSVALUE=optval)
    psfs = np.abs(fields)**2
    psf = np.sum( psfs, 0 ) / nlam

    ni = images / np.max(psf)

    fig, ax  = plt.subplots( nrows=1, ncols=3, figsize=(9,4) )

    for i in range(ncase):
        im = ax[i].imshow(ni[i,:,:], norm=LogNorm(vmin=1e-10,vmax=1e-7), cmap=plt.get_cmap('jet'))
        circ_in = Circle((npsf/2,npsf/2),3/final_sampling,edgecolor='white', facecolor='none')
        circ_out = Circle((npsf/2,npsf/2),9/final_sampling,edgecolor='white', facecolor='none')
        ax[i].add_patch(circ_in)
        ax[i].add_patch(circ_out)
        ax[i].set_title(cases[i])
        fig.colorbar(im, ax=ax[i], shrink=0.5) 

    plt.show()
    
if __name__ == '__main__':
    run_spc_spec()
