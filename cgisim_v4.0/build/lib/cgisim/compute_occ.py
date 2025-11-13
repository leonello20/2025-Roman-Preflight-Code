#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import numpy as np
import proper
from thin_film_filter_2 import thin_film_filter_2
from scipy.interpolate import interp2d

# materials: pmgi, ni, fused_silica, ti
   
def compute_occ( thickness_map_files, materials, wavelength_m, incident_angle_deg=6.69 ):
    
    incident_angle_radians = incident_angle_deg / 180.0 * np.pi
    nlam = len( wavelength_m )
    nmaterials = len( materials )
    nlayers = 2 + nmaterials    # first two layers are implied vacuum, last layer is infinitely thick
    nthick = len( thickness_map_files )

    if nthick != nlayers-3:
        raise Exception('Number of layers and number of thickness files not compatible')

    t = proper.prop_fits_read( thickness_map_files[0] )
    nx = t.shape[1]
    ny = t.shape[0]
    thick = np.zeros( (nlayers, ny, nx), dtype=np.float64 )
    for ilayer in range(2, nlayers-1):
        thick[ilayer,:,:] = proper.prop_fits_read( thickness_map_files[ilayer-2] )
    thick[1,:,:] = 1500e-9 - np.sum( thick, axis=0 )     # vacuum thickness

    thinfilm_dict = {};
    thinfilm_dict['pmgi_xs_n'] = np.array([.450e-6,  .550e-6, .650e-6, .750e-6, .880e-6, .910e-6]);
    thinfilm_dict['pmgi_ys_n'] = np.array([ 1.5547,  1.5434, 1.5374, 1.5339, 1.5316, 1.5300]);
    thinfilm_dict['pmgi_xs_k'] = np.array([.450e-6,  .550e-6, .650e-6, .750e-6, .880e-6, .910e-6]);
    thinfilm_dict['pmgi_ys_k'] = np.array([ 0.0,    0.0,  0.0,   0.0,    0.0, 0.0]);
    
    thinfilm_dict['ti_xs_n'] = np.array([.397e-6, .413e-6, .431e-6, .451e-6, .471e-6, .496e-6, .521e-6, .549e-6, .582e-6, .617e-6, .659e-6, .704e-6, 
        .756e-6, .821e-6, .892e-6, .984e-6, 1.088e-6, 1.216e-6]);
    thinfilm_dict['ti_ys_n'] = np.array([2.08,    2.14,    2.21,    2.27,    2.3,     2.36,    2.44,    2.54,    2.6,     2.67,    2.76,    2.86,    3.0, 
        3.21,    3.29,    3.35,    3.5,      3.62  ]); 
    thinfilm_dict['ti_xs_k'] = np.array([.397e-6, .413e-6, .431e-6, .451e-6, .471e-6, .496e-6, .521e-6, .549e-6, .582e-6, .617e-6, .659e-6, .704e-6, 
        .756e-6, .821e-6, .892e-6, .984e-6, 1.088e-6, 1.216e-6 ]);
    thinfilm_dict['ti_ys_k'] = np.array([2.95,    2.98,    3.01,    3.04,    3.1,     3.19,    3.2,     3.43,    3.58,    3.74,    3.84,    3.96,    4.01,
        4.01,    3.96,    3.97,    4.02,     4.15]);
    
    thinfilm_dict['ni_xs_n'] = np.array([.4e-6,  .44e-6,   .48e-6,  .5e-6, .51e-6, .52e-6, .53e-6, .54e-6, .55e-6, .56e-6, .57e-6, .58e-6, .59e-6, .6e-6, 
    .64e-6, .68e-6, .72e-6, .76e-6, .8e-6,  .88e-6, .90e-6, .92e-6, .96e-6,   1.0e-6,   1.04e-6]);
    thinfilm_dict['ni_ys_n'] = np.array([ 1.61,    1.62,   1.66163, 1.678, 1.697,  1.716,  1.735,  1.754,  1.773,  1.792,  1.811,  1.830,  1.849, 1.869, 
        1.98941, 2.11158, 2.25, 2.38625, 2.48, 2.63839, 2.6675, 2.69278,  2.74947, 2.80976, 2.85933]); 
    thinfilm_dict['ni_xs_k'] = np.array([.4e-6,  .44e-6,   .48e-6,  .5e-6, .51e-6, .52e-6, .53e-6, .54e-6, .55e-6, .56e-6, .57e-6, .58e-6, .59e-6, .6e-6, 
     .64e-6,.68e-6,.709e-6,.729e-6,.751e-6,.775e-6,.8e-6, .88e-6,  .90e-6, .92e-6,   .96e-6,    1.0e-6,  1.04e-6]);
    thinfilm_dict['ni_ys_k'] = np.array([ 2.36,  2.59353,  2.82958, 2.966, 3.023,  3.080,  3.137,  3.194,  3.251,  3.308,  3.365,  3.423,  3.480,  3.537, 
    3.75882,3.95737, 4.09,  4.18,    4.25, 4.31,   4.38, 4.61452, 4.67375, 4.73667,  4.86895, 4.99537,  5.12178]);
    
    thinfilm_dict['fused_silica_xs_n'] = np.array([.4e-6, .5e-6, .51e-6, .52e-6, .53e-6, .54e-6, .55e-6, .56e-6, .57e-6, .58e-6, .59e-6, .6e-6, .72e-6,
     .76e-6,   .8e-6,  .88e-6,  .90e-6,  1.04e-6]);
    thinfilm_dict['fused_silica_ys_n'] = np.array([ 1.47012, 1.462, 1.462,  1.461,  1.461,  1.460,  1.460,  1.460,  1.459,  1.459,  1.458,  1.458, 
        1.45485, 1.45404, 1.45332, 1.45204, 1.45175, 1.44992]);
    thinfilm_dict['fused_silica_xs_k'] = np.array([.4e-6,    .6e-6,    .72e-6, .76e-6,   .8e-6,  .88e-6,  .90e-6,  1.04e-6]);
    thinfilm_dict['fused_silica_ys_k'] = np.array([   0,       0,        0,      0,       0,       0,       0 ,      0]);

    n_material = np.zeros( (nlayers), dtype=np.complex128 )
    n_material[0:2] = 1.0 + 0.0j 

    occ_ref = np.zeros( (nlam,ny,nx), dtype=np.complex128 )
    occ_trans = np.zeros( (nlam,ny,nx), dtype=np.complex128 )

    thick0 = np.zeros( thick.shape, dtype=thick.dtype )
    thick0[:,:,:] = thick.copy()

    for ilam in range(nlam):
        lam_m = wavelength_m[ilam]

        for ilayer in range(2, nlayers):
            xs_n = thinfilm_dict[materials[ilayer-2]+'_xs_n']
            ys_n = thinfilm_dict[materials[ilayer-2]+'_ys_n']
            nval = np.interp(lam_m, xs_n, ys_n, left=-1e123, right=-1e123)
            xs_k = thinfilm_dict[materials[ilayer-2]+'_xs_k']
            ys_k = thinfilm_dict[materials[ilayer-2]+'_ys_k']
            kval = np.interp(lam_m, xs_k, ys_k, left=-1e123, right=-1e123)
            n_material[ilayer] = nval - 1.0j * kval

        xin = (np.linspace( 0.0, nx-1, nx ) - (nx//2))
        yin = (np.linspace( 0.0, ny-1, ny ) - (ny//2))
        xout = (np.linspace( 0.0, nx-1, nx ) - (nx//2)) / (wavelength_m[nlam//2] / lam_m) 
        yout = (np.linspace( 0.0, ny-1, ny ) - (ny//2)) / (wavelength_m[nlam//2] / lam_m)

        #for ithick in range(nlayers):
        #    f = interp2d( xin, yin, thick0[ithick,:,:], kind='linear', bounds_error=False, fill_value=thick0[ithick,0,0] ) 
        #    thick[ithick,:,:] = f( xout, yout )

        R, T, rr, tt = thin_film_filter_2( n_material, thick[1:nlayers-1,:,:], incident_angle_radians, lam_m ) 
        occ_ref[ilam,:,:] = rr
        occ_trans[ilam,:,:] = tt
    
    return occ_ref

