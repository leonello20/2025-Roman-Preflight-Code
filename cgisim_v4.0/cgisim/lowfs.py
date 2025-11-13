#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019
#
#  Modified 14 Oct 2019 - JEK - set output_opd to True in call to thin_film_filter_2
#  Modified Sept 2020 - JEK - updated to Phase C masks
#  Modified Oct 2020 - JEK - replaced PMGI index of refraction table with equation

import numpy as np
import proper
import roman_preflight_proper
import cgisim as cgisim
from .mft2 import mft2

#########################################################################################
def trim( input_image, output_dim ):

    input_dim = input_image.shape[1]

    if input_dim == output_dim:
        return input_image
    elif output_dim < input_dim:
        x1 = input_dim // 2 - output_dim // 2
        x2 = x1 + output_dim
        output_image = input_image[x1:x2,x1:x2].copy()
    else:
        output_image = np.zeros((output_dim,output_dim), dtype=input_image.dtype)
        x1 = output_dim // 2 - input_dim // 2
        x2 = x1 + input_dim
        output_image[x1:x2,x1:x2] = input_image

    return output_image

#########################################################################################
def thin_film_filter_2( n, d, theta, lam, tetm=1, output_opd=False ):
  #function [R, T, rr, tt] = thin_film_filter_2(n,d,theta,lam,tetm)
  # [R, T, rr, tt] = thin_film_filter_2(n,d,theta,lam,[tetm])
  # n = index of refraction for each layer. 
  #     n(1) = index of incident medium
  #     n(N) = index of transmission medium
  #     then length(n) must be >= 2
  # d = thickness of each layer, not counting incident medium or transmission
  #     medium. length(d) = length(n)-2
  # theta = angle of incidence [rad], scalar only
  # lam = wavelength. units of lam must be same as d, scalar only
  # tetm: 0 => TE (default is s), 1 => TM (1 is p)
  #
  # outputs:
  # R = normalized reflected intensity coefficient
  # T =        "   transmitted     "
  # rr = complex field reflection coefficient
  # tt =      "        transmission "
  #
  # Ref:
  # H. Angus Macleod, "Thin-Film Optical Filters", Fourth Edition, 2010, p.
  # 89-92, equation 3.16
  #####
  # Dwight added:
  #  d can be a list of arrays of the same shape
  #  output_opd=True makes it OPD before default was False
  ################

  N = len(n);
  if len(d) != N-2:
    errtx = 'n and d mismatch'
    raise BaseException(errtx)
  if type(d[0]) == type(1.0):
    d2 = np.zeros([N], np.float64);
    d2[1:N-1] = d
  else:
    sh1 = d[0].shape
    for ii in range(1,N-2):
      if not np.alltrue(d[ii].shape == sh1):
        errtx = 'thin_film_filter_2() not np alltrue(d[ii].shape == sh1)'
        raise BaseException(errtx)
    d2 = [0 for ii in range(N)]
    d2[0] = d2[-1] = np.zeros(sh1, np.float64)
    d2[1:N-1] = d
  d = d2

  kx = 2*np.pi*n[0]*np.sin(theta)/lam;
  kz = -np.sqrt( (2*np.pi*n/lam)**2 - kx**2 ); # sign agrees with measurement convention

  if tetm == 1:
     kzz = kz/(n**2);
  else:
     kzz = kz;

  if type(d[0]) == type(1.0):
    eep = np.exp(-1j*kz*d);
    eem = np.exp(1j*kz*d);
  else:
    eep = [np.exp(-1j*kz[ii]*d[ii]) for ii in range(N-1)]
    eem = [np.exp(1j*kz[ii]*d[ii]) for ii in range(N-1)]

  tin = 0.5*(kzz[0:N-1] + kzz[1:N])/kzz[0:N-1];
  ri  = (kzz[0:N-1] - kzz[1:N])/(kzz[0:N-1] + kzz[1:N]);

  Axx = Ayy = 1.0
  Axy = Ayx = 0.0
  for ii in range(0, N-1):
    v1 = tin[ii]
    xx = v1*eep[ii]; 
    xy = v1*ri[ii]*eep[ii];
    yx = v1*ri[ii]*eem[ii]; 
    yy = v1*eem[ii];
    # matrix mulitiply:
    Axx2 = Axx * xx + Axy * yx
    Axy2 = Axx * xy + Axy * yy
    Ayx2 = Ayx * xx + Ayy * yx
    Ayy2 = Ayx * xy + Ayy * yy
    Axx = Axx2
    Axy = Axy2
    Ayx = Ayx2
    Ayy = Ayy2

  rr = Ayx/Axx
  tt = 1.0/Axx

  R = abs(rr)**2;
  if tetm == 1:
    Pn = ((kz[N-1]/(n[N-1]**2))/(kz[0]/(n[0]**2))).real;
  else:
    Pn = ((kz[N-1]/kz[0])).real;
  T = Pn*abs(tt)**2;
  tt= np.sqrt(Pn)*tt;

  if output_opd:
    phs = np.arctan2(np.imag(tt),np.real(tt));
    phs = -phs;
    e1 = np.sqrt(np.real(tt*np.conjugate(tt)));
    tt = e1*(np.cos(phs)+1j*np.sin(phs))
    phs = -np.angle( rr )
    rr = np.abs(rr) * np.exp(1j * phs)

  return R, T, rr, tt

#########################################################################################
def compute_occ( thickness_map_files, materials, lam_m, incident_angle_deg=5.5 ):
    
    incident_angle_radians = incident_angle_deg / 180.0 * np.pi
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
    # original:
    #thinfilm_dict['pmgi_xs_n'] = np.array([.450e-6,  .550e-6, .650e-6, .750e-6, .880e-6, .910e-6]);
    #thinfilm_dict['pmgi_ys_n'] = np.array([ 1.5547,  1.5434, 1.5374, 1.5339, 1.5316, 1.5300]);
    #thinfilm_dict['pmgi_xs_k'] = np.array([.450e-6,  .550e-6, .650e-6, .750e-6, .880e-6, .910e-6]);
    #thinfilm_dict['pmgi_ys_k'] = np.array([ 0.0,    0.0,  0.0,   0.0,    0.0, 0.0]);
    # newer:
    #thinfilm_dict['pmgi_xs_n'] = np.array([.4499e-6,  .550e-6, .650e-6, .750e-6, .880e-6, 1.000001e-6]);
    #thinfilm_dict['pmgi_ys_n'] = np.array([ 1.5547,  1.5434, 1.5374, 1.5339, 1.5316, 1.5310]);
    #thinfilm_dict['pmgi_xs_k'] = np.array([.4499e-6,  .550e-6, .650e-6, .750e-6, .880e-6, 1.000001e-6]);
    #thinfilm_dict['pmgi_ys_k'] = np.array([ 0.0,    0.0,  0.0,   0.0,    0.0, 0.0]);
    
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

    thick0 = np.zeros( thick.shape, dtype=thick.dtype )
    thick0[:,:,:] = thick.copy()

    for ilayer in range(2, nlayers):
        if materials[ilayer-2] == 'pmgi':
            lam_um = lam_m * 1e6
            nval = 1.524 + 5.176e-3 / lam_um**2 + 2.105e-4 / lam_um**4
            kval = 0 
        else:
            xs_n = thinfilm_dict[materials[ilayer-2]+'_xs_n']
            ys_n = thinfilm_dict[materials[ilayer-2]+'_ys_n']
            nval = np.interp(lam_m, xs_n, ys_n, left=-1e123, right=-1e123)
            xs_k = thinfilm_dict[materials[ilayer-2]+'_xs_k']
            ys_k = thinfilm_dict[materials[ilayer-2]+'_ys_k']
            kval = np.interp(lam_m, xs_k, ys_k, left=-1e123, right=-1e123)
        n_material[ilayer] = nval - 1.0j * kval

    R, T, occ_ref, occ_trans = thin_film_filter_2( n_material, thick[1:nlayers-1,:,:], incident_angle_radians, lam_m, output_opd=True ) 
    
    return occ_ref

#########################################################################################
def lowfs( cor_type, fields_i, lam_i, grid_dim_out0 ):   
    # propagate pupil E-field in "fields_i" to FPM, apply LOWFS spot, and 
    # propagate to pupil image on LOWFS detector

    lowfs_dir = cgisim.lib_dir + '/lowfs_data/'

    nlam = len(lam_i)

    lowfs_pupil_diam = 38.0

    if cor_type == 'hlc':
        # FPM reflection arrays are same size in the datacube, "fpm_sampling_lam0" lam0/D sampling 
        pupil_diam_pix = 309.0
        oversampling_factor = 9     # oversampling of detector pixels, must be odd
        npup = 415      # pre-magnified pupil field grid dimensions, must be odd
        fpm_sampling0 = pupil_diam_pix / 4096.0 # fpm_lam0/D units
        fpm_lam0 = 0.575
        thickness_map_files = [ lowfs_dir + 'hlc20190210/' + s for s in [ 'run461_theta6.69imthicks_PMGIfnum32.5676970504_lam5.75e-07_.fits', 
              'run461_theta6.69imthicks_nifnum32.5676970504_lam5.75e-07_.fits', 'run461_theta6.69imthicks_tifnum32.5676970504_lam5.75e-07_.fits' ] ]
        materials = [ 'pmgi', 'ni', 'ti', 'fused_silica' ]
    else:  # SPC (wide, spec)
        pupil_diam_pix = 1000.0
        oversampling_factor = 27    # for 1000 pix pupil to 38 pix, must be odd
        npup = 1601      # must be odd
        # spots are stored as phase in radians @ spot_lam0
        if cor_type == 'spc-wide':
            fpm_filename = lowfs_dir + 'spc20200610_wfov/FPM_SPC-20200610_0.1_lamc_div_D.fits'   # annulus
            fpm_lam0 = 0.825    # reference wavelength for annulus dimensions
            fpm_sampling0 = 0.1     # FPM sampling in fpm_lam0/D
            spot_filename = lowfs_dir + 'spc20200610_wfov/spot_0.01lamdivD.fits' # LOWFS spot 
            spot_sampling0 = 0.01   # sampling of LOWFS spot in spot_lam0/D
        else:
            fpm_filename = lowfs_dir + 'spc20200617_spec/fpm_0.05lamD.fits'   # bow tie
            fpm_sampling0 = 0.05     # FPM sampling in fpm_lam0/D
            if cor_type == 'spc-spec_band3':
                fpm_lam0 = 0.730    # reference wavelength for bow tie dimensions
            else:
                fpm_lam0 = 0.660
            spot_filename = lowfs_dir + 'spc20200617_spec/spot_0.02lamdivD.fits' # LOWFS spot 
            spot_sampling0 = 0.02   # sampling of LOWFS spot in spot_lam0/D
        fpm = proper.prop_fits_read( fpm_filename )
        nfpm = fpm.shape[0]
        spot_phase = proper.prop_fits_read( spot_filename )  
        nspot = spot_phase.shape[0]
        spotmask = (spot_phase != spot_phase[0,0]).astype(int)  # 1 where spot exists
        spot_lam0 = 0.575       # reference wavelength for LOWFS spot size & phase

    grid_dim = int(grid_dim_out0) * int(oversampling_factor)
    mag = float(lowfs_pupil_diam) * oversampling_factor / pupil_diam_pix

    lowfs_images = np.zeros( (nlam, grid_dim, grid_dim), dtype=np.float64 )

    for ilam in range(nlam):
        lam = lam_i[ilam]
        fpm_sampling_lam = fpm_sampling0 * fpm_lam0 / lam 
        field = trim( fields_i[ilam,:,:], npup )

        if cor_type == 'hlc':
            # compute reflected E-field modulation of FPM
            fpm = compute_occ( thickness_map_files, materials, lam*1e-6 )   # compute complex-valued reflective modulation
            fpm_mask = (np.abs(fpm) > 0.5*np.amax(np.abs(fpm))).astype(int)     # zero outside of FPM
            nfpm = fpm.shape[0]
            fpm_field = mft2( field, fpm_sampling_lam, pupil_diam_pix, nfpm, -1 )   # to FPM
            ar_amp_reflectivity = np.sqrt(0.001)
            fpm_field *= fpm_mask * (fpm - ar_amp_reflectivity)
            lowfs_field = field * ar_amp_reflectivity + mft2( fpm_field, fpm_sampling_lam, pupil_diam_pix, npup, 1 )   # to pupil (LOWFS detector)
        else:
            fpm_field = mft2( field, fpm_sampling_lam, pupil_diam_pix, nfpm, -1 )  # to FPM
            # multiply by bow tie transmission
            fpm_field *= fpm
            # MFT back to pupil and subtract light that would go through bow tie
            lowfs_field = field - mft2( fpm_field, fpm_sampling_lam, pupil_diam_pix, npup, 1 )  # to pupil and subtract from original field
            # MFT remaining light from pupil to FPM and apply spot
            spot_sampling_lam = spot_sampling0 * spot_lam0 / lam 
            spot_field = mft2( lowfs_field, spot_sampling_lam, pupil_diam_pix, nspot, -1 ) # to LOWFS spot 
            spot_field *= (np.exp(1.0j * (spot_lam0 / lam) * spot_phase) - spotmask) * spotmask 
            lowfs_field += mft2( spot_field, spot_sampling_lam, pupil_diam_pix, npup, 1 )   # to pupil (LOWFS detector)

        lowfs_images[ilam,:,:] = np.abs( proper.prop_magnify( lowfs_field, mag, grid_dim, AMP_CONSERVE=True, QUICK=True ) )**2

    return lowfs_images, oversampling_factor

