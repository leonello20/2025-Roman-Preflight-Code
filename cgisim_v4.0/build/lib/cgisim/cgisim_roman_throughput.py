#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import numpy as np
from scipy.interpolate import interp1d
import cgisim as cgisim


def cgisim_roman_throughput( filter_name, bandpass, nd, mode, info_dir ):

    # returns system throughput at 17 wavelengths spanning bandpass (no mask reductions)

    # NOTE: reflectivity of Ni FPM included in FPM mask representation for LOWFS

    num_hrc = 7             # OTA + TCA

    if mode == 'excam':
        num_al = 2          # DMs
        num_fss99 = 13      # CGI mirrors (except DMs); assume SPC-WFOV is fss99, though it is really Al
        num_glass = 0
        num_ar_glass = 3    # FPM, air-spaced imaging doublet
    elif mode == 'lowfs':
        num_al = 2
        num_fss99 = 9
        num_glass = 0
        num_ar_glass = 4    # two air-spaced doublets
    elif mode == 'spec':
        num_al = 3          # SPAM SPC substrate coating is Al
        num_fss99 = 12
        num_glass = 0
        num_ar_glass = 6    # FPM, air-spaced imaging doublet, 3 prisms
    elif mode == 'clear':
        num_al = 0
        num_fss99 = 0
        num_glass = 0
        num_ar_glass = 0

    lam_hrc, reflectance_hrc = cgisim.cgisim_read_throughput( 'hrc_reflectivity', info_dir )
    dhrc = np.median( lam_hrc[1:] - lam_hrc[0:-1] )

    lam_fss99, reflectance_fss99 = cgisim.cgisim_read_throughput( 'fss99_reflectivity', info_dir )
    dfss99 = np.median( lam_fss99[1:] - lam_fss99[0:-1] )

    lam_al, reflectance_al = cgisim.cgisim_read_throughput( 'al_reflectivity', info_dir )
    dal = np.median( lam_al[1:] - lam_al[0:-1] )

    lam_filter, transmission_filter = cgisim.cgisim_read_throughput( 'filters/'+filter_name, info_dir )
    dfilter = np.median( lam_filter[1:] - lam_filter[0:-1] )

    if nd != 0:
        lam_nd, transmission_nd = cgisim.cgisim_read_throughput( 'filters/nd'+nd, info_dir )
        dnd = np.median( lam_nd[1:] - lam_nd[0:-1] )
    else:
        lam_nd = np.array([100.0, 5000.0, 10000.0])
        transmission_nd = np.array([1.0, 1.0, 1.0])
        dnd = lam_nd[1] - lam_nd[0]

    if mode == 'lowfs':
        ccd_name = 'locam_emccd'
    else:
        ccd_name = 'excam_emccd'
    lam_ccd, qe_ccd = cgisim.cgisim_read_throughput( ccd_name, info_dir )
    dccd = np.median( lam_ccd[1:] - lam_ccd[0:-1] )

    s = bandpass.split("lam")
    min_lam = float(s[1]) * 10      # convert nm to angstroms
    max_lam = float(s[2]) * 10
    lam_bandpass = np.linspace( min_lam, max_lam, 17 )
    throughput_bandpass = np.ones( 17 )
    dbandpass = lam_bandpass[1] - lam_bandpass[0]
    dlam = np.min( [dccd, dhrc, dal, dfss99, dfilter, dbandpass, dnd] )
    nlam = int( (max_lam - min_lam) / dlam + 1 )
    lam = np.linspace( min_lam, max_lam, nlam )

    f = interp1d( lam_hrc, reflectance_hrc, fill_value='extrapolate' )
    reflectance_hrc = np.clip( f( lam ), 0, None )

    f = interp1d( lam_fss99, reflectance_fss99, fill_value='extrapolate' )
    reflectance_fss99 = np.clip( f( lam ), 0, None )

    f = interp1d( lam_al, reflectance_al, fill_value='extrapolate' )
    reflectance_al = np.clip( f( lam ), 0, None )

    f = interp1d( lam_ccd, qe_ccd, fill_value='extrapolate' )
    qe_ccd = np.clip( f( lam ), 0, None )

    f = interp1d( lam_filter, transmission_filter, fill_value='extrapolate' )
    transmission_filter = np.clip( f( lam ), 0, None )

    f = interp1d( lam_nd, transmission_nd, fill_value='extrapolate' )
    transmission_nd = np.clip( f( lam ), 0, None )

    throughput = qe_ccd * transmission_filter * transmission_nd * reflectance_hrc**num_hrc * reflectance_al**num_al * reflectance_fss99**num_fss99 * 0.99**num_ar_glass * 0.9**num_glass

    return lam, throughput

