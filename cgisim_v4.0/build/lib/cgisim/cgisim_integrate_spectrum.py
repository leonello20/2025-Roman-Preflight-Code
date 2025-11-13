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


def cgisim_integrate_spectrum( lam, photlam, lam_bandpass=0, throughput_bandpass=0, min_lam=-1, max_lam=-1 ):

    spectrum = photlam
    if type(lam_bandpass) is not int:
        n = lam_bandpass.shape[0]
        f = interp1d( lam_bandpass, throughput_bandpass, kind='cubic', fill_value='extrapolate' )
        spectrum = f( lam ) * photlam
        min_lam = np.min( lam_bandpass )
        max_lam = np.max( lam_bandpass )
    else:
        n = spectrum.shape[0]
        if min_lam == -1:
            min_lam = np.min( lam )
        if max_lam == -1:
            max_lam = np.max( lam )

    lam_i = np.linspace( min_lam, max_lam, n )
    f = interp1d( lam, spectrum, fill_value='extrapolate' )
    photlam_i = np.clip( f( lam_i ), 0, None )
    old_value = np.trapz( photlam_i, lam_i )
    diff = 1.0

    while np.abs(diff) > 0.005:
        n = n * 2
        lam_i = np.linspace( min_lam, max_lam, n )
        f = interp1d( lam, spectrum, fill_value='extrapolate' )
        photlam_i = np.clip( f( lam_i ), 0, None )
        value = np.trapz( photlam_i, lam_i )
        diff = (value - old_value) / old_value
        old_value = value

    return value

