#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import numpy as np
import cgisim as cgisim


def cgisim_read_spectrum( spectrum_file, info_dir ):

    filename = info_dir + spectrum_file + '.dat'
    f = open( filename, 'r' )
    line = f.readline()
    units = line.split(',')
    lam_units = units[0]
    spectrum_units = units[1].rstrip()
    spectrum_units = spectrum_units.upper()
    lam_ang, flux = np.loadtxt( f, unpack=True )
    f.close()

    lam_um = lam_ang / 10000.0      # microns

    c = 3.0e14                      # speed of light in microns/sec
    h = 6.6266e-27                  # Plancks constant 
    photon_energy = h * c / lam_um

    if spectrum_units == 'FLAM':
        photlam = flux / photon_energy
    elif spectrum_units == 'JY':
        photlam = 3.0e-9 * flux / lam_um**2 / photon_energy
    elif spectrum_units == 'FNU':
        photlam = 3.0e14 * flux / lam_um**2 / photon_energy
    elif spectrum_units == 'PHOTLAM':
        photlam = flux
    else:
        raise Exception('ERROR - undefined or unknown spectrum flux units')

    return lam_ang, photlam

