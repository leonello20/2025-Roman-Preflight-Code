#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import numpy as np
import cgisim as cgisim

def cgisim_read_throughput( throughput_file, info_dir ):

    filename = info_dir + throughput_file + '.dat'
    f = open( filename, 'r' )
    lam_ang, throughput = np.loadtxt( f, unpack=True )
    f.close()

    return lam_ang, throughput

