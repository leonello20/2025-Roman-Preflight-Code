#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


def cgisim_readval2( f ):
    a = f.readline().split( '=' )
    a = ' '.join(a)
    a = a.split( ' ' )
    return float(a[0].strip()), float(a[2].strip())

