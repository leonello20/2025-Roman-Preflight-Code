#   Copyright 2020 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  23 June 2020


import cgisim as cgisim
import proper
import matplotlib.pylab as plt
from roman_preflight_proper import trim
import roman_preflight_proper
import numpy as np

def testsim_zwfs():
    cgi_mode = 'excam'
    cor_type = 'zwfs'
    bandpass = '1b'
    polaxis = -10       # compute images for mean X+Y polarization (don't compute at each polarization)


    params = {'use_errors':1, 'use_pupil_lens':1}
    a0_pupil, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
             star_spectrum='a0v', star_vmag=2.0, output_file='testresult_a0v_pupil_image.fits' )
    a0_pupil = trim( a0_pupil, 400 )

    fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(3,3), constrained_layout=True )
    im = ax.imshow( a0_pupil, cmap='gray' )
    ax.set_title('ZWFS image')

    plt.show()

if __name__ == '__main__':
    testsim_zwfs()
