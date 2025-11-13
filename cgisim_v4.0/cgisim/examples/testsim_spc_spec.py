#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import cgisim as cgisim
import proper
import roman_preflight_proper

def testsim_spc_spec():
    cgi_mode = 'spec'
    cor_type = 'spc-spec_band3'
    bandpass = '3'
    polaxis = -10       # compute images for mean X+Y polarization (don't compute at each polarization)

    # read in flat wavefront DM pattern

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-spec_flat_wfe_dm1_v.fits' )
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir+'/examples/spc-spec_flat_wfe_dm2_v.fits' )
    params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm2, 'use_dm2':1, 'dm2_v':dm2}

    a0_sim, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, star_spectrum='a0v', star_vmag=2.0, output_file='a0_spec_sim.fits' )
 
if __name__ == '__main__':
    testsim_spc_spec()
