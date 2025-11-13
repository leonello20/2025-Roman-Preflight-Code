#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import cgisim as cgisim

def cgisim_renormalize_spectrum( lam_star, photlam_star_mag0, mag_star, bandpass, info_dir ):

    lam_vega, photlam_vega = cgisim.cgisim_read_spectrum( 'vega', info_dir )
    lam_bandpass, throughput_bandpass = cgisim.cgisim_read_throughput( bandpass, info_dir )

    vega_flux = cgisim.cgisim_integrate_spectrum( lam_vega, photlam_vega, lam_bandpass, throughput_bandpass )

    star_flux_mag0 = cgisim.cgisim_integrate_spectrum( lam_star, photlam_star_mag0, lam_bandpass, throughput_bandpass )

    flux_ratio = 10.0**(-0.4 * mag_star)
    photlam_star = photlam_star_mag0 * vega_flux / star_flux_mag0 * flux_ratio

    return photlam_star

