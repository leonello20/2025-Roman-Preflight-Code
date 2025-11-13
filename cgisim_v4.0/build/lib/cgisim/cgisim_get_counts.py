#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import cgisim as cgisim


def cgisim_get_counts( spectrum_file, bandpass_name, bandpass, nd, mag, ref_filter, mode, info_dir ):

    # WFIRST parameters
    illuminated_area = 35895.212     # cm^2

    lam, photlam_mag0 = cgisim.cgisim_read_spectrum( spectrum_file, info_dir )

    photlam = cgisim.cgisim_renormalize_spectrum( lam, photlam_mag0, mag, ref_filter, info_dir )

    lam_system, throughput_system = cgisim.cgisim_roman_throughput( bandpass_name, bandpass, nd, mode, info_dir )

    # output of cgisim_integrate_spectrum is photons/cm^2/sec over bandpass
    flux = cgisim.cgisim_integrate_spectrum( lam, photlam, lam_system, throughput_system )
    flux = flux * illuminated_area

    return flux

