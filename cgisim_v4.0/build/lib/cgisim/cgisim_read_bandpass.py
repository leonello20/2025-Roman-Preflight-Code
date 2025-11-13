#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import cgisim as cgisim


def cgisim_read_bandpass( requested_bandpass, info_dir ):

    bandpass_data = {'lam0_um':0.0, 'nlam':0, 'minlam_um':0.0, 'maxlam_um':0.0}

    with open( info_dir+'cgisim_bandpasses.txt', 'r' ) as f:
        bandpass_found = False
        while bandpass_found == False:
            line = f.readline()
            if 'bandpass name' in line:
                bandpass_name = line.split( '=' )[0].strip()
                if bandpass_name == requested_bandpass:
                    bandpass_found = True
                    bandpass_data['lam0_um'] = float( cgisim.cgisim_readval1( f ) )
                    bandpass_data['nlam'] = int( cgisim.cgisim_readval1( f ) )
                    bandpass_data['minlam_um'], bandpass_data['maxlam_um'] = cgisim.cgisim_readval2( f )

    if not bandpass_found:
        raise Exception("Bandpass not found in bandpass list")

    return bandpass_data

