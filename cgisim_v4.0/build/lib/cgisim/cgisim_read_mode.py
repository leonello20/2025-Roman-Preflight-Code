#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import cgisim as cgisim

def cgisim_read_mode( cgi_mode, cor_type, requested_bandpass, info_dir ):

    # read sampling for mode

    mode_data = {'sampling_um':0.0, 'lamref_um':0.0, 'sampling_lamref_div_D':0.0, 'name':"", 'owa_lamref':0.0}  

    with open( info_dir+cgi_mode+'.txt', 'r' ) as f:
        line = f.readline()
        while line != '':
            if 'sampling_um' in line: 
                mode_data['sampling_um'] = float(f.readline())
            elif 'lamref_um' in line: 
                mode_data['lamref_um'] = float(f.readline())
            elif 'sampling_lamref_div_D' in line:
                mode_data['sampling_lamref_div_D'] = float(f.readline())
            line = f.readline()

    # read in coronagraph name, OWA, and allowable bandpass

    with open( info_dir+cor_type+'.txt', 'r' ) as f:
        line = f.readline()
        while line != '':
            if 'name' in line:
                mode_data['name'] = f.readline().strip()
            elif 'owa' in line:
                mode_data['owa_lamref'] = float(f.readline())
            elif 'bandpasses' in line:
                valid_bandpass = f.readlines()
                valid_bandpass = [x.strip() for x in valid_bandpass]
            line = f.readline()

    if (not requested_bandpass in valid_bandpass) and (not '*' in valid_bandpass): 
        raise Exception("ERROR: Requested bandpass %s not allowed for coronagraph type." %(requested_bandpass))

    bandpass_data = cgisim.cgisim_read_bandpass( requested_bandpass, info_dir )

    return mode_data, bandpass_data

