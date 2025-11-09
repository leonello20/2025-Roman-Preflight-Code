import numpy as np
import copy
import astropy.io.fits as pyfits
import proper
import roman_preflight_proper as rp

def load_cgi_dm_files( temp_c=26., version=rp.dm_version, dm_files_dir=rp.dm_files_dir ):
    """
    Load CGI DM-related files for DM model

    Call:  
        dm_struct = load_cgi_dm_files( temp_c, version, dm_files_dir )
    Inputs:
        temp_c:  float, DM operating temp (C)
        version: string, DM files version (e.g., '8.0')
        dm_files_dir: string, directory containing DM files
    Returns:
        dm_struct: DM parameter structure for all CGI DMs (1 & 2)
    """

    dm_names = ['dm_1236-2','dm_1236-5']
    min_temp = np.array([20,21])
    max_temp = np.array([26,26])
    min_temps = min_temp.astype(str)
    max_temps = max_temp.astype(str)

    # maps are stored looking at DM face-on

    for idm in range(2):
        dm = dm_names[idm]

        # tie map (map of tied actuator groups, dead actuators, good actuators)

        tie_map = pyfits.getdata( dm_files_dir+dm+'_tiemap_v'+version+'.fits' )

        # stroke_m_vs_volts = [nstroke_vs_volts, nact, nact]; stroke (m) vs voltage per actuator

        s1 = pyfits.getdata( dm_files_dir+dm+'_stroke_nm_vs_volts_'+min_temps[idm]+'c_v'+version+'.fits' )   # [nv,nact,nact] = stroke_nm/V
        s2, h = pyfits.getdata( dm_files_dir+dm+'_stroke_nm_vs_volts_'+max_temps[idm]+'c_v'+version+'.fits', header=True )
        nv = s2.shape[0]
        stroke_volts = np.arange(nv) * h['dvolts']
        # interpolate to specified temperature
        mult = (temp_c - min_temp[idm]) / (max_temp[idm] - min_temp[idm])
        stroke_m_vs_volts = ((s2 - s1) * mult + s1) * 1e-9      # also convert to stroke_m/V
        stroke_m_vs_volts = np.clip( stroke_m_vs_volts, 0, None )

        # DM vertical cross-coupling (actually useful for DM2 only)
        # Coupling maps are [3,3] for each actuator; element [0,0] is the fraction of stroke of actuator [y,x] 
        # that is added to the [y-1,x-1] actuator, for example. 

        c1 = pyfits.getdata( dm_files_dir+dm+'_couplings_'+min_temps[idm]+'c_v'+version+'.fits' )   # [nv,nact,nact,3,3]
        c2, h = pyfits.getdata( dm_files_dir+dm+'_couplings_'+max_temps[idm]+'c_v'+version+'.fits', header=True ) 
        nv = c1.shape[0]
        # interpolate to specified temperature
        coupling_volts = np.arange(nv) * h['dvolts']
        coupling = (c2 - c1) * mult + c1
        coupling = np.clip( coupling, 0, None )

        inf_file = dm_files_dir + dm + '_brian_inf_v' + version + '.fits'     # influence function (assume symmetric)

        # DM surface error (0V); DM2 is flipped left-right relative to DM1, so also do that to the surface errors

        static_sfe_map_m, h = pyfits.getdata( dm_files_dir+dm+'_0v_sfe_m_without_Z1-Z6_v'+version+'.fits', header=True )   
        if idm == 1:
            static_sfe_map_m = np.fliplr(static_sfe_map_m)
        static_sfe_map_dx_m = h['PIXSIZE']

        # surface "wrinkle" (meters) per bias volt map

        bias_proportional_sfe_map_m, h = pyfits.getdata( dm_files_dir+dm+'_sfe_per_Vbias_v'+version+'.fits', header=True )   
        if idm == 1:
            bias_proportional_sfe_map_m = np.fliplr(bias_proportional_sfe_map_m)
        bias_proportional_sfe_map_dx_m = h['PIXSIZE']

        temp_struct = { 'version':version, 
                        'dm_files_dir':dm_files_dir, 
                        'coupling':coupling, 
                        'coupling_volts':coupling_volts, 
                        'tie_map':tie_map, 
                        'inf_file':inf_file, 
                        'static_sfe_map_m':static_sfe_map_m, 
                        'static_sfe_map_dx_m':static_sfe_map_dx_m, 
                        'bias_proportional_sfe_map_m':bias_proportional_sfe_map_m, 
                        'bias_proportional_sfe_map_dx_m':bias_proportional_sfe_map_dx_m, 
                        'stroke_m_vs_volts':stroke_m_vs_volts, 
                        'stroke_volts':stroke_volts   }
        if idm == 0:
            a = copy.deepcopy(temp_struct)
        else:
            b = copy.deepcopy(temp_struct)
            dm_struct = [a, b]

    return dm_struct

