import proper
import roman_preflight_proper as rp

def testdm( lambda_m, gridsize0, PASSVALUE={'dummy':0} ):
    gridsize = 512
    diam = 0.0463
    pupil_diam_pix = 400.
    beam_ratio = pupil_diam_pix / gridsize

    dm_struct = rp.load_cgi_dm_files( temp_c=25., version='8.0' )

    wavefront = proper.prop_begin( diam, lambda_m, gridsize, beam_ratio)
    proper.prop_circular_aperture( wavefront, diam/2 )
    rp.cgi_dm( wavefront, dm_struct, PASSVALUE['dm'], PASSVALUE['dm1_v'], PASSVALUE['dm_sampling_m'],
                    PASSVALUE['dm1_xc_act'], PASSVALUE['dm1_yc_act'], 
                    PASSVALUE['dm1_xtilt_deg'], PASSVALUE['dm1_ytilt_deg'], PASSVALUE['dm1_ztilt_deg'] )
    wavefront, sampling_m = proper.prop_end( wavefront, NOABS=True )

    return (wavefront, sampling_m)

