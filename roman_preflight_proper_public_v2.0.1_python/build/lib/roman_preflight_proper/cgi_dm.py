import numpy as np
import proper
import roman_preflight_proper as rp

def cgi_dm( wavefront, dm_struct, dm, checked_v, dm_sampling_m=0.9906e-3, 
        dm_xc_act=23.5, dm_yc_act=23.5, 
        dm_xtilt_deg=0., dm_ytilt_deg=0., dm_ztilt_deg=0., 
        dm_v_quant=110.0 / 2.**16 ):

#-- CGI_DM
#-- Call: cgi_dm( wavefront, dm_struct, dm, checked_v [, dm_sampling=dm_sampling_m] 
#--                 [, dm_xc_act=dm_xc_act] [, dm_yc_act=dm_yc_act], [, dm_v_quant=dm_v_quant] 
#--                 [, dm_xtilt_deg=dm_xtilt_deg] [, dm_ytilt_deg=dm_ytilt_deg] [, dm_ztilt_deg=dm_ztilt_deg]
#--
#-- Note: Must call load_cgi_dm_files before calling this
#--
#--
#-- INPUTS:
#--   wavefront: 
#--     PROPER wavefront structure
#--   dm: 
#--     integer specifying which DM, either 1 or 2
#--   checked_v: 
#--     float, nact x nact array, voltages to send to the DM; these are assumed to meet neighbor and min/max volt rules.
#--       As voltage increases, the actuator is pulled down into the DM
#-- OPTIONAL INPUTS:
#--   dm_sampling_m:
#--     DM actuator spacing in meters
#--   dm_xc_act, dm_yc_act:
#--     Wavefront center on DM in actuators (nominally 23.5, 23.5)
#--   dm_xtilt_deg, dm_ytilt_deg, dm_ztilt_deg: 
#--     DM rotations in degrees; Z is axis perpendicular to DM surface
#--   dm_v_quant:
#--     Voltage resolution due to DAC
#--   surface_map: Set to a named variable that will contain the DM surface (not wavefront) map,
#--     excluding surface errors

    nact = checked_v.shape[0]
    dm_x_center_act = nact//2 - 0.5                 # X physical center of DM, actuators
    dm_y_center_act = nact//2 - 0.5                 # Y physical center of DM, actuators

    dm_v = np.floor(checked_v / dm_v_quant) * dm_v_quant

    tie_map = dm_struct[dm-1]['tie_map']
    wgood = (tie_map == 0).nonzero()  # non-dead actuators
    med_v = np.median(dm_v[wgood])

    # convert voltages to strokes to feed to prop_dm; note that volts_to_stroke() applies coupling

    actual_stroke_m = rp.volts_to_stroke( dm_struct, dm, dm_v )

    # assume zero median piston going into prop_dm
    # also, in prop_dm, a positive stroke pulls facesheet down, so change sign

    med_val_m = np.median(actual_stroke_m[wgood])
    prop_dm_stroke_m = -(actual_stroke_m - med_val_m)

    if ( dm == 1 ):
        surface_map = proper.prop_dm( wavefront, prop_dm_stroke_m, dm_xc_act, dm_yc_act, dm_sampling_m, 
            XTILT=dm_xtilt_deg, YTILT=dm_ytilt_deg, ZTILT=dm_ztilt_deg, 
            INFLUENCE_FUNCTION_FILE=dm_struct[dm-1]['inf_file'] )
    else:
        # DM2 map is flipped to face DM1
        surface_map = proper.prop_dm( wavefront, prop_dm_stroke_m, dm_xc_act, dm_yc_act, dm_sampling_m, 
            XTILT=dm_xtilt_deg, YTILT=dm_ytilt_deg, ZTILT=dm_ztilt_deg, 
            INFLUENCE_FUNCTION_FILE=dm_struct[dm-1]['inf_file'],
            FLIP_LR=1 )

    grid = proper.prop_get_gridsize( wavefront )

    # add static WFE (polishing errors)

    dx = dm_struct[dm-1]['static_sfe_map_dx_m']
    xoff = (dm_x_center_act - dm_xc_act) * dm_sampling_m / dx 
    yoff = (dm_y_center_act - dm_yc_act) * dm_sampling_m / dx
    mag = dx / proper.prop_get_sampling(wavefront)
    xmag = np.cos(dm_ytilt_deg/180*np.pi) * mag
    ymag = np.cos(dm_xtilt_deg/180*np.pi) * mag
    rotation = dm_ztilt_deg
    static_wfe_m = -2 * rp.shiftrot( dm_struct[dm-1]['static_sfe_map_m'], xoff, yoff, rotation, xmag, ymag )
    static_wfe_m = rp.trim(static_wfe_m, grid)

    # bias-dependent surface error ("wrinkle"); assume median voltage is bias

    dx = dm_struct[dm-1]['bias_proportional_sfe_map_dx_m']
    xoff = (dm_x_center_act - dm_xc_act) * dm_sampling_m / dx
    yoff = (dm_y_center_act - dm_yc_act) * dm_sampling_m / dx
    mag = dx / proper.prop_get_sampling(wavefront)
    xmag = np.cos(dm_ytilt_deg/180*np.pi) * mag
    ymag = np.cos(dm_xtilt_deg/180*np.pi) * mag
    rotation = dm_ztilt_deg
    bias_wfe_m = (-2 * med_v) * rp.shiftrot( dm_struct[dm-1]['bias_proportional_sfe_map_m'], xoff, yoff, rotation, xmag, ymag )
    bias_wfe_m = rp.trim(bias_wfe_m, grid)

    proper.prop_add_phase( wavefront, static_wfe_m+bias_wfe_m )

    # residual DM cylindrical deformation, after OAP alignment; assume Z4 has been removed with FCM 

    proper.prop_zernikes( wavefront, [6], [50.0e-9] )

    return surface_map

