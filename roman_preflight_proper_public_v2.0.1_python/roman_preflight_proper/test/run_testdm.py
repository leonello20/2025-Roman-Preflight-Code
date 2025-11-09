import numpy as np
import proper
import roman_preflight_proper as rp

lambda_um = 0.575

optval = {'dm':0, 'dm1_v':40+np.zeros((48,48)), 'dm_sampling_m':0.9906e-3, 
	  'dm1_xc_act':23.5, 'dm1_yc_act':23.5, 
          'dm1_xtilt_deg':9.65, 'dm1_ytilt_deg':0., 'dm1_ztilt_deg':0} 

optval['dm'] = 1
wavefront, sampling_m = proper.prop_run( 'testdm', lambda_um, 512, PASSVALUE=optval )
phase1 = np.angle(wavefront)
proper.prop_fits_write( 'python_dm1_phase.fits', phase1 )

optval['dm'] = 2
wavefront, sampling_m = proper.prop_run( 'testdm', lambda_um, 512, PASSVALUE=optval )
phase2 = np.angle(wavefront)
proper.prop_fits_write( 'python_dm2_phase.fits', phase2 )

dm_struct = rp.load_cgi_dm_files( temp_c=25., version='8.0');
dm_v = np.zeros((48,48)) + 40
stroke = rp.volts_to_stroke( dm_struct, 2, dm_v )
stroke0 = rp.backout_coupling( dm_struct, 2, stroke )
volts = rp.stroke_to_volts( dm_struct, 2, stroke0 )
proper.prop_fits_write( 'python_volts.fits', volts )

dm_v = proper.prop_fits_read( 'dm_v.fits' )
new_dm_v, neighbor_mask, limits_mask = rp.constrain_dm( dm_struct, 1, dm_v )
proper.prop_fits_write( 'python_dm1_constrained.fits', new_dm_v )

