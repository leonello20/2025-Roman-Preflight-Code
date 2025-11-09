import numpy as np
import proper
from roman_preflight_proper import trim

dm1_v=np.full((48,48),40.) 
dm2_v=np.full((48,48),30.) 
f,dx = proper.prop_run('roman_preflight',0.575,512,PASSVALUE={'use_errors':1,'use_dm1':1,'dm1_v':dm1_v,'use_dm2':1,'dm2_v':dm2_v, \
        'use_fpm':0,'use_lyot_stop':0,'use_field_stop':0,'end_at_exit_pupil':1})                  

f = trim( f, 350 )
phase = np.angle( f )

proper.prop_fits_write( 'python_dm_phase.fits', phase )

