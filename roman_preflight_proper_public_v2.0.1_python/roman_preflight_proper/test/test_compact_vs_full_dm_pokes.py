import numpy as np
import proper
import matplotlib.pylab as plt
from roman_preflight_proper import ffts

dm1_v=np.full((48,48),40.) 
dm1_v[24,15]=60. 
dm2_v=np.full((48,48),40.) 
dm2_v[12,32]=20.
dm2_v[34,24]=20.
dm_v=np.full((48,48),40.) 
a0,dx = proper.prop_run('roman_preflight',0.575,512,PASSVALUE={'use_errors':0,'use_dm1':1,'dm1_v':dm_v,'use_dm2':1,'dm2_v':dm_v, \
        'use_fpm':0,'use_lyot_stop':0,'use_field_stop':0})                  
a,dx = proper.prop_run('roman_preflight',0.575,512,PASSVALUE={'use_errors':0,'use_dm1':1,'dm1_v':dm1_v,'use_dm2':1,'dm2_v':dm2_v, \
        'use_fpm':0,'use_lyot_stop':0,'use_field_stop':0})                  

dm1_m=np.zeros((48,48))
dm1_m[24,15]=66e-9*1.2
dm2_m=np.zeros((48,48))
dm2_m[12,32]=-66e-9*1.2
dm2_m[34,24]=-66e-9*1.2
c0,dx = proper.prop_run('roman_preflight_compact',0.575,512,PASSVALUE={'use_fpm':0,'use_lyot_stop':0})                  
c,dx = proper.prop_run('roman_preflight_compact',0.575,512,PASSVALUE={'use_dm1':1,'dm1_m':dm1_m,'use_dm2':1,'dm2_m':dm2_m,'use_fpm':0,'use_lyot_stop':0})                  

fa = ffts(a,-1)
fa0 = ffts(a0,-1)
fc = ffts(c,-1)
fc0 = ffts(c0,-1)

fig, ax = plt.subplots(nrows=1, ncols=2)
im = ax[0].imshow( np.imag(fa-fa0), cmap='gray', origin='lower' )
ax[0].set_title('Full')
im = ax[1].imshow( np.imag(fc-fc0), cmap='gray', origin='lower' )
ax[1].set_title('Compact')
plt.show()
