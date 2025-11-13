import numpy as np
import proper
import matplotlib.pylab as plt
from roman_preflight_proper import trim

full,dx = proper.prop_run('roman_preflight',0.575,512,PASSVALUE={'use_errors':0,'source_x_offset':8.0,'source_y_offset':-4.0,'use_fpm':0,'use_field_stop':0})
full = trim(np.abs(full)**2,128)

compact,dx = proper.prop_run('roman_preflight_compact',0.575,512,PASSVALUE={'source_x_offset':8.0,'source_y_offset':-4.0,'use_fpm':0})     
compact = trim(np.abs(compact)**2,128)

print('total(compact) = '+str(np.sum(compact)))
print('total(full) = '+str(np.sum(full)))
print('max(full) = '+str(np.max(full)))
print('max(full-compact) = '+str(np.max(full-compact)))

full[62:67,64]=np.max(full)
full[64,62:67]=np.max(full)
compact[62:67,64]=np.max(compact)
compact[64,62:67]=np.max(compact)

fig, ax = plt.subplots(nrows=1, ncols=2)
im = ax[0].imshow( full, cmap='gray', origin='lower' )
ax[0].set_title('Full')
im = ax[1].imshow( compact, cmap='gray', origin='lower' )
ax[1].set_title('Compact')
plt.show()
