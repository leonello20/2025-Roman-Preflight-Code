#   Copyright 2022 California Institute of Technology
# ------------------------------------------------------------------

import proper
import numpy as np
import matplotlib.pylab as plt
import roman_preflight_proper

def run_defocus():

    lam = 0.575
    npsf = 512                 # output image dimension (must be power of 2)

    optval = []
    for i in range(5):
        optval.append({'cor_type':'hlc','use_fpm':0,'use_lyot_stop':0,'use_field_stop':0, \
                         'final_sampling_m':13e-6, 'output_dim':400, 'use_defocus_lens':0, 'use_pupil_lens':0})
    for i in range(4):
        optval[i]['use_defocus_lens'] = i + 1
    optval[4]['use_pupil_lens'] = 1

    (fields, sampling) = proper.prop_run_multi('roman_preflight', lam, npsf, QUIET=True, PASSVALUE=optval )
    images = np.abs(fields)**2

    fig, ax  = plt.subplots( nrows=2, ncols=3, figsize=(9,4) )

    im = ax[0,0].imshow(images[0,:,:])
    im = ax[0,1].imshow(images[1,:,:])
    im = ax[0,2].imshow(images[2,:,:])
    im = ax[1,0].imshow(images[3,:,:])
    im = ax[1,1].imshow(images[4,:,:])

    plt.show()
    
if __name__ == '__main__':
    run_defocus()
