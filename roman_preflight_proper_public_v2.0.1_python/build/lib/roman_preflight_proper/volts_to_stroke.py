import numpy as np
import roman_preflight_proper as rp

def volts_to_stroke( dm_struct, dm, dm_v, no_coupling=False ):
    """
    Convert volts to stroke (meters)
   
    dm_struct:
        array of DM parameters structures; must provide structures for all DMs, not just DM being used in this call
    dm: 
        integer specifying which DM, either 1 or 2
    dm_v:
        square array of voltages, 0 - 100, neighbor rule already checked
    no_coupling:
        if set to True, then do not apply coupling (default is False to apply coupling)

    NOTE: actuators move down towards the DM with increasing voltage

    """

    nact = dm_v.shape[0]

    # stroke_v_vs_volts is 3D array [nv,nact,nact] of stroke/voltage in 5V increments
    # coupling is [nv,nact,nact]

    sv = dm_struct[dm-1]['stroke_volts']
    nsv = sv.shape[0]
    cv = dm_struct[dm-1]['coupling_volts']
    ncv = cv.shape[0]

    padded_stroke = np.zeros((nact+2, nact+2))
    c = np.zeros((3,3))
    c[1,1] = 1.0

    for y in range(nact):
        ypad = y + 1
        y1 = ypad - 1
        y2 = ypad + 1
        for x in range(nact):
            u = dm_struct[dm-1]['stroke_m_vs_volts'][:,y,x].reshape((nsv))
            stroke = np.interp( dm_v[y,x], sv, u )

            if no_coupling == False:
                for j in range(3):
                    for i in range(3):
                        if i == 1 and j == 1:
                            continue
                        u = dm_struct[dm-1]['coupling'][:,y,x,j,i].reshape((ncv))
                        c[j,i] = np.interp( dm_v[y,x], cv, u )

            xpad = x + 1
            x1 = xpad - 1
            x2 = xpad + 1
            padded_stroke[y1:y2+1,x1:x2+1] += c * stroke

            stroke_m = rp.trim(padded_stroke, nact)

    # return value such that a positive stroke means the facesheet is moving up away from the DM
    # NOTE: these are not bias relative, but rather relative to 0V surface

    return -stroke_m
