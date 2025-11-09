import numpy as np

def stroke_to_volts( dm_struct, dm, dm_stroke_m ):
    """
    Convert stroke (meters) to volts
      dm_struct:
        array of DM parameters structures; must provide structures for all DMs, not just DM being used in this call
      dm:
        integer specifying which DM, either 1 or 2
      dm_stroke_m:
        a square array of DM strokes in meters, assuming a positive stroke is up away from the DM
        NOTE: these values are not bias-relative but rather relative to 0V surface

    NOTE: This routine does not take out coupling, if it is included in the input strokes.
    """

    nact = dm_stroke_m.shape[0]

    volts = np.zeros((nact, nact))
    v = dm_struct[dm-1]['stroke_volts']
    nv = v.shape[0]

    for y in range(nact):
        for x in range(nact):
            u = dm_struct[dm-1]['stroke_m_vs_volts'][:,y,x].reshape((nv))
            # as voltage increases, actuator moves closer to DM, so change sign of dm_stroke_m
            if np.sum(u) != 0: 
                volts[y,x] = np.interp( -dm_stroke_m[y,x], u, v )

    return volts
