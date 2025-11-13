import copy
import numpy as np
import proper
import roman_preflight_proper as rp

def backout_coupling( dm_struct, dm, stroke0_m ):

    #-- must first call load_cgi_dm_files before calling this
    #--
    #-- dm_struct is array of DM parameter dicts for all DMs 
    #-- dm is 1 or 2
    #-- stroke0_m is a square array, NOT bias subtracted (stroke0_m >= 0)
    #-- This routine only attempts to back out -Y coupling at 40V on DM2;
    #-- Coupling in other directions is not considered.

    nact = stroke0_m.shape[0]
    stroke_m = np.full((nact+2,nact+2), stroke0_m[0,0])
    stroke_m[1:nact+1,1:nact+1] = stroke0_m

    coupling = np.zeros((nact+2,nact+2))
    v = dm_struct[dm-1]['coupling_volts']
    nv = v.shape[0]

    for y in range(nact):
        for x in range(nact):
            #-- dm_struct.coupling is [nvolts,nact,nact,3,3]
            c = dm_struct[dm-1]['coupling'][:,y,x,0,1].reshape((nv))
            #-- extract -Y coupling at 40V
            coupling[y+1,x+1] = np.interp(40., v, c)

    old_stroke_m = copy.deepcopy(stroke_m)

    for i in range(15):
        new_stroke_m = stroke_m - np.roll(old_stroke_m*coupling, -1, axis=0)
        old_stroke_m[1:nact+1,1:nact+1] = rp.trim(new_stroke_m,nact)

    new_stroke_m = rp.trim(new_stroke_m, nact)
    return new_stroke_m
