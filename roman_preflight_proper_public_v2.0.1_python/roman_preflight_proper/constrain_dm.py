import numpy as np
import astropy.io.fits as pyfits

DME_LSB = 110.0/2**16   # DAC LSB (not same as required min accuracy)
VMIN = 0

##################################################################################
def apply_neighbor_rule(dm_v0, vneighbor, margin, x1, y1, x2, y2):
    """
    Take a pair of columns and/or rows, check for neighbor rule violations
    between them, and fix them.
    """

    dm_v = dm_v0.copy()

    # Subtract adjacent rows/cols
  
    diff = dm_v[y1, x1] - dm_v[y2, x2]

    # Find if any neighbor rule violations exist. Use mask to exclude good
    # neighbors from correction.
    
    diff_mask = np.abs(diff) > (vneighbor - margin)
    diff *= diff_mask

    # Split excess in half, with a little extra margin to keep
    # numerical errors from making this reappear.  2x margin to
    # overshoot the correction slightly so this doesn't pop up
    # again.
    
    delta = (diff - np.sign(diff) * (vneighbor - 2*margin))/2.

    # Half on each side of the violation
    
    dm_v[y1, x1] -= delta
    dm_v[y2, x2] += delta

    # Return a mask for all the elements that were fixed
    
    neighbors_mask = (dm_v != dm_v0)

    return dm_v, neighbors_mask

##################################################################################
def check_actuators(dm_v0, vlat, vdiag, vquant, vmax, flat_dm_v=0.):
    """
    Check that the DM actuators obey neighbor rules and voltages limits

    Arguments:
     dm_v0: (float) 2D square array of actuator settings in volts to be fixed
     vlat: (float) Allowed voltage difference between lateral neighbors
     vdiag: (float) Allowed voltage difference between diagonal neighbors
     vquant: (float) quantization step in the DAC (LSB). Must be >0.
     vmax: (float) Maximum voltage permitted

    Keyword Arguments:
     flat_dm_v: (float) 2D array of the same size as dm_v0, or None. Defines the surface-
      flat voltages for neighbor rule checking.  If None, assumes 0V, consistent
      with unpowered polish (CGI baseline).  Neighbor rule must only be
      maintained with respect to a surface-flat array.  flat_dm_v must be <= vmax,
      >= vmin at all points.

    Returns:
     a 2D array of the same size as dm_v0

    """

    margin = 2*vquant
    dm_v = dm_v0.astype(float).copy() 

    # clip between min, max voltages

    vmax_q = vquant * (vmax // vquant)
    dm_v = dm_v.clip(min=0, max=vmax_q)
    limits_mask = (dm_v != dm_v0)

    # subtract surface-flat DM voltages

    dm_v -= flat_dm_v 
    dm_v_before = dm_v.copy()

    nrdone = False
   
    while not nrdone:
        # check left-adjacent columns

        x1 = slice(1, None, 2)
        y1 = slice(0, None)
        x2 = slice(0, -1, 2)
        y2 = y1
        dm_v_left, neighbors_mask_left = apply_neighbor_rule(dm_v, vlat, margin, x1, y1, x2, y2)

        # check right-adjacent columns

        x1 = slice(1, -2, 2)
        y1 = slice(0, None)
        x2 = slice(2, -1, 2)
        y2 = y1
        dm_v_right, neighbors_mask_right = apply_neighbor_rule(dm_v, vlat, margin, x1, y1, x2, y2)
        
        # Only apply second step adjustments to elements that were not adjusted
        # in the first step. This will avoid adjusting the same elements twice.

        dm_v[neighbors_mask_left] = dm_v_left[neighbors_mask_left]
        dm_v[~neighbors_mask_left] = dm_v_right[~neighbors_mask_left]

        # check bottom-adjacent rows
       
        x1 = slice(0, None)
        y1 = slice(1, None, 2)
        x2 = x1
        y2 = slice(0, -1, 2)
        dm_v_bottom, neighbors_mask_bottom = apply_neighbor_rule(dm_v, vlat, margin, x1, y1, x2, y2)

        # check top-adjacent rows
        
        x1 = slice(0, None)
        y1 = slice(1, -2, 2)
        x2 = x1
        y2 = slice(2, -1, 2)
        dm_v_top, neighbors_mask_top = apply_neighbor_rule(dm_v, vlat, margin, x1, y1, x2, y2)

        dm_v[neighbors_mask_bottom] = dm_v_bottom[neighbors_mask_bottom]
        dm_v[~neighbors_mask_bottom] = dm_v_top[~neighbors_mask_bottom]

        # check adjacent -x,-y diagonal 
       
        x1 = slice(1, None, 2)
        y1 = slice(1, None)
        x2 = slice(0, -1, 2)
        y2 = slice(0, -1)
        dm_v_diag1a, neighbors_mask_diag_1a = apply_neighbor_rule(dm_v, vdiag, margin, x1, y1, x2, y2)

        # check adjacent +x,+y diagonal 
       
        x1 = slice(1, -2, 2)
        y1 = slice(0, -1)
        x2 = slice(2, -1, 2)
        y2 = slice(1, None)
        dm_v_diag1b, neighbors_mask_diag_1b = apply_neighbor_rule(dm_v, vdiag, margin, x1, y1, x2, y2)

        dm_v[neighbors_mask_diag_1a] = dm_v_diag1a[neighbors_mask_diag_1a]
        dm_v[~neighbors_mask_diag_1a] = dm_v_diag1b[~neighbors_mask_diag_1a]

        # check adjacent -x,+y diagonal
       
        x1 = slice(1, None, 2)
        y1 = slice(0, -1)
        x2 = slice(0, -1, 2)
        y2 = slice(1, None)
        dm_v_diag2a, neighbors_mask_diag_2a = apply_neighbor_rule(dm_v, vdiag, margin, x1, y1, x2, y2)

        # check adjacent +x,-y diagonal

        x1 = slice(1, -2, 2)
        y1 = slice(1, None)
        x2 = slice(2, -1, 2)
        y2 = slice(0, -1)
        dm_v_diag2b, neighbors_mask_diag_2b = apply_neighbor_rule(dm_v, vdiag, margin, x1, y1, x2, y2)

        dm_v[neighbors_mask_diag_2a] = dm_v_diag2a[neighbors_mask_diag_2a]
        dm_v[~neighbors_mask_diag_2a] = dm_v_diag2b[~neighbors_mask_diag_2a]

        # If any of them had violations, go around again
        
        nrdone = (neighbors_mask_left.sum() == 0) and (neighbors_mask_right.sum() == 0) and \
                 (neighbors_mask_bottom.sum() == 0) and (neighbors_mask_top.sum() == 0) and \
                 (neighbors_mask_diag_1a.sum() == 0) and (neighbors_mask_diag_1b.sum() == 0) and \
                 (neighbors_mask_diag_2a.sum() == 0) and (neighbors_mask_diag_2b.sum() == 0) 

    neighbors_mask = (dm_v != dm_v_before)

    dm_v += flat_dm_v 
    dm_v = dm_v.clip(min=VMIN, max=vmax_q)
    limits_mask = limits_mask & (dm_v != dm_v_before)

    return dm_v, neighbors_mask, limits_mask

##################################################################################
def tie_actuators(dm_v, tie_map):
    """
      Set tie_map=-1 (dead) actuators to 0
      tie_map=0 actuators are individually set, to do not changed them
      tie_map>0 actuators are tied, so all with same tie value are set to mean
    """

    dm_tied = dm_v.copy() * (tie_map != -1)    # avoid dead actuators

    for tienum in np.unique(tie_map):
        if tienum < 1:      # avoid untied actuators
            continue
        meanval = np.mean(dm_tied[tie_map == tienum]) 
        dm_tied[tie_map == tienum] = meanval

    return dm_tied

##################################################################################
def constrain_dm( dm_struct, dm, dm_v, vlat=50.0, vdiag=75.0, vquant=DME_LSB, vmax=100.0, flat_dm_v=0.0 ):
    """
    Constrain each individual voltage to be in 0 <= v <= vmax. 

    Constrain each pair of laterally-adjacent actuators to be <= vlat after
    subtraction of the DM flat map in flat_map. Constrain each pair of diagonally-
    adjacent actuators to be <= vdiag.

    Constrain all tied actuators (groups in the tie matrix with value > 0)
    to have the same voltage.  Constrains all dead actuators (groups in the
    tie matrix with value = -1) to be 0V.

    flat_map should be >= 0, <= vmax, have all ties at the same voltage, and have 
    all dead actuators at 0V voltage.

    tie_map should be -1, 0, or the integer range 1-N for some N (with no gaps)

    Arguments:
     dm_struct: (array of dicts) DM parameter structure from load_cgi_dm_files
     dm (int) DM number (1 or 2)
     dm_v (float) 2D array of voltages, before fixing.
     vmax: (float) max allowed voltage.  CGI defaults = 100V
     vlat: (float) max allowed voltage differential between laterally-adjacent
        actuators.  CGI default = 50V
     vdiag: (float) max allowed voltage differential between diagonally-adjacent
        actuators.  CGI defaults = 75V
     vquant: (float) smallest voltage step (1 LSB) that the electronics can
        produce.  Keeps the constraints from being broken after conversion. 
        Must be non-zero. CGI default is 110 / 2^16
     flat_dm_v: (float) 2D array of voltages, same size as dm_v.  These are
        voltages that produce a flat DM surface (zero stress between actuators and
        faceplate); CGI GITL default is 0V

    Returns:
      checked_v: float array, the constrained DM voltages
      neighbors_mask: int array, indicates which actuators were altered due to neighbor rule
      limits_mask: int array, indicates which actuators hit the voltage limits

    """

    dm_v_in = dm_v.copy()

    tie_map = dm_struct[dm-1]['tie_map']

    checked_v, neighbors_mask, limits_mask = check_actuators( dm_v_in, vlat, vdiag, vquant, vmax, flat_dm_v )

    maxiter = 1000

    tied_v = None
    i = 0
    while not (tied_v == checked_v).all() and i < maxiter:
        tied_v = tie_actuators( checked_v, tie_map )
        checked_v, neighbors_mask_i, limits_mask_i = check_actuators( tied_v, vlat, vdiag, vquant, vmax, flat_dm_v ) 
        neighbors_mask = neighbors_mask | (neighbors_mask_i != 0) 
        limits_mask = limits_mask | (limits_mask_i != 0)
        i += 1

    return checked_v, neighbors_mask, limits_mask

