# occulter.py
import proper
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from coronagraph import coronagraph # Import the coronagraph function from its module

def occulter(wavelength, diam, grid_size, occrad, PASSVALUE={'occulter_type': 'GAUSSIAN'}):
    f_lens = 24 * diam # Focal length of lenses (in meters)
    beam_ratio = 0.3 # percent of the grid diameter that will be filled by the initial beam

    # 1. Initialize the wavefront at the entrance pupil
    wfo = proper.prop_begin(diam, wavelength, grid_size, beam_ratio)

    # 2. Apply the primary aperture (telescope pupil)
    proper.prop_circular_aperture(wfo, diam) # Use diam directly for the diameter

    # 3. Define this as the entrance pupil for PROPER
    proper.prop_define_entrance(wfo)

    # 4. Call the coronagraph function to simulate the coronagraphic optics
    coronagraph(wfo, f_lens, PASSVALUE["occulter_type"], diam, occrad)

    # 5. End the PROPER simulation for this entire optical train
    sampling = proper.prop_get_sampling(wfo)
    return (wfo, sampling)