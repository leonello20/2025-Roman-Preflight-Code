import proper
import numpy as np
import matplotlib.pylab as plt
from coronagraph import coronagraph

def occulter(wavelength, diam, grid_size, occrad, pupil, dm1, dm2, PASSVALUE={'occulter_type': 'GAUSSIAN'}):
    f_lens = 24 * diam 
    beam_ratio = 0.3 
    pupil_sampling = diam / grid_size

    # 1. Initialize the wavefront at the entrance pupil
    wfo = proper.prop_begin(diam, wavelength, grid_size, beam_ratio)
    # proper.prop_define_entrance(wfo)

    # 2. Read the pupil map and apply it to the wavefront
    pupil_map = proper.prop_readmap(wfo, pupil, SAMPLING=pupil_sampling)
    proper.prop_multiply(wfo, pupil_map)

    amplitude = proper.prop_get_amplitude(wfo)
    amplitude = np.fft.fftshift(amplitude)

    # 3. Plot the entrance pupil
    plt.figure(figsize=(12,8))
    plt.imshow(np.sqrt(amplitude), origin = "lower", cmap = plt.cm.gray)
    plt.suptitle("Entrance Pupil", fontsize = 18)
    plt.show()

    # 3. Call the coronagraph function
    wfo = coronagraph(wfo, f_lens, diam, occrad, PASSVALUE)

    # 4. End the PROPER simulation
    sampling = proper.prop_get_sampling(wfo)
    
    return (wfo, sampling)