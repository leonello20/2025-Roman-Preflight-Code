import numpy as np
import cgisim
import matplotlib.pyplot as plt

def hlc_simulation(mode, cor, bandpass, polaxis, params, star_spectrum, vmag):
    image, counts = cgisim.rcgisim(
        mode,
        cor,
        bandpass,
        polaxis, 
        params, 
        star_spectrum=star_spectrum, 
        star_vmag=vmag
    )
    return image, counts

def hlc_analysis(image, counts):
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log10(image + 1e-15), cmap='magma')
    plt.colorbar(label='Log10 Intensity')
    plt.title('HLC Band 1 - Raw Simulation Output (Uncorrected)')
    plt.show()

    peak_intensity = np.max(image)
    average_intensity = np.mean(image)
    contrast_estimate = peak_intensity / average_intensity
    print(f"Estimated Contrast Level: {contrast_estimate:.2e}")
    return contrast_estimate