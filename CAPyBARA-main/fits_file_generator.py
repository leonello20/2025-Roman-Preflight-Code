from astropy.io import fits
import numpy as np

# Create an empty or minimal PrimaryHDU object
# This creates a minimal header and no data
hdu = fits.PrimaryHDU()

# Alternatively, create an HDU with a specified structure but fill data with zeros
# data = np.zeros((100, 100), dtype=np.float32) # example for a 100x100 image
# hdu = fits.PrimaryHDU(data=data)

# Write the HDU to a new FITS file
# Use overwrite=True if the file might already exist
filename = "C:/Users/leone/OneDrive/Documents/GitHub/CAPyBARA-main/data/2026-01-14_jacobian_matrix_575nm.fits"
hdu.writeto(filename, overwrite=True)
print(f"Created empty FITS file: {filename}")