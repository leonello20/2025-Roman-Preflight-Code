import numpy as np

def calculate_broadband_image_and_contrast(CAPyBARA_list, coron_images, wvl_weights):
    """
    Calculate mean broadband image and contrast.
    
    Parameters
    ----------
    CAPyBARA_list : list
        List of CAPyBARA simulation instances
    coron_images : list
        Coronagraphic images for each iteration
    wvl_weights : list
        Weights for each wavelength (typically all 1s)
    
    Returns
    -------
    broadband_images : list
        Mean broadband images
    broadband_contrast : list
        Average contrast values
    """
    def ensure_boolean_mask(mask, grid):
        if callable(mask):
            mask = mask(grid)
        return mask.astype(bool)
    
    broadband_images = []
    broadband_contrast = []
    num_wavelengths = len(wvl_weights)
    
    for i in range(len(coron_images)):
        broadband_image = 0.0
        img_iteration = coron_images[i]
        
        # Average over wavelengths
        for wl in range(num_wavelengths):
            ref_img_flat = CAPyBARA_list[wl].ref_img.ravel()
            normalized_image = img_iteration[wl] / np.max(ref_img_flat)
            broadband_image += normalized_image * wvl_weights[wl]
        
        broadband_image /= np.sum(wvl_weights)
        broadband_images.append(broadband_image)
        
        # Calculate contrast in dark zone
        dark_zone_mask = ensure_boolean_mask(
            CAPyBARA_list[0].dark_zone_mask,
            CAPyBARA_list[0].focal_grid
        )
        mask_flat = dark_zone_mask.ravel()
        contrast = np.mean(broadband_image.ravel()[mask_flat])
        broadband_contrast.append(contrast)
    
    return broadband_images, broadband_contrast

def get_average_contrast(CAPyBARA, arr, is_mono=False):
    def ensure_boolean_mask(mask):
        """ Ensure the mask is a boolean array and has the correct shape. """
        if callable(mask):
            # If the mask is a function, evaluate it (assuming the grid is available)
            mask = mask(CAPyBARA.focal_grid)  # Adjust if grid changes
        return mask.astype(bool)

    # Ensure dark_zone_mask is boolean and flattened
    # CAPyBARA.dark_zone_mask = ensure_boolean_mask(CAPyBARA.dark_zone_mask)
    mask_flat = CAPyBARA.dark_zone_mask.ravel()

    # Ensure ref_img is flattened and 1D
    ref_img_flat = CAPyBARA.ref_img.ravel()

    if is_mono:
        # Check shape compatibility for monochromatic case
        average_contrast = np.asarray([
            np.mean(img.ravel()[mask_flat] / ref_img_flat.max()) for img in arr
        ])
    else:
        # For broadband, CAPyBARA is expected to be a list of simulations
        average_contrast = []
        num_iterations = CAPyBARA[0].param['num_iteration']  # Get number of iterations

        for i in range(num_iterations):
            for j in range(len(CAPyBARA)):
                CAPyBARA[j].dark_zone_mask = ensure_boolean_mask(CAPyBARA[j].dark_zone_mask)

                # Flatten the mask and ref_img if needed
                mask_flat = CAPyBARA[j].dark_zone_mask.ravel()
                ref_img_flat = CAPyBARA[j].ref_img.ravel()

                _contrast = np.asarray([
                    np.mean(img.ravel()[mask_flat] / ref_img_flat.max()) for img in arr[i]
                ])
                average_contrast.append(_contrast)

    return average_contrast
