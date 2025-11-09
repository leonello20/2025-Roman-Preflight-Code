import roman_preflight_proper

# Shift, rotate, and magnify an input image using FFTs/MFTs.
# Input image may be complex-valued.
# The input image should be appropriately zero-padded (array should be at least
# twice as wide as the image span)
# At a minimum, the shifts must be specified. If magnification is given, then
# so must the rotation.
# Rotation is limited to within +/- 45 deg

from roman_preflight_proper import mft_rotate, fft_rotate, shift_image

def transform_image( image, xshift_pix, yshift_pix, rotation_deg=0, mag=1 ):

    if rotation_deg < -45 or rotation_deg > 45:
        print("transform_image: rotation is limited to within +/- 45 deg.")
        raise Exception(' ')

    if mag != 1:
        # if magnifying, have to use mft-based routines
        return  mft_rotate( image, rotation_deg, xshift_pix, yshift_pix, mag )
    else:
        if rotation_deg != 0:
            return fft_rotate( image, rotation_deg, xshift_pix, yshift_pix )
        else:
            # shift_image is faster if no rotation needed
            return shift_image( image, xshift_pix, yshift_pix )

