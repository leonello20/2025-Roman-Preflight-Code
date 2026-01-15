import os
from datetime import datetime
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from hcipy import *
import configparser

import importlib.resources

def read_fits (fname, axis=0):
    hdu = fits.open(fname)
    arr = hdu[axis].data

    return arr

def frame_rotate_interp(array, angle, center=None, mode='constant', cval=0, order=3):
    ''' Rotates a frame or 2D array.

        Parameters
        ----------
        array : Input image, 2d array.
        angle : Rotation angle (deg).
        center : Coordinates X,Y  of the point with respect to which the rotation will be
                    performed. By default the rotation is done with respect to the center
                    of the frame; central pixel if frame has odd size.
        interp_order: Interpolation order for the rotation. See skimage rotate function.
        border_mode : Pixel extrapolation method for handling the borders.
                    See skimage rotate function.

        Returns
        -------
        rotated_array : Resulting frame.

    '''
    dtype = array.dtype
    dims = array.shape
    angle_rad = -np.deg2rad(angle)

    if center is None:
        center = (np.array(dims)-1) / 2  # The minus 1 is because of python indexation at 0

    x, y = np.meshgrid(np.arange(dims[1], dtype=dtype), np.arange(dims[0], dtype=dtype))

    xp = (x-center[1])*np.cos(angle_rad) + (y-center[0])*np.sin(angle_rad) + center[1]
    yp = -(x-center[1])*np.sin(angle_rad) + (y-center[0])*np.cos(angle_rad) + center[0]

    rotated_array = ndimage.map_coordinates(array, [yp, xp], mode=mode, cval=cval, order=order)

    return rotated_array

def derotate_and_combine(cube, angles):

    """ Derotates a cube of images then mean-combine them.
    
    Parameters
    ----------
    cube : the input cube, 3D array.
    angles : the list of parallactic angles corresponding to the cube.
        
    Returns
    -------
    image_out : the mean-combined image
    cube_out : the cube of derotated frames.
    
    """
    if cube.ndim != 3:
        raise TypeError('Input cube is not a cube or 3d array')
    if angles.ndim != 1:
        raise TypeError('Input angles must be a 1D array')
    if len(cube) != len(angles):
        raise TypeError('Input cube and input angle list must have the same length')
        
    shape = cube.shape
    cube_out = np.zeros(shape)
    for im in range(shape[0]):
        cube_out[im] = frame_rotate_interp(cube[im], -angles[im])
    
    image_out = np.nanmean(cube_out, axis=0)
    return image_out, cube_out

def perturbation_evolution(perturbations, ppl_grid, radius_ppl_px = 336):
    """Make the perturbation evolve in a .mp4 file and datacube

    Parameters
    ------------
    title = title of the file, formalt ' .mp4'
    perturbations = array of fields
    radius_ppl_px = integer, radius of the pupil in pixels, 336 by default

    Return
    ----------
    turbulence_sequence = .mp4 file, evolution of the perturbation
    perturbation_datacube = datacube, perturbation
    """
    # turbulence_sequence = FFMpegWriter(title, framerate=5)
    perturbation_datacube = np.zeros((len(perturbations), radius_ppl_px*2, radius_ppl_px*2))
    for i in range(len(perturbations)):
        perturb = Field(perturbations[i], ppl_grid)
        # plt.clf()
        # imshow_field(perturb, vmax = 0.2, vmin = -0.2) # vmax=0.001, vmin=-0.001)
        # plt.colorbar()
        # turbulence_sequence.add_frame()
        perturbation_datacube[i,:,:] = perturb.shaped
    # plt.close()
    # turbulence_sequence.close()

    return perturbation_datacube

def load_and_print_default_txt():
    """
    Load and print the default.txt file bundled in the package.
    """
    try:
        # Use importlib.resources to access the data file inside the package
        with importlib.resources.open_text('CAPyBARA', 'show_capybara.txt') as file:
            content = file.read()
            # print("---- Contents of the default.txt file ----")
            print(content)
            # print("---- End of default.txt file ----")
    except FileNotFoundError:
        print("Error: The default.txt file was not found in the package.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

def read_ini_file(file_path):
    """
    Reads an .ini file and parses its content into a structured dictionary.

    Args:
        file_path (str): Path to the .ini configuration file.

    Returns:
        dict: Nested dictionary representing sections and their corresponding key-value pairs.
    """
    config = configparser.ConfigParser()
    config.read(file_path)

    ini_data = {}
    for section in config.sections():
        ini_data[section] = {key: parse_value(value) for key, value in config.items(section)}

    return ini_data

def parse_value(value):
    """
    Attempts to parse a string value into an appropriate Python data type.

    Args:
        value (str): The value to parse.

    Returns:
        The parsed value in the most appropriate data type (e.g., int, float, bool, list, or str).
    """
    # Check if the value is a list
    if ',' in value:
        value_list = value.split(',')
        return [parse_value(v.strip()) for v in value_list]

    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue

    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'

    return value

def write2fits(array: np.ndarray, key: str, wvl, path: str = None, additional_headers=None) -> None:
    """
    Legacy writer kept for compatibility with older debug dumps.
    I recommend using save_loop_products for new outputs.
    """
    date = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    hdu = fits.PrimaryHDU(array)

    if path is None:
        path = os.getcwd()
    else:
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.abspath(path)

    if additional_headers is not None:
        for hdr_key, value in additional_headers.items():
            try:
                hdu.header.set(hdr_key, value)
            except Exception:
                pass

    if key == "jacobian":
        fname = os.path.join(path, f"{date}_jacobian_matrix_{wvl}nm.fits")
    elif key == "dm":
        fname = os.path.join(path, f"{date}_dm_command.fits")
    elif key == "obs_seq":
        fname = os.path.join(path, f"{date}_observing_sequence.fits")
    elif key == "psf":
        fname = os.path.join(path, f"{date}_psf.fits")
    elif key == "coronagraphic":
        fname = os.path.join(path, f"{date}_coronagraphic_{wvl}nm.fits")
    else:
        raise ValueError("Unrecognised key.")

    hdu.writeto(fname, overwrite=True)
    print(f"File saved as {fname}")