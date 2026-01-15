import os
from datetime import datetime
from astropy.io import fits
import numpy as np
from matplotlib import pyplot as plt
from hcipy import *
import configparser

import importlib.resources

def get_field_object (arr, grid, is_cube=False):
    field_obj = Field(arr, grid)

    if is_cube is True: 
        field_obj = Field(arr[i], grid)

    return field_obj 

def save_output(CAPyBARA_list, param_rst, img_list, actuator_list, path):
    """
    Function to save output data such as reference images, coronagraphic images,
    deformable mirror (DM) surfaces, and actuator states for each iteration and wavelength.

    Parameters:
    -----------
    CAPyBARA_list : list or single instance
        List of CAPyBARA simulation objects, or a single instance.
    param_rst : dict
        Dictionary containing simulation parameters.
    img_list : list
        List of images for each iteration and wavelength.
    actuator_list : list
        List of actuator commands for each iteration.
    path : str
        Directory path to save the output files.

    Returns:
    --------
    None
    """
    # Check if CAPyBARA_list is a list or a single instance
    is_multiple_instances = isinstance(CAPyBARA_list, list)

    # If CAPyBARA_list is a single instance, wrap it in a list to treat it uniformly
    if not is_multiple_instances:
        CAPyBARA_list = [CAPyBARA_list]

    # Ensure that wvl is a list, even if it’s a single integer
    if isinstance(param_rst['wvl'], int):
        param_rst['wvl'] = [param_rst['wvl']]  # Convert to list if it's an integer

    # Iterate over the number of iterations
    for i in range(param_rst['num_iteration']):
        print(f'Saving iteration: {i}')

        # Loop over wavelengths
        for j in range(len(param_rst['wvl'])):
            wvl = param_rst['wvl'][j]
            print(f'Saving for wavelength: {wvl}')

            # Get the reference image
            CAPyBARA_list[j].get_reference_image(wvl=wvl * 1e-9, check=False)
            ref_img = CAPyBARA_list[j].ref_img.shaped

            # Save the reference image
            write2fits(ref_img, key='direct', wvl=wvl, path=os.path.join(path, f'iteration_{i:04n}'))

            # Save the coronagraphic image
            corona_img = img_list[i][j].shaped
            write2fits(corona_img, key='sci', wvl=wvl, path=os.path.join(path, f'iteration_{i:04n}'))

            # Apply actuators
            CAPyBARA_list[j].apply_actuators(actuator_list[i])

            # Save DM1 surface
            dm1_surface = CAPyBARA_list[j].dm1.surface.shaped
            write2fits(dm1_surface, key='dm1_surface', wvl=wvl, path=os.path.join(path, f'iteration_{i:04n}'))

            # Save DM2 surface
            dm2_surface = CAPyBARA_list[j].dm2.surface.shaped
            write2fits(dm2_surface, key='dm2_surface', wvl=wvl, path=os.path.join(path, f'iteration_{i:04n}'))

            # Save actuator data
            write2fits(actuator_list[i], key='dm', wvl=wvl, path=os.path.join(path, f'iteration_{i:04n}'))

def write2fits(array: np.ndarray, key: str, wvl, path: str = None) -> None:
    """
    Save a numpy array as a FITS file with specified headers.

    Args:
        array (np.ndarray): The data array to be saved.
        param (dict): Dictionary containing parameters such as 'iwa', 'owa', and 'spatial_resolution'.
        key (str): Key to determine the type of file being saved (e.g., 'jacobian', 'dm', 'obs_seq').
        additional_headers (dict, optional): Additional headers to add to the FITS file. Defaults to None.
        path (str, optional): Directory path where the FITS file will be saved. Defaults to the current directory.

    Raises:
        ValueError: If an unrecognized key is provided.
    """
    # Get current date
    date = datetime.today().strftime('%Y-%m-%d')

    # Create a Primary HDU
    hdu = fits.PrimaryHDU(array)

    # TODO - put them back but not urgent
    # For now disable these headers
    # hdu.header.set('aperture', 'rst')
    # hdu.header.set('iwa', param['iwa'] / param['spatial_resolution'])
    # hdu.header.set('owa', param['owa'] / param['spatial_resolution'])
    # hdu.header.set('DM', '2 DM')
    # hdu.header.set('darkhole', 'circular')
    # hdu.header.set('date', date)

    # Set the path to current directory if not specified
    if path is None:
        path = os.getcwd()
    else:
        # Ensure the directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.abspath(path)

    # # Add additional headers if provided
    # if additional_headers is not None:
    #     for hdr_key, value in additional_headers.items():
    #         hdu.header.set(hdr_key, value)

    # Construct the filename based on the provided key
    if key == 'jacobian':
        fname = os.path.join(path, f'{date}_jacobian_matrix_{wvl}nm.fits')
    elif key == 'dm':
        fname = os.path.join(path, f'{date}_dm_command.fits')
    elif key == 'dm1_surface':
        fname = os.path.join(path, f'{date}_dm1_surface.fits')
    elif key == 'dm2_surface':
        fname = os.path.join(path, f'{date}_dm2_surface.fits')
    elif key == 'obs_seq':
        fname = os.path.join(path, f'{date}_observing_sequence.fits')
    elif key == 'psf':
        fname = os.path.join(path, f'{date}_psf.fits')
    elif key == 'sci':
        fname = os.path.join(path, f'{date}_sci_{wvl}nm.fits')
    elif key == 'direct':
        fname = os.path.join(path, f'{date}_direct_{wvl}nm.fits')
    else:
        raise ValueError(f"Unrecognized key: {key}. Expected one of ['jacobian', 'dm', 'obs_seq', 'psf'].")

    # Write the FITS file
    try:
        hdu.writeto(fname, overwrite=True)
        print(f'File saved as {fname}')
    except Exception as e:
        print(f'File {fname} cannot be written. Error: {e}')

def view_field_diff(arr1, arr2, log10: bool = False) -> np.ndarray:
    """
    View the diff(arr1-arr2) and show the image with a colorbar.
    If "log10" is True, the image will be shown with the log10 scale. 

    Args:
        arr1 (np.ndarray, 2D): 2D array. 
        arr2 (np.ndarray, 2D): Another 2D array. 
        log10 (bool, optional): Choose the scale of plotting. Defaults to False. 
        If set to True, the image will be shown with a log10 scale. 

    Returns:
        difference (np.ndarray, 2D): 2D array
    """
    plt.title('Diff')

    if log10 is True:
        imshow_field(np.log10(np.real(arr1 - arr2)))
        plt.colorbar()

    else: 
        imshow_field(np.real(arr1 - arr2))
        plt.colorbar()

    np.testing.assert_array_almost_equal(arr1, arr2)  

def view_arr_diff (arr1, arr2):
    print(' ==== Checking arrays ==== ')
    return np.allclose(arr1, arr2)

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

def get_rms_modes (modes, aperture):
    rms_modes = []
    aperture_mask = aperture > 0
    for mode in modes:
        rms = np.sqrt(np.sum(mode**2 * aperture_mask) / np.sum(aperture_mask))
        rms_modes.append(rms)
    return rms_modes

def get_scaling_factor_for_rms(self, target_rms, current_rms):
    """
    Calculate the scaling factor to adjust a mode's RMS.

    Parameters
    ----------
    target_rms : float
        The desired RMS value.
    current_rms : float
        The current RMS value.

    Returns
    -------
    float
        Scaling factor to adjust the mode's RMS to the target RMS.
    """
    return target_rms / current_rms