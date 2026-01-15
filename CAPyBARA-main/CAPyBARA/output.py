"""
CAPyBARA output utilities 

This module handles saving simulation outputs to disk with proper normalization,
FITS file creation, and metadata management.

Output Structure:
  <output_dir>/
    <ini filename>              # copied
    <log filename>              # copied
    efc/
      YYYY-MM-DD_efc_coronagraphic_cube.fits
      YYYY-MM-DD_efc_dm_commands.fits
      YYYY-MM-DD_efc_aberration_cube.fits
      YYYY-MM-DD_efc_parameters_snapshot.json
    reference/
      YYYY-MM-DD_reference_star_no_probes.fits
      YYYY-MM-DD_reference_star_with_probes.fits
      YYYY-MM-DD_reference_dm_command.fits
      YYYY-MM-DD_reference_aberration_cube.fits
      YYYY-MM-DD_reference_parameters_snapshot.json
    science_*/
      YYYY-MM-DD_observation_coronagraphic_cube.fits
      YYYY-MM-DD_observation_dm_commands.fits
      YYYY-MM-DD_observation_aberration_cube.fits
      YYYY-MM-DD_observation_parameters_snapshot.json
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import astropy.io.fits as fits

# Import contrast calculation function
try:
    from .contrast import calculate_broadband_image_and_contrast
except ImportError:
    # Handle case where module is run directly
    from contrast import calculate_broadband_image_and_contrast


def _image_to_array(img):
    """
    Convert image object to numpy array.
    
    Handles HCIPy Field objects with .shaped attribute or raw arrays.
    """
    if hasattr(img, "shaped"):
        return np.asarray(img.shaped)
    return np.asarray(img)


def _field_to_array(field_obj):
    """
    Convert field object to numpy array.
    
    Tries multiple attributes: .shaped, .electric_field, .data
    """
    if hasattr(field_obj, "shaped"):
        return np.asarray(field_obj.shaped)
    if hasattr(field_obj, "electric_field"):
        return np.asarray(field_obj.electric_field)
    if hasattr(field_obj, "data"):
        return np.asarray(field_obj.data)
    return np.asarray(field_obj)


def _stack_coronagraphic_cube(img_list):
    """
    Stack images into a cube.
    
    Accepts:
      - Monochromatic: img_list[iter]
      - Broadband: img_list[iter][wvl]
    
    Returns:
      - Monochromatic: (n_iter, ny, nx)
      - Broadband: (n_iter, n_wvl, ny, nx)
    """
    if isinstance(img_list, np.ndarray):
        return img_list

    if not isinstance(img_list, (list, tuple)):
        return np.asarray(img_list)

    if len(img_list) == 0:
        return np.zeros((0,), dtype=float)

    first = img_list[0]
    if isinstance(first, (list, tuple)):
        # Broadband case
        frames = []
        for frame in img_list:
            wvls = [_image_to_array(im) for im in frame]
            frames.append(np.stack(wvls, axis=0))
        return np.stack(frames, axis=0)

    # Monochromatic case
    frames = [_image_to_array(im) for im in img_list]
    return np.stack(frames, axis=0)


def _stack_dm_commands(actuator_list):
    """
    Stack DM actuator commands.
    
    Accepts:
      - Static vector: (n_act,)
      - Time series: list of (n_act,) arrays
      - Pre-stacked ndarray
    
    Returns:
      - Static: (n_act,)
      - Series: (n_step, n_act)
    """
    if actuator_list is None:
        return np.zeros((0,), dtype=float)

    if isinstance(actuator_list, np.ndarray):
        return actuator_list

    if isinstance(actuator_list, (list, tuple)):
        if len(actuator_list) == 0:
            return np.zeros((0,), dtype=float)
        return np.stack([np.asarray(v) for v in actuator_list], axis=0)

    return np.asarray(actuator_list)

def _coerce_aberration_cube(aberration_cube):
    """
    Convert aberration data to standardized cube format.
    
    Accepts:
      - ndarray
      - list of Field objects
      - list of arrays
      - single Field object
    
    Returns:
      - Cube with shape (n_frames, ny, nx)
    """
    if aberration_cube is None:
        return np.zeros((0,), dtype=float)

    if isinstance(aberration_cube, np.ndarray):
        return aberration_cube

    if isinstance(aberration_cube, (list, tuple)):
        if len(aberration_cube) == 0:
            return np.zeros((0,), dtype=float)
        return np.stack([_field_to_array(x) for x in aberration_cube], axis=0)

    return _field_to_array(aberration_cube)


def _get_ref_max_list(CAPyBARA_list, num_wvl):
    """
    Extract reference image maxima per wavelength.
    
    Parameters
    ----------
    CAPyBARA_list : CAPyBARAsim or List[CAPyBARAsim]
        Single simulation (mono) or list (broadband)
    num_wvl : int
        Expected number of wavelengths
    
    Returns
    -------
    list of float
        Reference maximum for each wavelength
    
    Raises
    ------
    ValueError
        If CAPyBARA_list is None or has wrong length
    """
    if CAPyBARA_list is None:
        raise ValueError("CAPyBARA_list is required to normalise images by reference.")

    if isinstance(CAPyBARA_list, (list, tuple)):
        if len(CAPyBARA_list) < num_wvl:
            raise ValueError(
                f"CAPyBARA_list length ({len(CAPyBARA_list)}) < "
                f"number of wavelengths ({num_wvl})."
            )
        ref_max = []
        for j in range(num_wvl):
            ref_img = np.asarray(CAPyBARA_list[j].ref_img)
            max_val = float(np.max(ref_img.ravel()))
            if max_val == 0:
                raise ValueError(f"Reference image max is zero for wavelength {j}")
            ref_max.append(max_val)
        return ref_max

    # Monochromatic case
    ref_img = np.asarray(CAPyBARA_list.ref_img)
    max_val = float(np.max(ref_img.ravel()))
    if max_val == 0:
        raise ValueError("Reference image max is zero")
    return [max_val]


def _infer_img_shape_from_sim_list(sim_list):
    """
    Infer focal plane image shape from simulation object.
    
    Parameters
    ----------
    sim_list : CAPyBARAsim or List[CAPyBARAsim]
        Simulation instance(s)
    
    Returns
    -------
    tuple of int
        (ny, nx) shape, defaults to (64, 64) if not detectable
    """
    try:
        sim0 = sim_list[0] if isinstance(sim_list, (list, tuple)) else sim_list
        g = getattr(sim0, "focal_grid", None)
        if g is not None and hasattr(g, "shape") and g.shape is not None:
            return tuple(g.shape)
    except Exception:
        pass
    return (64, 64)


def _normalised_coronagraphic_cube(sim_list, coron_images, wvl_weights, img_shape=None):
    """
    Create normalized coronagraphic image cube.
    
    Normalization approach:
      - Broadband: uses calculate_broadband_image_and_contrast()
      - Monochromatic: divides by max(sim.ref_img)
    
    Parameters
    ----------
    sim_list : CAPyBARAsim or List[CAPyBARAsim]
        Single simulation (mono) or list (broadband)
    coron_images : list or ndarray
        Raw coronagraphic images
        Broadband: coron_images[iter][wvl]
        Monochromatic: coron_images[iter]
    wvl_weights : Sequence[float]
        Wavelength weights for broadband averaging
    img_shape : tuple of int, optional
        Expected (ny, nx) shape. Auto-detected if None.
    
    Returns
    -------
    ndarray
        Shape (n_frames, ny, nx) of normalized images
    
    Raises
    ------
    ValueError
        If reference max is zero or required attributes missing
    """
    simL = list(sim_list) if isinstance(sim_list, (list, tuple)) else [sim_list]

    if img_shape is None:
        img_shape = _infer_img_shape_from_sim_list(simL)

    # Monochromatic case
    if len(simL) == 1:
        sim0 = simL[0]
        if not hasattr(sim0, "ref_img"):
            raise ValueError("Monochromatic normalisation requires sim.ref_img attribute.")
        
        ref_max = float(np.max(np.asarray(sim0.ref_img).ravel()))
        if ref_max == 0:
            raise ValueError("Reference max is zero, cannot normalise.")

        # Handle pre-stacked arrays
        if isinstance(coron_images, np.ndarray):
            arr = coron_images
            if arr.ndim == 2:
                arr = arr[None, ...]
            return (arr / ref_max).reshape((arr.shape[0],) + img_shape)

        # Handle list of images
        frames = []
        for im in coron_images:
            arr = _image_to_array(im).reshape(img_shape)
            frames.append(arr / ref_max)
        return np.asarray(frames)

    # Broadband case
    broadband_images, _ = calculate_broadband_image_and_contrast(
        simL, coron_images, list(wvl_weights)
    )
    return np.asarray([np.asarray(img).reshape(img_shape) for img in broadband_images])


# ============================================================================
# CORE HELPER FUNCTIONS - Metadata and Headers
# ============================================================================

def _today_str():
    """Return today's date as YYYY-MM-DD string."""
    return datetime.today().strftime("%Y-%m-%d")


def _normalise_wvl_list(wvl):
    """
    Normalize wavelength input to list of floats.
    
    Accepts:
      - Single number: 600
      - List: [600] or [575, 600, 625]
      - Accidentally nested: [[575, 600, 625]]
    
    Returns:
      - list of float
    """
    if wvl is None:
        return []

    # Scalars
    if isinstance(wvl, (int, float, np.integer, np.floating)):
        return [float(wvl)]

    # Handle one level of accidental nesting
    if (isinstance(wvl, (list, tuple, np.ndarray)) and 
        len(wvl) == 1 and 
        isinstance(wvl[0], (list, tuple, np.ndarray))):
        wvl = wvl[0]

    return [float(x) for x in wvl]


def _build_common_headers(loop_name, param=None, seed=None, roll_deg=None, 
                          offset_rms_nm=None, wavelengths_nm=None, extra_headers=None):
    """
    Build common FITS header keywords.
    
    Parameters
    ----------
    loop_name : str
        Loop identifier (e.g., 'EFC', 'OBS', 'REF')
    param : dict, optional
        Parameter dictionary with num_iteration, loop_gain, etc.
    seed : int, optional
        Random seed for aberrations
    roll_deg : float, optional
        Telescope roll angle in degrees
    offset_rms_nm : float, optional
        RMS of offset aberration in nm
    wavelengths_nm : Sequence[float], optional
        Wavelengths in nm
    extra_headers : dict, optional
        Additional header keywords
    
    Returns
    -------
    dict
        Header keyword-value pairs
    """
    hdr = {}
    hdr["LOOP"] = str(loop_name)[:8].upper()

    if seed is not None:
        hdr["SEED"] = int(seed)
    if roll_deg is not None:
        hdr["ROLLDEG"] = float(roll_deg)
    if offset_rms_nm is not None:
        hdr["OFFRNM"] = float(offset_rms_nm)

    if wavelengths_nm is not None:
        w = _normalise_wvl_list(wavelengths_nm)
        hdr["NWVL"] = int(len(w))
        # Store up to 10 wavelengths
        for i, val in enumerate(w[:10]):
            hdr[f"WVL{i:03d}"] = float(val)

    if param:
        if "ref_wvl" in param:
            hdr["REFWVL"] = float(param["ref_wvl"])
        if "num_iteration" in param:
            hdr["NITER"] = int(param["num_iteration"])
        if "num_mode" in param:
            hdr["NMODE"] = int(param["num_mode"])
        if "loop_gain" in param:
            hdr["GAIN"] = float(param["loop_gain"])
        if "rcond" in param:
            hdr["RCOND"] = float(param["rcond"])

    if extra_headers:
        hdr.update(extra_headers)

    return hdr


def _add_headers_from_dict(hdr, headers):
    """
    Add headers from dictionary to FITS header object.
    
    Truncates keys to 8 characters and converts to uppercase per FITS standard.
    """
    if not headers:
        return
    for k, v in headers.items():
        try:
            key = str(k)[:8].upper()
            hdr.set(key, v)
        except Exception:
            pass


# ============================================================================
# CORE HELPER FUNCTIONS - File System
# ============================================================================

def _ensure_dir(path):
    """Create directory if it doesn't exist, return Path object."""
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_json(path, payload):
    """
    Write dictionary to JSON file.
    
    Returns
    -------
    str
        Absolute path to written file
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"File saved as {path}")
    return str(path)


def _write_parameters_snapshot_json(folder, basename, payload):
    """
    Write parameters snapshot JSON file with date prefix.
    
    Parameters
    ----------
    folder : Path-like
        Output directory
    basename : str
        Base filename (date will be prepended)
    payload : dict
        Data to save
    
    Returns
    -------
    str
        Absolute path to written file
    """
    folder = _ensure_dir(folder)
    date = _today_str()
    return _write_json(folder / f"{date}_{basename}.json", payload)


def create_experiment_dir(base_data_path, run_name=None):
    """
    Create experiment directory with timestamp.
    
    Parameters
    ----------
    base_data_path : Path-like
        Base directory for experiments
    run_name : str, optional
        Custom run name. If None, uses YYYY-MM-DD_HHMMSS format.
    
    Returns
    -------
    Path
        Created experiment directory
    
    Raises
    ------
    FileExistsError
        If directory already exists
    """
    base = Path(base_data_path).expanduser().resolve()

    if run_name is None:
        run_name = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    exp_dir = base / run_name
    exp_dir.mkdir(parents=True, exist_ok=False)

    return exp_dir


def copy_run_metadata(output_dir, ini_path=None, log_path=None):
    """
    Copy ini and log files to output directory root.
    
    Safe to call multiple times.
    
    Parameters
    ----------
    output_dir : Path-like
        Destination directory
    ini_path : Path-like, optional
        Path to INI configuration file
    log_path : Path-like, optional
        Path to log file
    """
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if ini_path is not None:
        ini_path = Path(ini_path).expanduser().resolve()
        if ini_path.exists():
            shutil.copy2(ini_path, output_dir / ini_path.name)

    if log_path is not None:
        log_path = Path(log_path).expanduser().resolve()
        if log_path.exists():
            shutil.copy2(log_path, output_dir / log_path.name)


def prepare_output_dir(output_dir, ini_path=None, log_path=None):
    """
    Prepare output directory and copy metadata files.
    
    Parameters
    ----------
    output_dir : Path-like
        Output directory to create
    ini_path : Path-like, optional
        INI file to copy
    log_path : Path-like, optional
        Log file to copy
    
    Returns
    -------
    dict
        Paths to copied files: {"ini": path or None, "log": path or None}
    """
    out = _ensure_dir(output_dir)
    copied = {"ini": None, "log": None}

    if ini_path is not None:
        ini = Path(ini_path).expanduser().resolve()
        if ini.exists():
            dst = out / ini.name
            shutil.copy2(str(ini), str(dst))
            copied["ini"] = str(dst)

    if log_path is not None:
        log = Path(log_path).expanduser().resolve()
        if log.exists():
            dst = out / log.name
            shutil.copy2(str(log), str(dst))
            copied["log"] = str(dst)

    return copied


# ============================================================================
# FITS Writing
# ============================================================================

def write2fits(array, key, wvl, path, additional_headers=None):
    """
    Write array to FITS file with standardized naming.
    
    Parameters
    ----------
    array : ndarray
        Data to write
    key : str
        File type identifier (e.g., 'efc_coronagraphic_cube')
    wvl : float
        Wavelength (currently unused but kept for API compatibility)
    path : Path-like
        Output directory
    additional_headers : dict, optional
        Additional FITS header keywords
    
    Returns
    -------
    str
        Path to saved file
    
    Raises
    ------
    ValueError
        If key is not recognized
    """
    date = _today_str()
    path = _ensure_dir(path)
    hdu = fits.PrimaryHDU(array)

    if additional_headers is not None:
        _add_headers_from_dict(hdu.header, additional_headers)

    # Map keys to filenames
    filename_map = {
        "efc_coronagraphic_cube": f"{date}_efc_coronagraphic_cube.fits",
        "efc_dm_commands": f"{date}_efc_dm_commands.fits",
        "efc_aberration_cube": f"{date}_efc_aberration_cube.fits",
        "observation_coronagraphic_cube": f"{date}_observation_coronagraphic_cube.fits",
        "observation_dm_commands": f"{date}_observation_dm_commands.fits",
        "observation_aberration_cube": f"{date}_observation_aberration_cube.fits",
        "reference_star_no_probes": f"{date}_reference_star_no_probes.fits",
        "reference_star_with_probes": f"{date}_reference_star_with_probes.fits",
        "reference_dm_command": f"{date}_reference_dm_command.fits",
        "reference_aberration_cube": f"{date}_reference_aberration_cube.fits",
    }

    if key not in filename_map:
        raise ValueError(f"Unrecognised key: {key}")

    fname = path / filename_map[key]
    hdu.writeto(str(fname), overwrite=True)
    print(f"File saved as {fname}")
    return str(fname)


# ============================================================================
# Save 
# ============================================================================

def save_efc_products(output_dir, param_efc, img_list, actuator_list, sim_list, 
                      wvl_weights, aberration_cube=None, seed=None, wavelengths_nm=None, 
                      extra_headers=None, parameters_snapshot=None, img_shape=None):
    """
    Save EFC (Electric Field Conjugation) loop products.
    
    Creates efc/ subdirectory containing:
      - Normalized coronagraphic image cube
      - DM actuator commands
      - Aberration cube (optional)
      - Parameters snapshot JSON
    
    Parameters
    ----------
    output_dir : Path-like
        Base output directory
    param_efc : dict
        EFC parameters from config
    img_list : list or ndarray
        Coronagraphic images (mono: [iter], broad: [iter][wvl])
    actuator_list : list or ndarray
        DM actuator commands per iteration
    sim_list : CAPyBARAsim or list
        Simulation instance(s)
    wvl_weights : Sequence[float]
        Wavelength weights
    aberration_cube : optional
        Aberration field data
    seed : int, optional
        Random seed
    wavelengths_nm : Sequence[float], optional
        Wavelengths in nm
    extra_headers : dict, optional
        Additional FITS headers
    parameters_snapshot : dict, optional
        Additional parameters to save
    img_shape : tuple, optional
        Image shape (ny, nx)
    
    Returns
    -------
    dict
        Paths to saved files
    """
    out = _ensure_dir(output_dir)
    efc_dir = _ensure_dir(out / "efc")

    # Build common headers
    hdr = _build_common_headers(
        "EFC",
        param=param_efc,
        seed=seed,
        wavelengths_nm=wavelengths_nm,
        extra_headers=extra_headers,
    )

    # Create normalized coronagraphic cube
    coro_cube = _normalised_coronagraphic_cube(
        sim_list=sim_list,
        coron_images=img_list,
        wvl_weights=wvl_weights,
        img_shape=img_shape,
    )
    coro_path = write2fits(
        coro_cube, 
        key="efc_coronagraphic_cube", 
        wvl=0, 
        path=efc_dir, 
        additional_headers=hdr
    )

    # Save DM commands
    dm_cmds = _stack_dm_commands(actuator_list)
    hdr_dm = dict(hdr)
    if dm_cmds.ndim == 1:
        hdr_dm["DMTIME"] = "STATIC"
    else:
        hdr_dm["DMTIME"] = "SERIES"
        hdr_dm["NDMSTEP"] = int(dm_cmds.shape[0])
    dm_path = write2fits(
        dm_cmds, 
        key="efc_dm_commands", 
        wvl=0, 
        path=efc_dir, 
        additional_headers=hdr_dm
    )

    # Save aberration cube if provided
    ab_path = ""
    if aberration_cube is not None:
        ab = _coerce_aberration_cube(aberration_cube)
        ab_path = write2fits(
            ab, 
            key="efc_aberration_cube", 
            wvl=0, 
            path=efc_dir, 
            additional_headers=hdr
        )

    # Save parameters snapshot
    snap_payload = {
        "loop": "EFC",
        "seed": seed,
        "wavelengths_nm": list(_normalise_wvl_list(wavelengths_nm)),
        "wvl_weights": list(wvl_weights),
        "img_shape": list(img_shape) if img_shape is not None else list(
            _infer_img_shape_from_sim_list(sim_list)
        ),
        "files": {
            "coronagraphic_cube": coro_path,
            "dm_commands": dm_path,
            "aberration_cube": ab_path if ab_path else None,
        },
        "parameters": parameters_snapshot or {},
    }
    snap_path = _write_parameters_snapshot_json(
        efc_dir, "efc_parameters_snapshot", snap_payload
    )

    return {
        "coronagraphic_cube": coro_path,
        "dm_commands": dm_path,
        "aberration_cube": ab_path,
        "snapshot_json": snap_path,
    }


def save_observation_products(output_dir, obs_name, param_obs, img_list, actuator_list, 
                              sim_list, wvl_weights, aberration_cube=None, seed=None, 
                              roll_deg=None, offset_rms_nm=None, wavelengths_nm=None, 
                              extra_headers=None, parameters_snapshot=None, img_shape=None):
    """
    Save science observation products.
    
    Creates science_*/ subdirectory containing:
      - Normalized coronagraphic image cube
      - DM actuator commands
      - Aberration cube (optional)
      - Parameters snapshot JSON
    
    Parameters
    ----------
    output_dir : Path-like
        Base output directory
    obs_name : str
        Observation identifier (e.g., 'science_A1')
    param_obs : dict
        Observation parameters from config
    img_list : list or ndarray
        Coronagraphic images
    actuator_list : list or ndarray
        DM actuator commands
    sim_list : CAPyBARAsim or list
        Simulation instance(s)
    wvl_weights : Sequence[float]
        Wavelength weights
    aberration_cube : optional
        Aberration field data
    seed : int, optional
        Random seed
    roll_deg : float, optional
        Telescope roll angle
    offset_rms_nm : float, optional
        Slew offset RMS in nm
    wavelengths_nm : Sequence[float], optional
        Wavelengths in nm
    extra_headers : dict, optional
        Additional FITS headers
    parameters_snapshot : dict, optional
        Additional parameters to save
    img_shape : tuple, optional
        Image shape (ny, nx)
    
    Returns
    -------
    dict
        Paths to saved files
    """
    out = _ensure_dir(output_dir)
    obs_dir = _ensure_dir(out / obs_name)

    # Build common headers
    hdr = _build_common_headers(
        "OBS",
        param=param_obs,
        seed=seed,
        roll_deg=roll_deg,
        offset_rms_nm=offset_rms_nm,
        wavelengths_nm=wavelengths_nm,
        extra_headers=extra_headers,
    )

    # Create normalized coronagraphic cube
    coro_cube = _normalised_coronagraphic_cube(
        sim_list=sim_list,
        coron_images=img_list,
        wvl_weights=wvl_weights,
        img_shape=img_shape,
    )
    coro_path = write2fits(
        coro_cube,
        key="observation_coronagraphic_cube",
        wvl=0,
        path=obs_dir,
        additional_headers=hdr
    )

    # Save DM commands
    dm_cmds = _stack_dm_commands(actuator_list)
    hdr_dm = dict(hdr)
    if dm_cmds.ndim == 1:
        hdr_dm["DMTIME"] = "STATIC"
    else:
        hdr_dm["DMTIME"] = "SERIES"
        hdr_dm["NDMSTEP"] = int(dm_cmds.shape[0])
    dm_path = write2fits(
        dm_cmds,
        key="observation_dm_commands",
        wvl=0,
        path=obs_dir,
        additional_headers=hdr_dm
    )

    # Save aberration cube if provided
    ab_path = ""
    if aberration_cube is not None:
        ab = _coerce_aberration_cube(aberration_cube)
        ab_path = write2fits(
            ab,
            key="observation_aberration_cube",
            wvl=0,
            path=obs_dir,
            additional_headers=hdr
        )

    # Save parameters snapshot
    snap_payload = {
        "loop": "OBSERVATION",
        "name": obs_name,
        "seed": seed,
        "roll_deg": roll_deg,
        "offset_rms_nm": offset_rms_nm,
        "wavelengths_nm": list(_normalise_wvl_list(wavelengths_nm)),
        "wvl_weights": list(wvl_weights),
        "img_shape": list(img_shape) if img_shape is not None else list(
            _infer_img_shape_from_sim_list(sim_list)
        ),
        "files": {
            "coronagraphic_cube": coro_path,
            "dm_commands": dm_path,
            "aberration_cube": ab_path if ab_path else None,
        },
        "parameters": parameters_snapshot or {},
    }
    snap_path = _write_parameters_snapshot_json(
        obs_dir, "observation_parameters_snapshot", snap_payload
    )

    return {
        "coronagraphic_cube": coro_path,
        "dm_commands": dm_path,
        "aberration_cube": ab_path,
        "snapshot_json": snap_path,
    }


def save_reference_products(output_dir, ref_no_probes_img_list, ref_with_probes_img_list, 
                            sim_list, wvl_weights, dm_command=None, aberration_cube=None, 
                            seed=None, param_ref=None, wavelengths_nm=None, extra_headers=None, 
                            parameters_snapshot=None, img_shape=None):
    """
    Save reference star observation products.
    
    Creates reference/ subdirectory containing:
      - Reference star without probes
      - Reference star with probes
      - DM command (optional)
      - Aberration cube (optional)
      - Parameters snapshot JSON
    
    Parameters
    ----------
    output_dir : Path-like
        Base output directory
    ref_no_probes_img_list : list or ndarray
        Reference images without probes
    ref_with_probes_img_list : list or ndarray
        Reference images with probes
    sim_list : CAPyBARAsim or list
        Simulation instance(s)
    wvl_weights : Sequence[float]
        Wavelength weights
    dm_command : optional
        DM actuator command
    aberration_cube : optional
        Aberration field data
    seed : int, optional
        Random seed
    param_ref : dict, optional
        Reference parameters
    wavelengths_nm : Sequence[float], optional
        Wavelengths in nm
    extra_headers : dict, optional
        Additional FITS headers
    parameters_snapshot : dict, optional
        Additional parameters to save
    img_shape : tuple, optional
        Image shape (ny, nx)
    
    Returns
    -------
    dict
        Paths to saved files
    """
    out = _ensure_dir(output_dir)
    ref_dir = _ensure_dir(out / "reference")

    # Build common headers
    hdr = _build_common_headers(
        "REF",
        param=param_ref,
        seed=seed,
        wavelengths_nm=wavelengths_nm,
        extra_headers=extra_headers,
    )

    # Create normalized cubes
    ref_no_cube = _normalised_coronagraphic_cube(
        sim_list=sim_list,
        coron_images=ref_no_probes_img_list,
        wvl_weights=wvl_weights,
        img_shape=img_shape,
    )
    ref_with_cube = _normalised_coronagraphic_cube(
        sim_list=sim_list,
        coron_images=ref_with_probes_img_list,
        wvl_weights=wvl_weights,
        img_shape=img_shape,
    )

    no_probe_path = write2fits(
        ref_no_cube,
        key="reference_star_no_probes",
        wvl=0,
        path=ref_dir,
        additional_headers=hdr
    )
    with_probe_path = write2fits(
        ref_with_cube,
        key="reference_star_with_probes",
        wvl=0,
        path=ref_dir,
        additional_headers=hdr
    )

    # Save DM command if provided
    dm_path = ""
    if dm_command is not None:
        dm_path = write2fits(
            np.asarray(dm_command),
            key="reference_dm_command",
            wvl=0,
            path=ref_dir,
            additional_headers=hdr
        )

    # Save aberration cube if provided
    ab_path = ""
    if aberration_cube is not None:
        ab = _coerce_aberration_cube(aberration_cube)
        ab_path = write2fits(
            ab,
            key="reference_aberration_cube",
            wvl=0,
            path=ref_dir,
            additional_headers=hdr
        )

    # Save parameters snapshot
    snap_payload = {
        "loop": "REFERENCE",
        "seed": seed,
        "wavelengths_nm": list(_normalise_wvl_list(wavelengths_nm)),
        "wvl_weights": list(wvl_weights),
        "img_shape": list(img_shape) if img_shape is not None else list(
            _infer_img_shape_from_sim_list(sim_list)
        ),
        "files": {
            "reference_star_no_probes": no_probe_path,
            "reference_star_with_probes": with_probe_path,
            "dm_command": dm_path if dm_path else None,
            "aberration_cube": ab_path if ab_path else None,
        },
        "parameters": parameters_snapshot or {},
    }
    snap_path = _write_parameters_snapshot_json(
        ref_dir, "reference_parameters_snapshot", snap_payload
    )

    return {
        "reference_star_no_probes": no_probe_path,
        "reference_star_with_probes": with_probe_path,
        "dm_command": dm_path,
        "aberration_cube": ab_path,
        "snapshot_json": snap_path,
    }