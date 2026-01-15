"""
CAPyBARA Simulation Runner - Following Notebook Workflow Exactly

This script follows the exact workflow from example_observing_sequence.ipynb

Usage:
    python run_CAPyBARA.py --mode jacobian --config capy-pup-900.ini
    python run_CAPyBARA.py --mode mono --config capy-pup-900.ini
    python run_CAPyBARA.py --mode broadband --config capy-pup-900.ini
"""

import argparse
import copy
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.visualization import astropy_mpl_style
from hcipy import *
import scipy.fftpack

# CAPyBARA imports
import CAPyBARA.aberration as rst_aberration
import CAPyBARA.plotting as plotting
import CAPyBARA.rst_functions as rst_func
import CAPyBARA.utils as utils
from CAPyBARA import CAPyBARAsim
from CAPyBARA.efc import EFieldConjugation
from CAPyBARA.observing_sequence import ObservingSequence
from CAPyBARA.plotting import plt_field, plt_std_vs_iteration
from CAPyBARA.rst_design import _make_rst_aperture, _make_lyot_mask
from CAPyBARA.rst_functions import *
from CAPyBARA.utils import read_ini_file, load_and_print_default_txt

# Import new output functions
from CAPyBARA.output import (
    create_experiment_dir,
    copy_run_metadata,
    save_efc_products,
    save_observation_products,
    save_reference_products,
)

# Import contrast calculation
from CAPyBARA.contrast import calculate_broadband_image_and_contrast, get_average_contrast

# Set matplotlib style
plt.style.use(astropy_mpl_style)
plt.rcParams['image.origin'] = 'lower'


def setup_logging(log_path):
    """Configure logging to write to file."""
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w'
    )
    
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def create_jacobian_matrix(param_rst, config_path):
    """
    Create and save Jacobian matrix for EFC control.
    
    Following notebook - no aberrations needed for jacobian.
    """
    logging.info("=== Starting Jacobian Matrix Calculation ===")
    
    # Initialize simulation
    CAPyBARA = CAPyBARAsim(param_rst['telescope'])
    CAPyBARA.get_param(param_rst, sequence='efc')
    CAPyBARA.get_grid()
    
    logging.info(f"Grid setup complete: {CAPyBARA.pupil_grid.shape}")
    
    # Create influence function
    logging.info("Creating influence function...")
    influence_function = make_xinetics_influence_functions(
        CAPyBARA.pupil_grid, 
        CAPyBARA.param['num_actuator'], 
        CAPyBARA.param['actuator_spacing']
    )
    
    CAPyBARA.get_system(influence_function)
    CAPyBARA.get_prop()
    
    # Calculate Jacobian
    logging.info("Calculating Jacobian matrix...")
    jac = rst_func.get_jacobian_matrix(CAPyBARA)
    
    logging.info(f"Jacobian shape: {jac.shape}")
    
    # Save Jacobian
    data_path = param_rst['path']['data_path']
    jacobian_name = param_rst['path']['jacobian']
    
    # Create proper filename with wavelength
    wvl = param_rst['efc']['wvl']
    if isinstance(wvl, list):
        wvl = wvl[0]
    
    date = datetime.today().strftime('%Y-%m-%d')
    filename = f"{date}_{jacobian_name}{int(wvl)}.fits"
    output_path = Path(data_path) / filename
    
    # Write to FITS
    hdu = fits.PrimaryHDU(jac)
    hdu.header['WVL'] = (wvl, 'Wavelength in nm')
    hdu.header['NACT'] = (CAPyBARA.param['num_actuator'], 'Number of actuators')
    hdu.header['DATE'] = (date, 'Creation date')
    hdu.writeto(output_path, overwrite=True)
    
    logging.info(f"Jacobian saved to: {output_path}")
    
    return str(output_path)


def run_broadband(param_rst, config_path, experiment_dir):
    """
    Run broadband EFC and observation sequence.
    
    Follows notebook exactly: Cell 13-38
    """
    logging.info("=== Starting Broadband Simulation ===")
    
    wvls = param_rst['efc']['wvl']
    logging.info(f"Wavelengths: {wvls} nm")
    
    # =================================================================
    # CELL 13: Initialize lists for multiple wavelengths
    # =================================================================
    CAPyBARA_list = []
    aberration_class_list = []
    n_zernike_coeff_list = []
    
    influence_function = None
    
    logging.info(f"\n=== Setting up {len(wvls)} wavelength channels ===")
    
    for i in range(len(wvls)):
        logging.info(f"\n--- Wavelength {i+1}: {wvls[i]} nm ---")
        
        # Deep copy parameters for this wavelength
        _param = copy.deepcopy(param_rst)
        _param['efc']['wvl'] = wvls[i]
        
        # Initialize CAPyBARA instance
        CAPyBARA = CAPyBARAsim(_param['telescope'])
        CAPyBARA.get_param(_param, sequence='efc')
        CAPyBARA.get_grid()
        
        # Create influence function (only once)
        if i == 0:
            logging.info('Calculating influence function...')
            influence_function = make_xinetics_influence_functions(
                CAPyBARA.pupil_grid,
                CAPyBARA.param['num_actuator'],
                CAPyBARA.param['actuator_spacing']
            )
        
        # Setup optical system
        CAPyBARA.get_system(influence_function)
        CAPyBARA.get_prop()
        
        # Setup aberration model
        aberration_class = rst_aberration.CAPyBARAaberration(
            sim=CAPyBARA,
            param=_param['efc']
        )
        aberration_class.set_aberration()
        
        # Get reference image with static aberrations
        CAPyBARA.get_reference_image(
            _param['efc']['wvl'] * 1e-9,
            aberration_class.static_aberration_func
        )
        
        # Create Zernike basis for quasi-static aberrations
        aberration_class.set_zernike_basis(num_mode=_param['efc']['num_mode'])
        
        logging.info(f"  ✓ System initialized")
        logging.info(f"  ✓ Reference image calculated")
        logging.info(f"  ✓ {_param['efc']['num_mode']} Zernike modes created")
        
        # Append to lists
        CAPyBARA_list.append(CAPyBARA)
        aberration_class_list.append(aberration_class)
    
    logging.info("\n✓ All wavelength channels initialized!")
    
    # =================================================================
    # CELL 14: Generating Aberration Evolution
    # =================================================================
    logging.info("\n=== Generating Aberration Evolution ===")
    
    efc_seed = 1338485434
    
    # Extract initial components from reference wavefront (NOTEBOOK WAY - no aberration_func!)
    zernike_coeff = np.zeros((_param['efc']['num_mode']))
    step0_components_array = aberration_class.extract_component(CAPyBARA.wf_ref.phase)
    starting_field = Field(
        np.dot(step0_components_array, aberration_class.zernike_basis),
        CAPyBARA.pupil_grid
    )
    
    # Generate time-varying aberrations
    logging.info("Tracking Zernike component evolution...")
    n_zernike_coeff, n_field, n_aberration = aberration_class.track_zernike_component(
        zernike_coeff=zernike_coeff,
        wvl=CAPyBARA.param['wvl'] * 1e-9,
        starting_field=starting_field,
        seed=efc_seed
    )
    
    # Apply to all wavelength channels
    for i in range(len(wvls)):
        aberration_class_list[i].get_aberration_data_cube(n_field)
    
    logging.info(f"✓ Generated {len(n_field)} aberration frames")
    
    # =================================================================
    # CELL 15: Running Broadband EFC
    # =================================================================
    logging.info("\n=== Running Broadband EFC ===")
    
    # Initialize EFC controller
    efc_exp = EFieldConjugation(CAPyBARA_list, aberration_class_list)
    
    # Convert wavelengths to meters
    wavelengths_in_meters = [wvl * 1e-9 for wvl in wvls]
    
    logging.info(f"Control wavelengths: {wvls} nm")
    logging.info(f"Number of iterations: {param_rst['efc']['num_iteration']}")
    logging.info(f"Starting EFC loop...")
    
    # Run control loop
    start_time = time.time()
    actuators_list, e_field_list, img_list, wf_lyot_list, wf_residual_list = \
        efc_exp.control(wvl=wavelengths_in_meters)
    elapsed = time.time() - start_time
    
    logging.info(f"\n✓ EFC completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logging.info(f"  Generated {len(actuators_list)} DM commands")
    logging.info(f"  Final image shape: {np.shape(img_list[-1])}")
    
    # =================================================================
    # CELL 16: Calculate broadband contrast
    # =================================================================
    weights = [1.0] * len(wvls)
    broadband_images, broadband_contrast = calculate_broadband_image_and_contrast(
        CAPyBARA_list, img_list, weights
    )
    logging.info(f"Final broadband contrast: {broadband_contrast[-1]:.2e}")
    
    # =================================================================
    # CELL 19: Save EFC products
    # =================================================================
    logging.info("Saving EFC products...")
    save_efc_products(
        output_dir=experiment_dir,
        param_efc=param_rst['efc'],
        img_list=img_list,
        actuator_list=actuators_list,
        sim_list=CAPyBARA_list,
        wvl_weights=weights,
        aberration_cube=n_field,
        seed=efc_seed,
        wavelengths_nm=wvls,
        parameters_snapshot={
            'ini_file': Path(config_path).name,
            'efc': param_rst['efc'],
            'telescope': param_rst['telescope'],
        },
    )
    
    # Check if observation sequence requested
    if not param_rst.get('sequence', {}).get('is_observation', False):
        logging.info("Observation sequence not requested, skipping.")
        return
    
    # =================================================================
    # CELL 20: Setting up Observing Sequence
    # =================================================================
    logging.info("\n=== Setting up Observing Sequence ===")
    
    obs_list = []
    obs_aberration_class_list = []
    
    # Final DM command from EFC
    current_actuators = actuators_list[-1]
    
    obs_wvls = param_rst['observation']['wvl']
    
    for i in range(len(obs_wvls)):
        _param = copy.deepcopy(param_rst)
        _param['observation']['wvl'] = obs_wvls[i]
        _wvl = _param['observation']['wvl']
        
        logging.info(f"Setting up wavelength {i+1}: {_wvl} nm")
        
        # Initialize observation system
        obs = CAPyBARAsim(_param['telescope'])
        obs.get_param(_param, sequence='observation')
        obs.get_grid()
        
        if i == 0:
            influence_function = make_xinetics_influence_functions(
                obs.pupil_grid,
                obs.param['num_actuator'],
                obs.param['actuator_spacing']
            )
        
        obs.get_system(influence_function)
        obs.get_prop()
        
        # Setup observation aberrations
        obs_aberration_class = rst_aberration.CAPyBARAaberration(
            sim=obs,
            param=_param['observation']
        )
        obs_aberration_class.set_aberration()
        obs_aberration_class.set_zernike_basis(num_mode=_param['observation']['num_mode'])
        
        # Get reference image
        obs.get_reference_image(
            _wvl * 1e-9,
            obs_aberration_class.static_aberration_func
        )
        obs.apply_actuators(actuators=current_actuators)
        
        obs_list.append(obs)
        obs_aberration_class_list.append(obs_aberration_class)
    
    logging.info("✓ Observation systems initialized!")
    
    # =================================================================
    # CELL 21: Generating observation aberration sequence
    # =================================================================
    logging.info("\nGenerating observation aberration sequence...")
    
    ref_obs_seed = 1246748186
    
    # Start from last EFC aberration (NOTEBOOK WAY!)
    starting_field = Field(
        np.dot(n_zernike_coeff[-1], aberration_class.zernike_basis),
        CAPyBARA.pupil_grid
    )
    
    # Use first obs_aberration_class for tracking (they share the same aberration evolution)
    obs_aberration_class = obs_aberration_class_list[0]
    
    # Track aberrations during observation
    n_zernike_coeff_obs, n_field_obs, n_aberration_obs = \
        obs_aberration_class.track_zernike_component(
            zernike_coeff=n_zernike_coeff[-1],
            wvl=param_rst['observation']['ref_wvl'] * 1e-9,
            starting_field=starting_field,
            seed=ref_obs_seed
        )
    
    # Apply to all observation wavelengths
    for i in range(len(obs_wvls)):
        obs_aberration_class_list[i].get_aberration_data_cube(n_field_obs)
    
    logging.info(f"✓ Generated {len(n_field_obs)} observation frames")
    
    # =================================================================
    # CELL 23: Generate probes (simplified - using DM commands directly)
    # =================================================================
    logging.info("\nSetting up probe sequence...")
    
    # Setup probe aberration
    probe_aberration = Probe(
        sim=CAPyBARA_list[0],
        param=param_rst['probe']
    )
    
    probe_aberration.set_zernike_probe_basis(starting_mode=5, ending_mode=6)

    # Define probe amplitudes
    list_of_amplitudes = [0.05e-9, 0.038e-9, 0.024e-9, 0.01e-9]  # meters
    probe_injection_list_dm1 = []
    
    logging.info(f"Generating probe sequence with {len(list_of_amplitudes)} amplitudes")
    
    # Generate probes for 2 modes, 4 amplitudes
    for j in range(len(list_of_amplitudes)):
        for i in range(30):  # 30 frames per amplitude
            probe_injection_list_dm1.append(
                probe_aberration.apply_zernike_probe(
                    amplitude=list_of_amplitudes[j],
                    n_iteration=i
                )
            )
    
    logging.info(f"✓ Generated {len(probe_injection_list_dm1)} probe frames")
    
    # Create full DM commands (DM1 with probes, DM2 zeros)
    probe_map_list = []
    for i in range(len(probe_injection_list_dm1)):
        custom_probe = np.concatenate((
            CAPyBARA_list[0].dm1.actuators + probe_injection_list_dm1[i],
            CAPyBARA_list[0].dm2.actuators
        ))
        probe_map_list.append(custom_probe)
    
    logging.info(f"✓ Probe DM commands prepared")
    
    # =================================================================
    # CELL 26: Running Reference Star Acquisition
    # =================================================================
    logging.info("\n=== Running Reference Star Acquisition ===")
    
    # Initialize observing sequence
    obs_run = ObservingSequence(obs_list, obs_aberration_class_list)
    
    # Convert wavelengths to meters
    wvl = [w * 1e-9 for w in obs_wvls]
    
    # Run WITHOUT probes
    logging.info("Acquiring reference star (without probes)...")
    wo_probe_img_list, wo_probe_e_field_list, wo_probe_wf_lyot_list, wo_probe_wf_residual_list = \
        obs_run.acquisition_loop(wvl=wvl, actuators=current_actuators)
    
    # Run WITH probes
    logging.info("Acquiring reference star (with probes)...")
    w_probe_img_list, w_probe_e_field_list, w_probe_wf_lyot_list, w_probe_wf_residual_list = \
        obs_run.acquisition_loop(wvl=wvl, actuators=probe_map_list)
    
    logging.info(f"✓ Reference star acquisition complete")
    logging.info(f"  Frames acquired: {len(wo_probe_img_list)}")
    
    # =================================================================
    # CELL 27: Save reference products
    # =================================================================
    obs_weights = [1.0] * len(obs_wvls)
    
    save_reference_products(
        output_dir=experiment_dir,
        ref_no_probes_img_list=wo_probe_img_list,
        ref_with_probes_img_list=w_probe_img_list,
        sim_list=obs_list,
        wvl_weights=obs_weights,
        dm_command=probe_map_list,
        aberration_cube=n_field_obs,
        seed=ref_obs_seed,
        param_ref=param_rst["observation"],
        wavelengths_nm=obs_wvls,
    )
    
    # =================================================================
    # CELL 28: Running Science Target Acquisition
    # =================================================================
    logging.info("\n=== Running Science Target Acquisition ===")
    
    # Function to add slew/roll offset
    def add_slew_offset(last_field, offset_rms_nm, seed, slew_aberration_class):
        """Add a static offset to simulate slew/roll."""
        opd2phase = 2 * np.pi / (1e-9 * param_rst['observation']['ref_wvl'])
        
        def normalize_phase_to_rms(phase, target_rms):
            """Normalize a phase map to target RMS value."""
            current_rms = np.sqrt(np.mean(phase**2))
            if current_rms == 0:
                return phase
            return phase * (target_rms / current_rms)
        
        d_field = normalize_phase_to_rms(last_field, offset_rms_nm * opd2phase)
        step0_components = slew_aberration_class.extract_component(last_field + d_field)
        starting_field = Field(
            np.dot(step0_components, slew_aberration_class.zernike_basis),
            CAPyBARA_list[0].pupil_grid
        )
        _, n_field_new, _ = slew_aberration_class.track_zernike_component(
            zernike_coeff=step0_components,
            wvl=CAPyBARA_list[0].param['wvl'] * 1e-9,
            starting_field=starting_field,
            seed=seed
        )
        return n_field_new
    
    # Setup slew aberration class
    slew_aberration_class = rst_aberration.CAPyBARAaberration(
        sim=CAPyBARA_list[0],
        param=param_rst['observation']
    )
    slew_aberration_class.set_aberration()
    slew_aberration_class.set_zernike_basis(num_mode=param_rst['observation']['num_mode'])
    
    # Roll A1 (first science target position)
    logging.info("\nRoll A1 (0° position)...")
    A1_seed = 1137818213
    A1_roll = 0
    slew_offset = 0.
    
    n_field_A1 = add_slew_offset(n_field_obs[-1], slew_offset, A1_seed, slew_aberration_class)
    for i in range(len(obs_aberration_class_list)):
        obs_aberration_class_list[i].get_aberration_data_cube(n_field_A1)
    
    obs_run_A1 = ObservingSequence(obs_list, obs_aberration_class_list)
    A1_img_list, _, _, _ = obs_run_A1.acquisition_loop(
        wvl=wvl, actuators=current_actuators
    )
    
    # Save A1
    save_observation_products(
        output_dir=experiment_dir,
        obs_name="science_A1",
        param_obs=param_rst["observation"],
        img_list=A1_img_list,
        actuator_list=current_actuators,
        sim_list=obs_list,
        wvl_weights=obs_weights,
        aberration_cube=n_field_A1,
        seed=A1_seed,
        roll_deg=A1_roll,
        offset_rms_nm=slew_offset,
        wavelengths_nm=obs_wvls,
        parameters_snapshot={
            "parent": "n_field_obs[-1]",
            "roll": "A1",
        },
    )
    
    # Roll B1 (rotated 26°)
    logging.info("Roll B1 (26° position)...")
    B1_seed = 1118375116
    B1_roll = 26
    n_field_B1 = add_slew_offset(n_field_A1[-1], slew_offset, B1_seed, slew_aberration_class)
    for i in range(len(obs_aberration_class_list)):
        obs_aberration_class_list[i].get_aberration_data_cube(n_field_B1)
    
    obs_run_B1 = ObservingSequence(obs_list, obs_aberration_class_list)
    B1_img_list, _, _, _ = obs_run_B1.acquisition_loop(
        wvl=wvl, actuators=current_actuators
    )
    
    save_observation_products(
        output_dir=experiment_dir,
        obs_name="science_B1",
        param_obs=param_rst["observation"],
        img_list=B1_img_list,
        actuator_list=current_actuators,
        sim_list=obs_list,
        wvl_weights=obs_weights,
        aberration_cube=n_field_B1,
        seed=B1_seed,
        roll_deg=B1_roll,
        offset_rms_nm=slew_offset,
        wavelengths_nm=obs_wvls,
        parameters_snapshot={
            "parent": "n_field_A1[-1]",
            "roll": "B1",
        },
    )
    
    # Roll A2 (back to 0°)
    logging.info("Roll A2 (return to 0°)...")
    A2_seed = 1208545444
    A2_roll = 0
    n_field_A2 = add_slew_offset(n_field_B1[-1], slew_offset, A2_seed, slew_aberration_class)
    for i in range(len(obs_aberration_class_list)):
        obs_aberration_class_list[i].get_aberration_data_cube(n_field_A2)
    
    obs_run_A2 = ObservingSequence(obs_list, obs_aberration_class_list)
    A2_img_list, _, _, _ = obs_run_A2.acquisition_loop(
        wvl=wvl, actuators=current_actuators
    )
    
    save_observation_products(
        output_dir=experiment_dir,
        obs_name="science_A2",
        param_obs=param_rst["observation"],
        img_list=A2_img_list,
        actuator_list=current_actuators,
        sim_list=obs_list,
        wvl_weights=obs_weights,
        aberration_cube=n_field_A2,
        seed=A2_seed,
        roll_deg=A2_roll,
        offset_rms_nm=slew_offset,
        wavelengths_nm=obs_wvls,
        parameters_snapshot={
            "parent": "n_field_B1[-1]",
            "roll": "A2",
        },
    )
    
    # Roll B2 (back to 26°)
    logging.info("Roll B2 (return to 26°)...")
    B2_seed = 1286514517
    B2_roll = 26
    n_field_B2 = add_slew_offset(n_field_A2[-1], slew_offset, B2_seed, slew_aberration_class)
    for i in range(len(obs_aberration_class_list)):
        obs_aberration_class_list[i].get_aberration_data_cube(n_field_B2)
    
    obs_run_B2 = ObservingSequence(obs_list, obs_aberration_class_list)
    B2_img_list, _, _, _ = obs_run_B2.acquisition_loop(
        wvl=wvl, actuators=current_actuators
    )
    
    save_observation_products(
        output_dir=experiment_dir,
        obs_name="science_B2",
        param_obs=param_rst["observation"],
        img_list=B2_img_list,
        actuator_list=current_actuators,
        sim_list=obs_list,
        wvl_weights=obs_weights,
        aberration_cube=n_field_B2,
        seed=B2_seed,
        roll_deg=B2_roll,
        offset_rms_nm=slew_offset,
        wavelengths_nm=obs_wvls,
        parameters_snapshot={
            "parent": "n_field_A2[-1]",
            "roll": "B2",
        },
    )
    
    logging.info("\n✓ All science rolls acquired!")
    logging.info("  A1 (0°):  120 frames")
    logging.info("  B1 (26°): 120 frames")
    logging.info("  A2 (0°):  120 frames")
    logging.info("  B2 (26°): 120 frames")
    
    logging.info("Broadband simulation complete!")


def main(mode, config_file):
    """
    Main entry point for CAPyBARA simulations.
    """
    # Load configuration
    param_rst = read_ini_file(config_file)
    
    # Setup paths
    data_path = Path(param_rst['path']['data_path'])
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Update jacobian path
    param_rst['jacobian'] = str(
        data_path / param_rst['path']['jacobian']
    )
    
    if mode == 'jacobian':
        # Simple jacobian mode - no experiment directory needed
        log_path = data_path / 'jacobian_creation.log'
        setup_logging(log_path)
        
        logging.info("CAPyBARA Jacobian Matrix Creation")
        logging.info(f"Config file: {config_file}")
        
        jac_path = create_jacobian_matrix(param_rst, config_file)
        
        logging.info(f"Jacobian matrix saved to: {jac_path}")
        logging.info("Jacobian creation complete!")
        
    elif mode in ['mono', 'broadband']:
        logging.warning(f"CAPyBARA {mode.upper()} - Routine in this script is not fully debugged.")
        
        # Create experiment directory for full simulations
        experiment_dir = create_experiment_dir(data_path)
        
        # Setup logging in experiment directory
        log_path = experiment_dir / 'capybara_run.log'
        setup_logging(log_path)
        
        logging.info(f"CAPyBARA {mode.upper()} Simulation")
        logging.info(f"Config file: {config_file}")
        logging.info(f"Experiment directory: {experiment_dir}")
        
        # Copy metadata files
        copy_run_metadata(
            output_dir=experiment_dir,
            ini_path=config_file,
            log_path=None
        )
        
        # Run simulation
        if mode == 'mono':
            logging.error("Monochromatic mode not yet implemented in this version.")
            logging.info("Please use --mode broadband")
            sys.exit(1)
        else:  # broadband
            run_broadband(param_rst, config_file, experiment_dir)
        
        # Copy final log
        if log_path.exists():
            import shutil
            shutil.copy2(log_path, experiment_dir / log_path.name)
        
        logging.info(f"All outputs saved to: {experiment_dir}")
        logging.info("Simulation complete!")
        
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'jacobian', 'mono', or 'broadband'")


if __name__ == "__main__":
    # Print banner
    load_and_print_default_txt()
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Run CAPyBARA simulation following notebook workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create Jacobian matrix
  python run_CAPyBARA.py --mode jacobian --config capy-pup-900.ini
  
  # Run broadband EFC + observations (full workflow from notebook)
  python run_CAPyBARA.py --mode broadband --config capy-pup-900.ini
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['jacobian', 'mono', 'broadband'],
        help='Simulation mode'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to INI configuration file'
    )
    
    args = parser.parse_args()
    
    # Track execution time
    start_time = time.time()
    
    try:
        main(mode=args.mode, config_file=args.config)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)