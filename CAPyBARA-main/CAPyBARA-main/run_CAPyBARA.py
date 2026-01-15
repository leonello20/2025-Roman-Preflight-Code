import matplotlib.pyplot as plt
import numpy as np

import CAPyBARA.aberration as rst_aberration
from hcipy import *
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
#import astropy.io
#from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from CAPyBARA.rst_design import _make_rst_aperture, _make_lyot_mask
from CAPyBARA.rst_functions import *
#import matplotlib.colors as mpl
import scipy.fftpack
import os
import sys
plt.rcParams['image.origin'] = 'lower'

from CAPyBARA import CAPyBARAsim

import CAPyBARA.rst_functions as rst_func
from CAPyBARA.plotting import plt_field, plt_std_vs_iteration
from CAPyBARA.efc import EFieldConjugation 

from CAPyBARA.observing_sequence import ObservingSequence 
import CAPyBARA.plotting  as plotting

import time

import yaml
import argparse

import copy  # Use the copy module to perform deep copies

from datetime import datetime

import CAPyBARA.utils as utils
from CAPyBARA.utils import read_ini_file, load_and_print_default_txt, save_output
# Set matplotlib style
plt.style.use(astropy_mpl_style)
plt.rcParams['image.origin'] = 'lower'

# TODO - make the simulation save the parameter used during the simulation as .ymal

# TODO - change it to control wvl 

# TODO - check if these are being handle by the get_system in CaPy

# TODO - continue refactoring 

# TODO - add unit tests in here before refactoring

import os

def run_efc_simulation (CAPyBARA, aberration_class, param, save=True, path=None):
    """
    Wrapper around setting up and running efc control


    Args:
        CAPyBARA (_type_): _description_
        aberration_class (_type_): _description_
        param (_type_): _description_
        save (bool, optional): _description_. Defaults to True.
        path (_type_, optional): _description_. Defaults to None.
    """
    # extract component from the wavefront 
    step0_components_array = aberration_class.extract_component(aberration_class.aberration_func(CAPyBARA.wf_ref)) 

    # Project the zernike basis and coeff. back to wavefront
    field0 = Field(np.dot(step0_components_array, aberration_class.zernike_basis), CAPyBARA.pupil_grid)

    if param_rst['aberration']['chromaticity'] is True:
        wvl_aberration = CAPyBARA.param['ref_wvl']
    else: 
        wvl_aberration = 556 #CAPyBARA.param['wvl']

    updated_wf, updated_zernike_coeff, n_aberration = aberration_class.apply_perturbation_to_wavefront(step0_components_array, wvl_aberration, seed=1)

    n_zernike_coeff, n_field, n_aberration = aberration_class.track_zernike_component(step0_components_array, wvl_aberration, field0)

    aberration_class.get_aberration_data_cube(n_field)

    #%% Running the control
    efc_exp = EFieldConjugation(CAPyBARA, aberration_class)
    actuator_list, e_field_list, img_list, wf_lyot_list, wf_post_lyot_list, wf_list = efc_exp.control(wvl=[CAPyBARA.param['wvl']])

    if save is True: 
        date = datetime.today().strftime('%Y-%m-%d-%H-%M')
        efc_path = path + date + '_' + efc_exp.name
        save_output(CAPyBARA, param_rst['efc'], img_list, actuator_list, efc_path)

    # Get the average contrast
    print(f'Checking the shape of the image_list {np.shape(img_list)}')
    # average_contrast = rst_func.get_average_contrast(CAPyBARA,img_list, is_mono=True)

    # print(f'Average contrast: {average_contrast}')      



    pass



def run_observing_sequence ():


    pass

# def run_broadband_efc 

# def run_broad_observing_sequence 

# def calculate_jacobian 

def main(mode: str, config_file: str) -> None:
    """
    Main function to run the CAPyBARA simulation.
    
    Parameters
    ----------
    mode : str
        Observation mode ('mono' for monochromatic, 'broadband' for broadband).
    config_file : str
        Path to the .ini configuration file.
    """
    print('Running CAPYBARA Simulation')

    # Load parameters from the .ini configuration file
    param_rst = read_ini_file(config_file)
    path = param_rst['path']['data_path']

    save = True
    
    # Determine the observing sequence and run the simulation based on mode
    if mode == 'mono':
        # Initialise simulation
        if param_rst['sequence']['is_efc'] and param_rst['sequence']['is_observation'] is True:
            # TODO - check the wvl whrther everyhing is in nm or m 

            # First setup the system for CaPYBARA - set up an army of them
            param_rst['jacobian'] = param_rst['path']['data_path']+param_rst['path']['jacobian']

            # initialise the system setup  
            CAPyBARA = CAPyBARAsim(param_rst['telescope'])
            CAPyBARA.get_param(param_rst,sequence='efc') 
            CAPyBARA.get_grid()

            # make the influence function
            influence_function = make_xinetics_influence_functions(CAPyBARA.pupil_grid, CAPyBARA.param['num_actuator'], CAPyBARA.param['actuator_spacing'])
            CAPyBARA.get_system(influence_function)

            # setup the propagation
            CAPyBARA.get_prop()

            CAPyBARA.get_reference_image(wvl=CAPyBARA.param['wvl']*1e-9, static_aberration_func=None, wavefront_error=None, check=False)

            # setup the aberration instance
            aberration_class = rst_aberration.CAPyBARAaberration(sim=CAPyBARA, param=param_rst['efc'])
            aberration_class.set_aberration()
            aberration_class.set_zernike_basis(num_mode=param_rst['efc']['num_mode'])

            try:
                phase_data = aberration_class.aberration_func(CAPyBARA.wf_ref)
            except AttributeError:
                # Fallback if the method isn't in the class yet
                phase_data = CAPyBARA.wf_ref.phase

                # 2. Extract Zernike coefficients from that phase
                step0_components_array = aberration_class.extract_component(phase_data)

                # 3. Project back to the grid
                # Note: np.dot is used to multiply the coefficients by the basis modes
                projected_phase = np.dot(step0_components_array, aberration_class.zernike_basis)
                field0 = Field(projected_phase, CAPyBARA.pupil_grid)

            # extract component from the wavefront 
            step0_components_array = aberration_class.extract_component(aberration_class.aberration_func(CAPyBARA.wf_ref)) 

            # Project the zernike basis and coeff. back to wavefront
            field0 = Field(np.dot(step0_components_array, aberration_class.zernike_basis), CAPyBARA.pupil_grid)

            if param_rst['aberration']['chromaticity'] is True:
                wvl_aberration = CAPyBARA.param['ref_wvl']
            else: 
                wvl_aberration = 556 #CAPyBARA.param['wvl']

            updated_wf, updated_zernike_coeff, n_aberration = aberration_class.apply_perturbation_to_wavefront(step0_components_array, wvl_aberration, seed=1)

            n_zernike_coeff, n_field, n_aberration = aberration_class.track_zernike_component(step0_components_array, wvl_aberration, field0)

            aberration_class.get_aberration_data_cube(n_field)

            #%% Running the control

            efc_exp = EFieldConjugation(CAPyBARA, aberration_class)
            actuator_list, e_field_list, img_list, wf_list, wf_post_lyot_list = efc_exp.control(wvl=[CAPyBARA.param['wvl']])

            #%% Save efc output 
            # TODO - move it somewhere else 
            if save is True: 
                date = datetime.today().strftime('%Y-%m-%d-%H-%M')
                efc_path = path + date + '_' + efc_exp.name
                save_output(CAPyBARA, param_rst['efc'], img_list, actuator_list, efc_path)

            # Get the average contrast
            print(f'Checking the shape of the image_list {np.shape(img_list)}')
            # average_contrast = rst_func.get_average_contrast(CAPyBARA,img_list, is_mono=True)
            
            # print(f'Average contrast: {average_contrast}')             

            #%% Monochromatic observing sequence
            print('Start Monochromatic Observing Sequence')

            # set up a new aberration for reference star 
            obs_aberration_class = rst_aberration.CAPyBARAaberration(sim=CAPyBARA, param=param_rst['observation'])

            obs_aberration_class.set_aberration()
            obs_aberration_class.set_zernike_basis(num_mode=param_rst['observation']['num_mode'])

            # If the observing wvl has been changed, get to the new wvl, and get a new reference image 
            CAPyBARA.get_reference_image(wvl=param_rst['observation']['wvl']*1e-9, static_aberration_func=None, check=True)
            
            print(f'What is the shape of the actuator? {np.shape(actuator_list[-1])}')

            last_wf_from_efc = CAPyBARA.create_wavefront(wvl=param_rst['observation']['wvl']*1e-9, current_aberration=aberration_class.aberration_cube[-1])

            print('Try to send the dm shapes and wf to observing sequence')
            CAPyBARA.get_coronagrphic_image(wf_post_lyot_list[-1][0], check=True)

            _img = CAPyBARA.get_image(aberration_class.aberration_cube[-1], wvl= param_rst['observation']['wvl']*1e-9, actuators=actuator_list[-1], include_aberration=True, wf=last_wf_from_efc)

            # upto here, this is still fine. Can confirm the functions are passed properly

            step0_components_array = obs_aberration_class.extract_component(obs_aberration_class.aberration_func(last_wf_from_efc))

            field0 = Field(np.dot(step0_components_array, obs_aberration_class.zernike_basis), CAPyBARA.pupil_grid)

            if param_rst['aberration']['chromaticity'] is True:
                wvl_aberration = CAPyBARA.param['ref_wvl']
            else: 
                wvl_aberration = 556 #CAPyBARA.param['wvl']

            updated_wf, updated_zernike_coeff, n_aberration = obs_aberration_class.apply_perturbation_to_wavefront(step0_components_array, wvl_aberration, seed=10)

            n_zernike_coeff, n_field, n_aberration = obs_aberration_class.track_zernike_component(step0_components_array, wvl_aberration, field0)

            obs_aberration_class.get_aberration_data_cube(n_field)

            # wvl for the EFC is different from the science acquistion
            OS_class = ObservingSequence(CAPyBARA, obs_aberration_class)

            # CHECK - see if I have the same image or not
            # obs_aberration_class.aberration_cube[0] = aberration_class.aberration_cube[-1]

            # CHECK - Here to check if I can get back to the dark hole or not
            ref_psf_list, ref_dm, wf_lyot_list, wf_residual_list = OS_class.acquisition_loop(wvl=[param_rst['observation']['wvl']], actuators=actuator_list[-1])

            # obs_aberration_class.aberration_cube

            print(f'Check the shape {np.shape(ref_psf_list)} {type(ref_psf_list)}')

            CAPyBARA.param['wvl'] = param_rst['observation']['wvl']

            if save is True: 
                date = datetime.today().strftime('%Y-%m-%d-%H-%M')
                obs_path = path + date + '_' + OS_class.name
                save_output(CAPyBARA, param_rst['observation'], ref_psf_list, ref_dm, obs_path)

        elif param_rst['sequence']['is_efc'] is True and  param_rst['sequence']['is_observation'] is False: 
            print('Run EFC Only (Not implenmented yet)')

        elif param_rst['sequence']['is_efc'] is False and  param_rst['sequence']['is_observation'] is True: 
            print('Run observing Only (Not implenmented yet)')

        print('End')

    elif mode == 'broadband':
        experiment = 'BroadbandEFC'
        print('Try Broadband')
        
        # First setup the system for CaPYBARA - set up an army of them
        param_rst['jacobian'] = param_rst['data_path']+param_rst['jacobian']

        CAPyBARA_list = []
        aberration_class_list = []
        obs_aberration_class_list = []

        for i in range(len(param_rst['efc']['wvl'])):
            _param = copy.deepcopy(param_rst) # Correct: call the copy method to create a new dictionary
            _param['efc']['wvl'] = param_rst['efc']['wvl'][i]  # Now this works

            # Setup the system
            # initialise the system setup  
            CAPyBARA = CAPyBARAsim(_param['telescope'])

            CAPyBARA.get_param(_param,sequence='efc') 
            CAPyBARA.get_grid()

            if i == 0: 
                print('Calculating influence function')
                influence_function = make_xinetics_influence_functions(CAPyBARA.pupil_grid, CAPyBARA.param['num_actuator'], CAPyBARA.param['actuator_spacing'])

            CAPyBARA.get_system(influence_function)
            CAPyBARA.get_prop()

            # setup the aberration instance
            aberration_class = rst_aberration.CAPyBARAaberration(sim=CAPyBARA, param=_param['efc'])
            aberration_class.set_aberration()
            aberration_class.set_zernike_basis(num_mode=_param['efc']['num_mode'])

            CAPyBARA.get_reference_image(wvl=CAPyBARA.param['wvl']*1e-9)

            # extract component from the wavefront 
            step0_components_array = aberration_class.extract_component(aberration_class.aberration_func(CAPyBARA.wf_ref)) 

            # Project the zernike basis and coeff. back to wavefront
            field0 = Field(np.dot(step0_components_array, aberration_class.zernike_basis), CAPyBARA.pupil_grid)

            if param_rst['aberration']['chromaticity'] is True:
                wvl_aberration = CAPyBARA.param['ref_wvl']
            else: 
                wvl_aberration = 556 #CAPyBARA.param['wvl']

            updated_wf, updated_zernike_coeff, n_aberration = aberration_class.apply_perturbation_to_wavefront(step0_components_array, wvl_aberration, seed=1)

            n_zernike_coeff, n_field, n_aberration = aberration_class.track_zernike_component(step0_components_array, wvl_aberration, field0)

            aberration_class.get_aberration_data_cube(n_field)


            CAPyBARA_list.append(CAPyBARA)
            aberration_class_list.append(aberration_class)

        #%% Running the control
        efc_exp = EFieldConjugation(CAPyBARA_list, aberration_class_list)

        actuator_list, e_field_list, img_list, wf_lyot_list, wf_residual_list, wf_list = efc_exp.control(wvl=param_rst['efc']['wvl'])

        # average_contrast = rst_func.get_average_contrast(CAPyBARA_list,img_list)

        if save is True: 
            date = datetime.today().strftime('%Y-%m-%d-%H-%M')
            efc_path = path + date + '_' + efc_exp.name
            save_output(CAPyBARA_list, param_rst['efc'], img_list, actuator_list, efc_path)
            
        #%% observing
        obs_aberration_class_list = []

        # TODO - check the OPD which I am injecting per frame

        for i in range(len(param_rst['observation']['wvl'])):
           
            _param = copy.deepcopy(param_rst) # Correct: call the copy method to create a new dictionary
            _param['observation']['wvl'] = param_rst['observation']['wvl'][i]  # Now this works

            # set up a new aberration for reference star 
            obs_aberration_class = rst_aberration.CAPyBARAaberration(sim=CAPyBARA_list[i], param=_param['observation'])

            obs_aberration_class.set_aberration()
            obs_aberration_class.set_zernike_basis(num_mode=_param['observation']['num_mode'])
            
            last_wf_from_efc = CAPyBARA_list[i].create_wavefront(wvl=_param['observation']['wvl']*1e-9, current_aberration=aberration_class_list[i].aberration_cube[-1])

            # upto here, this is still fine. Can confirm the functions are passed properly
            step0_components_array = obs_aberration_class.extract_component(obs_aberration_class.aberration_func(last_wf_from_efc))

            field0 = Field(np.dot(step0_components_array, obs_aberration_class.zernike_basis), CAPyBARA_list[i].pupil_grid)

            if param_rst['aberration']['chromaticity'] is True:
                wvl_aberration = CAPyBARA.param['ref_wvl']
            else: 
                wvl_aberration = 556 #CAPyBARA.param['wvl']

            # last_n_field = n_field[-1]
            # array = np.tile(last_n_field, (np.shape(n_field)[0], 1))

            updated_wf, updated_zernike_coeff, n_aberration = obs_aberration_class.apply_perturbation_to_wavefront(step0_components_array, wvl_aberration, seed=10)
            n_zernike_coeff, n_field, n_aberration = obs_aberration_class.track_zernike_component(step0_components_array, wvl_aberration, field0)

            obs_aberration_class.get_aberration_data_cube(n_field)

            obs_aberration_class.aberration_cube[0] = aberration_class.aberration_cube[-1]  # Directly copy the last EFC aberration

            # CHECK - Same aberration cube as the efc
            obs_aberration_class_list.append(obs_aberration_class)

        # wvl for the EFC is different from the science acquistion
        OS_class = ObservingSequence(CAPyBARA_list, obs_aberration_class_list)

        # CHECK - see if I have the same image or not
        # obs_aberration_class.aberration_cube[0] = aberration_class.aberration_cube[-1]

        ref_psf_list, ref_dm = OS_class.accquisition_loop(wvl=param_rst['observation']['wvl'], aberration_sequence=
        OS_class.aberration_cube, last_dm_command=actuator_list[-1])

        # obs_aberration_class.aberration_cube

        print(f'Check the shape {np.shape(ref_psf_list)} {type(ref_psf_list)}')

        CAPyBARA.param['wvl'] = param_rst['observation']['wvl']

        if save is True: 
            date = datetime.today().strftime('%Y-%m-%d-%H-%M')
            obs_path = path + date + '_' + OS_class.name
            save_output(CAPyBARA_list, param_rst['observation'], ref_psf_list, ref_dm, obs_path)

    elif mode == 'jacobian':
        # Jacobian matrix calculation mode
        print('Run Jacobian')

        print(f'Where to save jac? {path}')

        CAPyBARA = CAPyBARAsim(param_rst['telescope'])
        CAPyBARA.get_param(param_rst,sequence='efc') # initialise the system setup  
        CAPyBARA.get_grid()
        influence_function = make_xinetics_influence_functions(CAPyBARA.pupil_grid, CAPyBARA.param['num_actuator'], CAPyBARA.param['actuator_spacing'])
        CAPyBARA.get_system(influence_function)
        CAPyBARA.get_prop()
        jac  = rst_func.get_jacobian_matrix(CAPyBARA)
        utils.write2fits(jac, key='jacobian', wvl=CAPyBARA.param['wvl'] ,path=path)
    else:
        print('Invalid mode specified.')

    print('Simulation completed.')


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run CAPyBARA simulation with specified mode and config file.")
    parser.add_argument("--mode", type=str, required=True, help="Mode in which the script runs (mono, broadband, jacobian)")
    parser.add_argument("--config", type=str, required=True, help="Path to the .ini configuration file")

    # Parse the arguments
    args = parser.parse_args()
    load_and_print_default_txt()

    # Track the start time
    start_time = time.time()

    # Call the main function with mode and config file
    main(mode=args.mode, config_file=args.config)

    print("--- %s seconds ---" % (time.time() - start_time))