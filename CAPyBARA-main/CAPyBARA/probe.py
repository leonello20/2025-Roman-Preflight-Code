import numpy as np
import matplotlib.colors as mpl
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
from astropy.io import fits
import scipy.fftpack
from hcipy import *
import CAPyBARA.mode_basis as zernike
import CAPyBARA.utils as utils
from CAPyBARA.logging_config import setup_logging
import logging

class Probe:
    setup_logging()
    log = logging.getLogger(__name__)

    def __init__(self, param, sim):
        self.param = param
        self.sim = sim

    def set_zernike_probe_basis (self, starting_mode, ending_mode):
        """
        Getting the basis for the zernike probes

        Args:
            starting_mode (int): starting zernike mode in noll index
            ending_mode (int): ending zernike mode in noll index
        """
        self._starting_mode = starting_mode
        self._ending_mode = ending_mode
        num_mode = ending_mode - starting_mode + 1

        # make zernike modes with unit of metres 
        actutor_grid = make_pupil_grid(48)
        zernike_basis = make_zernike_basis(num_mode, D=47/48, grid=actutor_grid, starting_mode=starting_mode)

        aperture = make_circular_aperture(diameter=47/48)(actutor_grid)

        # using the aperture to make the modes orthogonal
        orthogonal_modes = zernike.make_custom_orthogonal_modes(zernike_basis, aperture)

        # Convert to a ModeBasis object
        self.probe_zernike_basis = ModeBasis(orthogonal_modes, actutor_grid)
        self.probe_rms_modes = get_rms_modes(orthogonal_modes, aperture)

        zernike_coeff = np.random.randn(len(self.probe_zernike_basis))
        zernike_coeff /= np.sqrt(len(self.probe_zernike_basis))

        self.zernike_coeff = zernike_coeff

    def get_custom_probe (self, input_probe, n_iteration, amplitude=1e-9, key=None, current_mode_index=None):
        if key is None: 
            self.log.info('No probe, iteration %d', n_iteration)
            output_probe = input_probe*0

        elif key == '+':
            self.log.info('Injecting %s probe, iteration %d', key, n_iteration)
            
            if current_mode_index is None: 
                output_probe = input_probe*amplitude
            else: 
                self.log.info('Injecting %s probe, iteration %d to mode %d', key, n_iteration, current_mode_index)
                output_probe = np.zeros_like(input_probe)
                output_probe[current_mode_index] = input_probe[current_mode_index] * amplitude
        elif key == '-': 
            self.log.info('Injecting %s probe, iteration %d', key, n_iteration)
            output_probe = np.zeros_like(input_probe)
            output_probe[current_mode_index] = input_probe[current_mode_index] * amplitude

        return output_probe

    def generate_zernike_probe (self, n_iteration, amplitude=1e-9, key=None, current_mode_index=None):
        if key is None:
            self.log.info('No probe, iteration %d', n_iteration)
            zernike_coeff = self.zernike_coeff.copy()
            zernike_coeff *= 0 # zero out coefficients
        
        elif key == '+':
            self.log.info('Injecting %s probe, iteration %d', key, n_iteration)
            _zernike_coeff = self.zernike_coeff.copy()

            if current_mode_index is None:
                zernike_coeff = _zernike_coeff*amplitude
            else: 
                self.log.info('Injecting %s probe, iteration %d to mode %d', key, n_iteration, current_mode_index)
                _zernike_coeff *= 0
                _zernike_coeff[current_mode_index] = amplitude
                zernike_coeff = _zernike_coeff
        
        elif key == '-':
            self.log.info('Injecting %s probe, iteration %d', key, n_iteration)
            _zernike_coeff = self.zernike_coeff.copy()

            if current_mode_index is None:
                zernike_coeff  = _zernike_coeff*(-amplitude)
            else:
                _zernike_coeff *= 0
                _zernike_coeff[current_mode_index] = -amplitude
                zernike_coeff = _zernike_coeff
                
        return zernike_coeff

    def generate_single_mode_probe (self, amplitude, n_iteration):
        sub_amplitude = [1, 0, -1]
        iterations_per_sub_mode = int(self.param['num_probe_iteration'])
        total_interation_per_mode = iterations_per_sub_mode * len(sub_amplitude)

        # Determine the current mode and sub-amplitude based on self.n
        current_mode_index = n_iteration // total_interation_per_mode  # Which mode we are in (0-based)

        # Determine the sub-amplitude index within the current mode
        mode_iteration = n_iteration % total_interation_per_mode  # Iteration index within the mode
        sub_amplitude_index = mode_iteration // iterations_per_sub_mode

        is_zernike = True if self.param['probe_mode'] == 'zernike' else False

        if is_zernike is True:
            current_mode = self._starting_mode + current_mode_index  # Mode numbers start from 4
        else: 
            current_mode = current_mode_index
    
        current_amplitude = sub_amplitude[sub_amplitude_index]

        if is_zernike is True:
            # Perform the injection
            coeff = self.generate_zernike_probe(n_iteration, amplitude, key={1: '+', 0: None, -1: '-'}[current_amplitude], current_mode_index=current_mode_index)
        else:
            _probe = self.get_custom_probe(amplitude, n_iteration)
            if _probe is None:
                raise Exception('Custom mode not implemented yet.')

        return coeff 

    def apply_zernike_probe(self, amplitude, n_iteration):
        """Applies a Zernike probe based on iteration and probe mode settings."""
        
        # TODO: Check the bug in here
        if self.param['probe_mode'] == 'zernike' and self.param['is_single'] is True:
            self.log.info("Inject single-mode Zernike probe")
            zernike_coeff = self.generate_single_mode_probe(amplitude, n_iteration)

        else:
            iteration_within_combined = (
                n_iteration % (self.param['num_probe_iteration'] * 3)
            )

            if iteration_within_combined < self.param['num_probe_iteration']:
                self.log.info("Injecting combined Zernike + probe")
                zernike_coeff = self.generate_zernike_probe(
                    key='+', amplitude=amplitude, n_iteration=n_iteration
                )
            elif iteration_within_combined < 2 * self.param['num_probe_iteration']:
                self.log.info("Injecting combined 0 probe")
                zernike_coeff = self.generate_zernike_probe(
                    key=None, amplitude=amplitude, n_iteration=n_iteration
                )
            else:
                self.log.info("Injecting combined - probe")
                zernike_coeff = self.generate_zernike_probe(
                    key='-', amplitude=amplitude, n_iteration=n_iteration
                )

        mode_inject = zernike_coeff / self.probe_rms_modes
        delta_actuator = self.probe_zernike_basis.transformation_matrix.dot(mode_inject)

        return delta_actuator

    def apply_custom_probe(self, amplitude, probe_inputs, n_iteration):
        iteration_within_combined = n_iteration % (self.param['num_probe_iteration'] * 3)

        self.log.info('Injecting combined mode probe')

        if iteration_within_combined < self.param['num_probe_iteration']:
            self.log.info('Injecting combined + probe')
            multiplier = 1
        elif iteration_within_combined < 2 * self.param['num_probe_iteration']:
            self.log.info('Injecting combined 0 probe')
            multiplier = 0
        else:
            self.log.info('Injecting combined - probe')
            multiplier = -1

        # Apply the multiplier to each element of probe_inputs
        modified_probe_inputs = [amplitude * input_value * multiplier for input_value in probe_inputs]

        return modified_probe_inputs

    def inject_probe (self, amplitude, n_interation):
        # TODO - check the bug in here
        if self.param['probe_mode'] == 'zernike' and self.param['is_single'] == True:
            print('Injecting single mode probe')
            zernike_coeff = self.get_single_mode_probe(amplitude, n_interation)

        else:
            iteration_within_combined = n_interation % (self.param['num_probe_iteration'] * 3)
            print('Injecting combined mode probe', self.param['probe_mode'], self.param['is_single'])

            if iteration_within_combined < self.param['num_probe_iteration']:
                print('Injecting combined + probe')
                zernike_coeff = self.get_zernike_probe(key='+', amplitude=amplitude, n_interation=n_interation)
            elif iteration_within_combined < 2 * self.param['num_probe_iteration']:
                print('Injecting combined 0 probe')
                zernike_coeff = self.get_zernike_probe(key=None, amplitude=amplitude, n_interation=n_interation)
            else:
                print('Injecting combined - probe')
                zernike_coeff = self.get_zernike_probe(key='-', amplitude=amplitude, n_interation=n_interation)

        mode_inject = zernike_coeff / self.probe_rms_modes
        delta_actuator = self.probe_zernike_basis.transformation_matrix.dot(mode_inject)

        return delta_actuator

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