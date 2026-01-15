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
from CAPyBARA.mode_basis import CustomBasis

class CAPyBARAaberration:
    # TODO - change nzernikumber of iteration in here and check the modes
    def __init__(self, sim, param, tip_tilt_focus = None, is_ref=False, custom_basis=None):
        self.sim = sim
        self.zernike_basis = None
        self.param = param
        self.tip_tilt_focus = tip_tilt_focus
        self.custom_basis = custom_basis

        if is_ref is False: 
            self._opd2phase = 2 * np.pi / (self.sim.param['wvl']*1e-9)
        else:
            self._opd2phase = 2 * np.pi / (self.sim.param['ref_wvl']*1e-9)

        self._starting_mode = 4

    def set_zernike_basis (self, num_mode):
        # make zernike modes with unit of metres 
        zernike_basis = make_zernike_basis(num_mode, self.sim.param['diameter'], self.sim.pupil_grid, starting_mode = 4)

        # using the aperture to make the modes orthogonal
        orthogonal_modes = zernike.make_custom_orthogonal_modes(zernike_basis, self.sim.aperture)
       
        # Convert to a ModeBasis object
        self.zernike_basis = ModeBasis(orthogonal_modes, self.sim.pupil_grid)
        self.rms_modes = utils.get_rms_modes(orthogonal_modes, self.sim.aperture)

    def set_aberration(self, seed=42): 
        # TODO - move this function to aberration
        if seed is None:
            seed = 42
            print(f'Seed = {seed}')

        __tip_tilt_focus = make_zernike_basis(3, self.sim.param['diameter'], self.sim.pupil_grid, starting_mode = 2)
        _tip_tilt_focus = zernike.make_custom_orthogonal_modes(__tip_tilt_focus, self.sim.aperture)

        self.tip_tilt_focus = ModeBasis(_tip_tilt_focus, self.sim.pupil_grid)

        # in metres # TODO - change the input value 4*17e-9 
        self.static_aberration_func = SurfaceAberration(self.sim.pupil_grid, 4*17*1e-9, self.sim.param['diameter'], remove_modes = self.tip_tilt_focus, exponent = -3, seed = seed)

    def extract_component(self, phase):
        return self.zernike_basis.coefficients_for(phase)

    def set_zernike_probe_basis (self, starting_mode, ending_mode, seed=1):
        num_mode = ending_mode - starting_mode + 1

        # make zernike modes with unit of metres 
        actutor_grid = make_pupil_grid(48)
        zernike_basis = make_zernike_basis(num_mode, D = 47/48, grid = actutor_grid, starting_mode = starting_mode)

        aperture = make_circular_aperture(diameter=47/48)(actutor_grid)

        # using the aperture to make the modes orthogonal
        orthogonal_modes = zernike.make_custom_orthogonal_modes(zernike_basis, aperture)

        # Convert to a ModeBasis object
        self.probe_zernike_basis = ModeBasis(orthogonal_modes, actutor_grid)
        self.probe_rms_modes = utils.get_rms_modes(orthogonal_modes, aperture)

        zernike_coeff = np.random.randn(len(self.probe_zernike_basis))
        zernike_coeff /= np.sqrt(len(self.probe_zernike_basis))

        self.zernike_coeff = zernike_coeff

    def get_probe (self, n_interation, amplitude=1e-9, key=None, current_mode_index=None):

        if key is None: 
            print('No probe, iteration')
            # return non? 
        elif key is '+':
            print('Injecting + probe')
            # return ?? 
        elif key is '-': 
            print('inject -  probe')

            # return ?? 

    def get_zernike_probe (self, n_interation, amplitude=1e-9, key=None, current_mode_index=None):
        if key is None:
            print('No probe, iteration')
            zernike_coeff = self.zernike_coeff.copy()
            zernike_coeff *= 0 # zero out coefficients
        
        elif key == '+':
            print('Injecting + probe')
            _zernike_coeff = self.zernike_coeff.copy()

            if current_mode_index is None:
                zernike_coeff = _zernike_coeff*amplitude
            else: 
                print(f'Injecting + probe to mode {current_mode_index}')
                _zernike_coeff *= 0
                _zernike_coeff[current_mode_index] = amplitude
                zernike_coeff = _zernike_coeff
        
        elif key == '-':
            print('Injecting - probe')
            _zernike_coeff = self.zernike_coeff.copy()

            if current_mode_index is None:
                zernike_coeff  = _zernike_coeff*(-amplitude)
            else:
                _zernike_coeff *= 0
                _zernike_coeff[current_mode_index] = -amplitude
                zernike_coeff = _zernike_coeff
                
        return zernike_coeff

    def get_single_mode_probe (self, amplitude, n_interation):
        sub_amplitude = [1, 0, -1]
        iterations_per_sub_mode = int(self.param['num_probe_iteration'])
        total_interation_per_mode = iterations_per_sub_mode * len(sub_amplitude)

        # Determine the current mode and sub-amplitude based on self.n
        current_mode_index = n_interation // total_interation_per_mode  # Which mode we are in (0-based)

        # Determine the sub-amplitude index within the current mode
        mode_iteration = n_interation % total_interation_per_mode  # Iteration index within the mode
        sub_amplitude_index = mode_iteration // iterations_per_sub_mode
        current_mode = self._starting_mode + current_mode_index  # Mode numbers start from 4
        current_amplitude = sub_amplitude[sub_amplitude_index]

        # Perform the injection
        zernike_coeff = self.get_zernike_probe(n_interation, amplitude, key={1: '+', 0: None, -1: '-'}[current_amplitude], current_mode_index=current_mode_index)

        return zernike_coeff 

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

    def inject_custom_probe(self, amplitude, probe_inputs, n_interation):
        iteration_within_combined = n_interation % (self.param['num_probe_iteration'] * 3)

        print('Injecting combined mode probe')

        if iteration_within_combined < self.param['num_probe_iteration']:
            print('Injecting combined + probe')
            multiplier = 1
        elif iteration_within_combined < 2 * self.param['num_probe_iteration']:
            print('Injecting combined 0 probe')
            multiplier = 0
        else:
            print('Injecting combined - probe')
            multiplier = -1

        # Apply the multiplier to each element of probe_inputs
        modified_probe_inputs = [amplitude * input_value * multiplier for input_value in probe_inputs]

        return modified_probe_inputs


    def apply_perturbation_to_wavefront(self, zernike_coeff, wvl, seed):
        """
        Apply a single perturbation to the wavefront.

        Args:
            zernike_coeff (array): Zernike coefficients for the current iteration.
            wvl (float): Wavelength in meters.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple:
                - phase_aberration (Field): Updated wavefront aberration phase.
                - updated_zernike_coeff (array): Updated Zernike coefficients.
                - n_aberration (array): Perturbations applied to each mode.
        """
        updated_zernike_coeff = np.zeros(len(self.zernike_basis))
        n_aberration = np.zeros(len(self.zernike_basis))
        rng = np.random.default_rng(seed)

        # TODO - add an assertation in here: if the zernike_coeff is not the same length as the zernike_basis, raise an error

        for i in range(len(self.zernike_basis)):
            # rng = np.random.default_rng(seed)
            n_coeff, _ = noll_to_zernike(i + 4)

            print(f'Mode{i+4}, c: {zernike_coeff[i]}')

            # Standard deviation based on wavelength and mode
            standard_deviation = (0.2e-9 / 60) / ((n_coeff + 1) ** 2 * wvl)
            delta_aberration = rng.normal(scale=standard_deviation)
            n_aberration[i] = delta_aberration

            updated_zernike_coeff[i] = (
                self.param['leaking_factor'] * zernike_coeff[i]
                + (self.param['implementation_parameter'] * delta_aberration) * (2 * np.pi / wvl)
            )

        phase_aberration = self.zernike_basis.linear_combination(updated_zernike_coeff)
        
        return phase_aberration, updated_zernike_coeff, n_aberration

    def track_zernike_component(self, zernike_coeff, wvl, starting_field, random_seed=False):
        """
        Track Zernike component perturbations and wavefront evolution over iterations.

        Args:
            zernike_coeff (array): Initial Zernike coefficients.
            wvl (float): Wavelength in meters.
            starting_field (Field): Initial wavefront field.

        Returns:
            tuple:
                - n_field (array): Wavefront aberrations over all iterations.
                - n_zernike_coeff (array): Zernike coefficients over all iterations.
                - n_aberration (array): Perturbations applied at each iteration.
        """
        num_iterations = self.param['num_iteration']
        n_zernike_coeff = np.zeros((num_iterations, len(zernike_coeff)))
        n_zernike_coeff[0, :] = np.array(zernike_coeff)

        n_field = []
        n_aberration = np.zeros([num_iterations, len(zernike_coeff)])

        for i in range(num_iterations):
            if random_seed is False: 
                seed = i
            else: 
                seed = np.random.randint(0, 2**32)

            if i == 0:
                zernike_coeff_input = zernike_coeff
            else:
                zernike_coeff_input = new_zernike_coefficients

            print(f'iteration{i}, {zernike_coeff_input[0]}')
            phase_aberration, new_zernike_coefficients, step_between_iterations = self.apply_perturbation_to_wavefront(zernike_coeff_input, wvl, seed)

            n_zernike_coeff[i,:] = new_zernike_coefficients
            n_field.append(phase_aberration)
            n_aberration[i, :] = step_between_iterations
            
        return n_zernike_coeff, n_field, n_aberration 

    def apply_perturbation_to_wavefront_jacobian(self, jacobian_coeff, wvl, seed):

        """
        Apply a single perturbation to the wavefront using Jacobian basis.

        Args:
            jacobian_coeff (array): Coefficients in Jacobian SVD basis for the current iteration.
            wvl (float): Wavelength in meters.
            seed (int): Random seed for reproducibility.

        Returns:
            tuple:
                - phase_aberration (Field): Updated wavefront aberration phase.
                - updated_jacobian_coeff (array): Updated coefficients.
                - n_aberration (array): Perturbations applied to each mode.
        """
        num_modes = len(jacobian_coeff)
        updated_jacobian_coeff = np.zeros(num_modes)
        n_aberration = np.zeros(num_modes)
        rng = np.random.default_rng(seed)

        for i in range(10):
            print(f'Mode {i}, c: {jacobian_coeff[i]}')
            # Similar standard deviation scaling
            standard_deviation = (0.01*0.2e-9 / 60) / ((i + 1) ** 2 * wvl) # (0.2e-9* 0.01 / 60) --> to decrease the drift
            delta_aberration = rng.normal(scale=standard_deviation)
            n_aberration[i] = delta_aberration

            updated_jacobian_coeff[i] = (
                self.param['leaking_factor'] * jacobian_coeff[i]
                + (self.param['implementation_parameter'] * delta_aberration) * (2 * np.pi / wvl)
            )

        # Combine DM actuator modes using the coefficients
        full_dm_command = self.custom_basis.V_modes_physical @ updated_jacobian_coeff
        self.sim.apply_actuators(full_dm_command)

        # get the phase aberration on dm1 and dm2
        phase1 = self.sim.dm1.phase_for(wvl)
        phase2 = self.sim.dm2.phase_for(wvl)

        phase_aberration = phase1 + phase2

        return phase_aberration, updated_jacobian_coeff, n_aberration

    def track_jacobian_component(self, jacobian_coeff, wvl, starting_field, random_seed=False):
        """
        Track Jacobian component perturbations and wavefront evolution over iterations.

        Args:
            jacobian_coeff (array): Initial coefficients.
            wvl (float): Wavelength in meters.
            starting_field (Field): Initial wavefront field.

        Returns:
            tuple:
                - n_field (array): Wavefront aberrations over all iterations.
                - n_jacobian_coeff (array): Coefficients over all iterations.
                - n_aberration (array): Perturbations applied at each iteration.
        """
        num_iterations = self.param['num_iteration']
        n_jacobian_coeff = np.zeros((num_iterations, len(jacobian_coeff)))
        n_jacobian_coeff[0, :] = np.array(jacobian_coeff)

        n_field = []
        n_aberration = np.zeros([num_iterations, len(jacobian_coeff)])

        for i in range(num_iterations):
            seed = i if not random_seed else np.random.randint(0, 2**32)

            jacobian_coeff_input = jacobian_coeff if i == 0 else new_jacobian_coefficients

            print(f'iteration {i}, coeff0: {jacobian_coeff_input[0]}')
            phase_aberration, new_jacobian_coefficients, step_between_iterations = self.apply_perturbation_to_wavefront_jacobian(
                jacobian_coeff_input, wvl, seed
            )

            n_jacobian_coeff[i, :] = new_jacobian_coefficients
            n_field.append(phase_aberration)
            n_aberration[i, :] = step_between_iterations        

        return n_jacobian_coeff, n_field, n_aberration

    def get_aberration_data_cube(self, n_field, radius_ppl_px=336):
        self.aberration_cube = n_field
    
"""

    def inject_zernike_probe (self, amplitude=1e-9, seed=1):
        np.random.seed(seed)

        zernike_coeff = np.random.randn(len(self.probe_zernike_basis)) 
        zernike_coeff /= np.sqrt(len(self.probe_zernike_basis))

        mode_inject = amplitude * zernike_coeff / self.probe_rms_modes

        delta_actutor = self.probe_zernike_basis.transformation_matrix.dot(mode_inject)

        return delta_actutor
"""
