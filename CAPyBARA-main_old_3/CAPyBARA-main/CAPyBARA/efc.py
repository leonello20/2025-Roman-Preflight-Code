import matplotlib.pyplot as plt
import numpy as np
from hcipy import *
from astropy.visualization import astropy_mpl_style
plt.style.use(astropy_mpl_style)
          
from astropy.io import fits
from CAPyBARA.rst_design import _make_rst_aperture, _make_lyot_mask
from CAPyBARA.rst_functions import *
import scipy.fftpack
import os
import sys
from CAPyBARA import CAPyBARAsim
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import scipy.ndimage as ndimage

from CAPyBARA.plotting import *
import CAPyBARA.rst_functions as rst_func
import CAPyBARA.aberration as rst_aberration
from astropy.io import fits

class EFieldConjugation: 
    def __init__(self, sim_list, aberration_list):
        self.sim_list = sim_list
        self.aberration_list = aberration_list

        # setting internal parameters to be passed to the efc 
        if isinstance(self.sim_list, CAPyBARAsim):
            print('Single simulation')
            # Case 1: Single simulation (not in a list)
            self.name = 'EFC'
            self.param = self.aberration_list.param  # Directly use the simulation instance's parameters
            self.jacobian = self.sim_list.jacobian
            self.aberration_cube = self.aberration_list.aberration_cube  # Single aberration cube
            self.num = self.aberration_list.param['num_iteration']
            self.num_modes = 2 * len(self.sim_list.influence_function)
            self.rcond = self.aberration_list.param['rcond']

        elif isinstance(self.sim_list, list) and len(self.sim_list) > 1:
            # Case 2: Multiple simulations in a list (BroadbandEFC)
            self.name = 'BroadbandEFC'
            self.param = [aberration.param.copy() for aberration in self.aberration_list]
            self.G = []

            for sim in self.sim_list: 
                self.G.append(sim.jacobian.copy())

            self.jacobian = np.vstack(self.G)  # Stack all Jacobians
            self.aberration_cube = [aberration.aberration_cube.copy() for aberration in self.aberration_list]
            self.num = self.aberration_list[0].param['num_iteration']
            self.rcond = self.aberration_list[0].param['rcond']
            self.num_modes = 2 * len(self.sim_list[0].influence_function)

        else:
            raise TypeError("sim_list must be a CAPyBARAsim instance or a list of CAPyBARAsim objects.")
    
    def get_aberration(self, aberration_list, index):
        """Get the correct aberration depending on the type of simulation list."""
        if self.name == 'EFC':
            # Single wavelength case
            print(f' ==== Get Current aberration (single wavelength): {aberration_list.aberration_cube[index]} ==== ')
            return aberration_list.aberration_cube[index]
        else:
            # Broadband case
            print(f' ==== Get Current aberration (broadband): {aberration_list.aberration_cube[index]} ==== ')
            return aberration_list.aberration_cube[index]

    def update_actuators(self, current_actuators, efc_matrix, x):
        """Update actuators based on the EFC matrix and the current electric field."""
        y = efc_matrix.dot(x)

        # Adjust actuators based on single or multiple simulations
        if isinstance(self.aberration_list, list):
            return current_actuators - self.param[0]['loop_gain'] * y
        return current_actuators - self.param['loop_gain'] * y

    def control(self, wvl):
        if len(wvl) > 1:
            print('Broadband EFC')
        else:
            print('Monochromatic')

        # Initialize lists to store results
        actuators_list = []
        img_list = []
        e_field_list = []
        wf_list = []
        wf_residual_list = []

        # Initialize actuators and matrix
        current_actuators = np.zeros(self.num_modes)
        self.command = np.zeros(self.num_modes)
        beta_pumping = 0
        efc_matrix = self.compute_efc_matrix()

        # Main loop over iterations
        for iteration in range(self.num):
            self.iteration = iteration
            print(f'Iteration {iteration}/{self.num}')
            
            if iteration % 10 == 0:
                beta_pumping += 1
                if beta_pumping <= 6:
                    print(f'Beta pumping: {beta_pumping}')
                    efc_matrix = self.compute_efc_matrix()

            # Inject aberration
            x, images, electric_fields, wf, wf_residual = self.process_wavelengths(wvl, self.command)

            self.current_command = self.command.copy()
            # Save iteration results
            actuators_list.append(self.current_command)
            img_list.append(images)
            e_field_list.append(electric_fields)
            wf_list.append(wf)
            wf_residual_list.append(wf_residual)

            # Update actuators for next iteration
            self.command = self.update_actuators(self.command, efc_matrix, x)

        return actuators_list, e_field_list, img_list, wf_list, wf_residual_list

    def compute_efc_matrix(self):
        """Compute or update the EFC matrix."""
        return inverse_tikhonov(self.jacobian, self.rcond)

    def process_wavelengths(self, wvl, current_actuators):
        """Process all wavelengths (single or multiple) and return the necessary data."""
        x = []
        images = []
        electric_fields = []
        wf = []
        wf_residual = []

        # Ensure wvl is always treated as a list, even if it contains a single value
        if not isinstance(wvl, list):
            wvl = [wvl]

        for j in range(len(wvl)):
            print(f'Processing wavelength {j+1}/{len(wvl)}')

            aberration_list = self.aberration_list[j] if isinstance(self.aberration_list, list) else self.aberration_list

            # Handle aberration depending on single or multiple simulations
            aberration = self.get_aberration(aberration_list, self.iteration)

            print(f'Current aberration at index {self.iteration}: {np.sum(aberration)}')

            print(f'Check the shape of aberration {np.shape(aberration)}')
            # Get the output in here
            img, e_field, wf_j, wf_residual_j = self.process_simulation(j, aberration, wvl[j], current_actuators)

            # Collect results
            electric_fields.append(e_field)
            wf.append(wf_j)
            wf_residual.append(wf_residual_j)
            images.append(img.intensity)

            # Concatenate real and imaginary parts of electric field for the dark zone mask
            sim = self.sim_list if isinstance(self.sim_list, CAPyBARAsim) else self.sim_list[j]
            e_field_sum = np.concatenate(
                (img.electric_field[sim.dark_zone_mask.ravel()].real,
                img.electric_field[sim.dark_zone_mask.ravel()].imag)
            )
            x.append(e_field_sum)

        return np.concatenate(x), images, electric_fields, wf, wf_residual


    def process_simulation(self, j, aberration, wvl, current_actuators):
        """Process a single simulation (or a single wavelength if it's monochromatic)."""
        sim = self.sim_list[j] if isinstance(self.sim_list, list) else self.sim_list

        include_aberration = None 
        # # quasi_static_func
        # current_aberration = aberration[j] # in terms of phase, no longer in the unit of OPD

        img = sim.get_image(
            current_aberration=aberration, 
            wvl=wvl, 
            actuators=current_actuators, 
            include_aberration=include_aberration
        )
        print(f'What is the current actuator? {np.sum(current_actuators)}')

        return img, img.electric_field, sim.current_wf, sim.wf_post_field_stop