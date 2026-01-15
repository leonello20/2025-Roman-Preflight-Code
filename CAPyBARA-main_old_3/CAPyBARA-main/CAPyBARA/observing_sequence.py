import matplotlib.pyplot as plt
import numpy as np
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

import CAPyBARA.mode_basis as zernike
import CAPyBARA.utils as utils

class ObservingSequence:
    def __init__(self, sim_list, aberration_list):
        self.sim_list = sim_list
        self.aberration_list = aberration_list

        if isinstance(self.sim_list, CAPyBARAsim):
            self.name = 'SingleWavelength'
            self.param = self.aberration_list.param
            self.aberration_cube = self.aberration_list.aberration_cube
            self.num = len(self.aberration_list.aberration_cube)
        elif isinstance(self.sim_list, list) and len(self.sim_list) > 1:
            self.name = 'MultiWavelength'
            self.param = [aberration.param.copy() for aberration in self.aberration_list]
            self.aberration_cube = [aberration.aberration_cube.copy() for aberration in self.aberration_list]
            self.num = len(self.aberration_list[0].aberration_cube) #self.aberration_list[0].aberration_cube.shape[0]

        else:
            raise TypeError("sim_list must be a CAPyBARAsim instance or a list of CAPyBARAsim objects.")

    def set_sim(self):
        sim = self.sim_list if isinstance(self.sim_list, CAPyBARAsim) else self.sim_list[j]
        return sim 

    def acquisition_loop(self, wvl, actuators):
        if len(wvl) > 1:
            print('Multi-wavelength acquisition')
        else:
            print('Single-wavelength acquisition')

        img_list = []
        e_field_list = []
        wf_lyot_list = []
        wf_residual_list = []

        for iteration in range(self.num):
            self.iteration = iteration
            print(f'Iteration {iteration}/{self.num}')

            current_actuators = actuators[iteration] if isinstance(actuators, list) else actuators

            images, electric_fields, wf_lyot, wf_residual = self.process_wavelengths(wvl, current_actuators)

            img_list.append(images)
            e_field_list.append(electric_fields)
            wf_lyot_list.append(wf_lyot)
            wf_residual_list.append(wf_residual)

        return img_list, e_field_list, wf_lyot_list, wf_residual_list

    def process_wavelengths(self, wvl, actuators):
        x = []
        images = []
        electric_fields = []
        wf_lyot = []
        wf_residual = []

        if not isinstance(wvl, list):
            wvl = [wvl]

        for j in range(len(wvl)):
            aberration_list = self.aberration_list[j] if isinstance(self.aberration_list, list) else self.aberration_list

            # Handle aberration depending on single or multiple simulations
            aberration = self.get_aberration(aberration_list, self.iteration)
            print(f'What is the current aberration? {np.sum(aberration)}')

            result = self.process_simulation(j, aberration, wvl[j], actuators)

            images.append(result[0])
            electric_fields.append(result[1])
            wf_lyot.append(result[2])
            wf_residual.append(result[3])

        return images, electric_fields, wf_lyot, wf_residual

    def process_simulation(self, j, aberration, wvl, current_actuators):
        """Process a single simulation (or a single wavelength if it's monochromatic)."""
        sim = self.sim_list[j] if isinstance(self.sim_list, list) else self.sim_list

        include_aberration = None 

        img = sim.get_image(
            current_aberration=aberration, 
            wvl=wvl, 
            actuators=current_actuators, 
            include_aberration=include_aberration
        )

        print(f'What is the current actuator? {np.sum(current_actuators)}')

        return img.intensity, img.electric_field, sim.wf_post_lyot, sim.wf_post_field_stop

    def get_aberration(self, aberration_list, index):
        """Get the correct aberration depending on the type of simulation list."""
        if self.name == 'SingleWavelength':
            # Single wavelength case
            print(f' ==== Get Current aberration (single wavelength): {aberration_list.aberration_cube[index]} ==== ')
            return aberration_list.aberration_cube[index]
        else:
            # Broadband case
            print(f' ==== Get Current aberration (broadband): {aberration_list.aberration_cube[index]} ==== ')
            return aberration_list.aberration_cube[index]