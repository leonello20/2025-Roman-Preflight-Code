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
import copy

from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import scipy.ndimage as ndimage

from CAPyBARA.plotting import *
import CAPyBARA.rst_functions as rst_func
import CAPyBARA.aberration as rst_aberration
from astropy.io import fits

import CAPyBARA.utils as utils

# spatial sampling = .0218 arcsec per pixel
# N_q = 2 at 500 nm
# sampling = wvl/(res*D)
# TODO - Find out what unit we are in for CAPyBARA
# TODO - check wavefront unit at each step

# Physical unit in here should be in metre
# Use the focal grid at the reference wvl

""" Logic of the simulator is as follow

Instead of doing the iterative approach, i.e. injecting aberrations frame by frame while running the EFC, you created the sequence first, then modify the electric field of the cube. Hence almost all functions will need the aberraiton cube first

Returns:
    _type_: _description_

    ## TODO - Seperate CAPyBARA into different classes
"""
class CAPyBARAsim:
    def __init__(self, param, tip_tilt_focus=None):
        """
        Initialize the CAPyBARAsim class with telescope parameters.

        Parameters:
        -----------
        telescope_params : dict
            Dictionary containing the parameters specific to the telescope.
        tip_tilt_focus : None or np.ndarray
            Optional initial settings for tip/tilt and focus aberrations.
        """
        self.param = copy.deepcopy(param)  # Only the telescope parameters
        self.tip_tilt_focus = tip_tilt_focus

    def get_param(self, full_param, sequence=None):
        """
        Calculate and initialize key system parameters based on whether EFC or observation mode is used.

        Parameters:
        -----------
        full_param : dict
            Full dictionary containing all parameters (EFC, observation, etc.).
        sequence : str
            String flag determining the sequence mode ('efc' or 'observation').
        """
        import copy

        # Deep copy of the telescope parameters to preserve the original
        params = copy.deepcopy(self.param)

        # Ensure 'path' and its keys exist in full_param
        path_params = full_param.get('path', {})
        params.setdefault('path', {})
        params['path']['jacobian'] = path_params.get('jacobian', None)
        params['path']['data_path'] = path_params.get('data_path', None)

        # Determine the wavelength to use based on the sequence mode
        if sequence == 'efc':
            efc_params = full_param.get('efc', {})
            params['wvl'] = efc_params.get('wvl', params.get('wvl'))
            params['ref_wvl'] = efc_params.get('ref_wvl', params.get('ref_wvl'))
        elif sequence == 'observation':
            obs_params = full_param.get('observation', {})
            params['wvl'] = obs_params.get('wvl', params.get('wvl'))
            params['ref_wvl'] = obs_params.get('ref_wvl', params.get('ref_wvl'))

        # Handle other parameters shared between modes
        for key in ['num_iteration', 'is_static', 'num_mode', 'aberration_ptv',
                    'implementation_parameter', 'leaking_factor']:
            if sequence == 'efc' and key in efc_params:
                params[key] = efc_params[key]
            elif sequence == 'observation' and key in obs_params:
                params[key] = obs_params[key]

        # Additional system parameters
        params['actuator_spacing'] = 1.05 / params['num_actuator'] * params['diameter']
        params['focal_length'] = params['diameter'] * params['f_number']
        params['spatial_resolution'] = params['f_number'] * params['wvl'] * 1e-9

        # Spatial resolution based on the reference wavelength
        self._spatial_res = params['f_number'] * params['ref_wvl'] * 1e-9  # unit [m]

        # Lambda / D calculations and other physical parameters
        params['r_field_stop'] = 13.2 * params['f_number'] * params['ref_wvl'] * 1e-9  # Reference wavelength is at 556 nm
        params['radius_hlc_obstruction'] = params['radius_hlc_obstruction']
        params['talbot_distance'] = 5.546e4

        # Update the object's parameters
        self.param.update(params)

        return params

    def get_grid (self):
        # TODO - param['radius_pupil_pixels'] - this is actually a diameter, no *2 for all 
        self.pupil_grid = make_pupil_grid(self.param['radius_pupil_pixels'], diameter = 1.1* self.param['diameter'])
        self.focal_grid = make_focal_grid(q=2, num_airy = 16, spatial_resolution = self._spatial_res)
        self.dark_zone_mask = self.get_dark_zone() #(grid=self.focal_grid) #(grid=self.focal_grid) # checked

    def get_dark_zone(self):
        def dark_zone_generator():
            dark_zone = (
                make_circular_aperture(2 * self.param['owa']*self._spatial_res)(self.focal_grid) -
                make_circular_aperture(2 * self.param['iwa']*self._spatial_res)(self.focal_grid)
            ).astype(bool)

            return dark_zone
        return dark_zone_generator()

    def get_prop (self):
        self.prop_ptf = FraunhoferPropagator(self.pupil_grid, self.focal_grid, focal_length=self.param['focal_length'])
        self.prop_dm = FresnelPropagator(self.pupil_grid, self.param['talbot_distance'])

    def get_actuator (self): 
        self.actuator = 2*len(self.influence_function)

    def get_system (self, influence_function, check = False):
        if self.param['path']['jacobian'] != 'None': 
            print('Read jacobian matrix:', self.param['path']['jacobian'])
            print(type(self.param['path']['jacobian']))
            _path = os.path.join(self.param['path']['data_path'],self.param['path']['jacobian']+f"{self.param['wvl']}nm.fits")
            self.jacobian = utils.read_fits(_path)
    
        self.influence_function = influence_function
        self.aperture = evaluate_supersampled(_make_rst_aperture(normalized=False), self.pupil_grid, 4) #mask checked
        
        _lyot_mask_gen = _make_lyot_mask(normalized = False)
        self.lyot_mask = _lyot_mask_gen(self.pupil_grid) #mask checked

        self.ppm = evaluate_supersampled(make_circular_aperture(1*self.param['diameter']), self.pupil_grid, 4) #mask checked

        self.fpm = evaluate_supersampled(make_obstruction
                                        (make_circular_aperture
                                        (self.param['radius_hlc_obstruction']*2*self._spatial_res)), self.focal_grid, 16) #mask checked
        
        self.lyot_coronagraph = LyotCoronagraph(self.pupil_grid, self.fpm, lyot_stop=self.lyot_mask, focal_length=self.param['focal_length'] )

        self.field_stop_mask = evaluate_supersampled(make_circular_aperture(self.param['r_field_stop']*2), self.focal_grid, 16) # checked

        self.field_stop = LyotCoronagraph(self.pupil_grid, self.field_stop_mask, lyot_stop=None, focal_length=self.param['focal_length'])

        self.dm1 = DeformableMirror(self.influence_function)
        self.dm2 = DeformableMirror(self.influence_function)

        self.wf_ref = Wavefront(self.aperture, self.param['wvl']*1e-9) # wavefront without aberration

        # TODO - Optional: general mask, open mask 
        if check is True:
            plt_field(self.aperture, title='RST aperture')
            plt_field(self.lyot_mask, title='lyot mask')
            plt_field(self.ppm, title = 'Pupil plane mask')
            plt_field(self.fpm, title = 'Focal plane mask')
            plt_field(self.dark_zone_mask, title = 'Dark zone mask')

    def get_reference_image(self, wvl, static_aberration_func, wavefront_error = None, check = False):
        "wvl should be in metre in here"
        "Create an initial wavefront without any aberration, and a direct image for calibrating the contrast"
        
        self.wf_ref = Wavefront(self.aperture, wvl) # wavefront without aberration

        # wavefront with static aberration
        self.wf_surface_aberration =  self.create_wavefront(wvl=wvl, current_aberration=None, include_aberration=static_aberration_func)
    
        if wavefront_error is not None:
            # directly inject the wavefront error in here
            _opd2phase = 2 * np.pi / (wvl)

            phase_offset = wavefront_error * _opd2phase

            self.wf_surface_aberration.electric_field *= np.exp(1j * phase_offset)

        # you need to include surface aberration for the reference images
        self.ref_img = self.prop_ptf(self.wf_surface_aberration).intensity
        
        if check is True: 
            print(f'Current wvl:{wvl}')
            imshow_field(np.log10(self.ref_img/self.ref_img.max()), grid=self.focal_grid)
        
    def apply_actuators(self, actuators=None):
        """
        Apply actuator settings to deformable mirrors (DM1 and DM2).

        Parameters:
        -----------
        actuators : np.ndarray or None
            The actuator settings to be applied. If None, no actuators are applied.
        """
        if actuators is not None:
            self.dm1.actuators = actuators[:len(self.influence_function)]
            self.dm2.actuators = actuators[len(self.influence_function):]
    
    def create_wavefront(self, wvl, current_aberration=None, include_aberration=None):
        """
        Create the wavefront with optional aberrations.

        Parameters:
        -----------
        wvl : float or np.ndarray
            Wavelength or array of wavelengths in meters.
        current_aberration : np.ndarray or None
            The current aberration to apply to the wavefront. If None, no aberration is applied.
        include_aberration : 
            Whether to include aberrations in the wavefront creation.

        Returns:
        --------
        wf : Wavefront
            The created wavefront with or without aberrations.
        """
        if current_aberration is None and include_aberration is not None: 
            print('Static aberration')
            wf = include_aberration(self.wf_ref) # introduce static aberration 

        elif include_aberration is None and current_aberration is not None:
            print('Quasi-Static aberration') 

            wf = Wavefront(self.aperture, wvl) # wavefront without aberration

            self.current_aberration = current_aberration

            total_aberration = current_aberration + self.wf_surface_aberration.phase

            wf.electric_field *= np.exp(1j * total_aberration) # introduce quasi-static aberration

            self.current_wf = wf

        elif current_aberration is None and include_aberration is None:
            print('Perfect wavefront')
            wf = self.wf_ref

        return wf

    def propagate_wavefront(self, wf):
        """
        Propagate the wavefront through the system: deformable mirrors, coronagraph, and field stop.

        Parameters:
        -----------
        wf : Wavefront
            The wavefront to propagate.

        Returns:
        --------
        wf_post_field_stop : Wavefront
            The propagated wavefront after the field stop.
        """
        self.wf_post_dms = self.prop_dm.backward(self.dm2(self.prop_dm.forward(self.dm1(wf)))) # at pupil plane
        self.wf_post_lyot = self.lyot_coronagraph(self.wf_post_dms)
        self.wf_post_field_stop = self.field_stop(self.wf_post_lyot)

    # TODO - this function maybe should only take a wavefront? not sure about that right now
    def get_wf_post_lyot_field_stop(self, num_modes, current_aberration, wvl, actuators=None, include_aberration=None):
        """
        Get the wavefront after the Lyot stop and field stop.

        Parameters:
        -----------
        num_modes : int
            Number of Zernike modes (not used here, but could be for future functionality).
        current_aberration : np.ndarray
            The current aberration to apply to the wavefront.
        wvl : float or np.ndarray
            Wavelength(s) to create the wavefront.
        actuators : np.ndarray or None
            Actuator settings to apply to the deformable mirrors.
        include_aberration : bool
            Whether to include aberration in the wavefront.

        Returns:
        --------
        wf_post_field_stop : Wavefront
            The wavefront after Lyot stop and field stop.
        """
        self.apply_actuators(actuators)

        # Create wavefront and propagate
        wf = self.create_wavefront(wvl, current_aberration, include_aberration)
        self.propagate_wavefront(wf)


    def get_image(self, current_aberration, wvl, actuators=None, include_aberration=None, wf=None):
        """
        Generate a coronagraphic image by propagating the wavefront through the entire system.

        Parameters:
        -----------
        current_aberration : np.ndarray
            The current aberration to apply to the wavefront.
        wvl : float or np.ndarray
            Wavelength or array of wavelengths in nanometers.
        actuators : np.ndarray or None
            Actuator settings to apply to the deformable mirrors.
        include_aberration : bool
            Whether to include aberration in the wavefront creation.

        Returns:
        --------
        img : np.ndarray
            The generated image after propagation.
        """
        # Apply actuators and create the wavefront
        self.apply_actuators(actuators)

        if wf is None: 
            wf = self.create_wavefront(wvl, current_aberration, include_aberration)
            self.wf = wf
            
        # Propagate the wavefront and generate the image
        self.propagate_wavefront(wf)
        img = self.prop_ptf(self.wf_post_field_stop)

        return img

    def get_coronagrphic_image (self, wf_post_field_stop, check = False):
        self._img_final = self.prop_ptf(wf_post_field_stop).intensity

        if check is True:
            plt_field(np.log10(self._img_final), title = 'On axis source', cmap = 'inferno')