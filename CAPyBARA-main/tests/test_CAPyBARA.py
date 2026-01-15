import pytest
import numpy as np
from CAPyBARA import CAPyBARAsim
from CAPyBARA.utils import read_ini_file  # Assuming this function is in utils or other module

@pytest.fixture
def setup_sim(tmp_path):
    """
    This fixture sets up the CAPyBARAsim class instance with parameters from the .ini file.
    """
    # Define or load your config file path
    config_file = tmp_path / "test_config.ini"

    # Write the .ini file content
    with open(config_file, 'w') as f:
        f.write("""
        [path]
        data_path = /Users/ylau/Data/CAPyBARA/
        jacobian = 2024-09-13_jacobian_matrix_

        [efc]
        wvl = 556
        loop_gain = 0.5
        num_iteration = 50
        rcond = 1e-3
        is_static = False
        num_mode = 21
        aberration_ptv = 0.3
        implementation_parameter = 50
        leaking_factor = 0.99
        ref_wvl = 556

        [observation]
        wvl = 556
        last_dm_command = None
        num_iteration = 50
        is_static = False
        num_mode = 21
        aberration_ptv = 0.3
        implementation_parameter = 50
        leaking_factor = 0.99
        ref_wvl = 556
        loop_gain = 0.5

        [aberration]
        chromaticity = False

        [telescope]
        diameter = 2.36
        central_obs = 0.303
        radius_pupil_pixels = 336
        f_number = 7.9
        num_actuator = 48
        iwa = 3.0
        owa = 9.0
        radius_hlc_obstruction = 2.6

        [sequence]
        is_efc = True
        is_observation = True
        """)

    # Load parameters from the .ini configuration file (mimicking the actual script)
    param_rst, sequence = read_ini_file(str(config_file))

    # Initialize the simulation system
    sim = CAPyBARAsim(param_rst['telescope'])
    return sim, param_rst

def test_get_param(setup_sim):
    # Testing the get_param function
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')

    # Test if parameters of the telescope has been loaded correctly
    assert sim.param['diameter'] == param_rst['telescope']['diameter']
    assert sim.param['wvl'] == param_rst['efc']['wvl']

def test_apply_actuators(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')

    # Apply some actuator command (mocking this for the test)
    actuators = np.zeros(sim.num_modes)
    actuators[5] = 0.1  # Apply an arbitrary value to actuator 5

    # Apply the actuators
    sim.apply_actuators(actuators)

    # Validate the actuator state (assuming there's a way to get the state)
    assert np.allclose(sim.dm1.surface, actuators), "DM1 surface does not match applied actuators"
    assert np.allclose(sim.dm2.surface, actuators), "DM2 surface does not match applied actuators"

def test_get_image(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')

    # Set up some mock actuators and image
    actuators = np.zeros(sim.num_modes)
    actuators[2] = 0.1  # Apply an arbitrary value to actuator 2

    sim.apply_actuators(actuators)

    # Test image creation (assuming get_image returns an object with intensity)
    img = sim.get_image(current_aberration=None, wvl=param_rst['efc']['wvl'], actuators=actuators, include_aberration=False)

    # Check that the image is correctly generated (depending on expected behavior)
    assert img.intensity is not None, "Image intensity should not be None"
    assert img.intensity.shape == sim.pupil_grid.shape, "Image dimensions should match the pupil grid"

    # Optionally, check for numerical values in intensity
    assert np.max(img.intensity) > 0, "Image intensity should have positive values"

def test_get_grid(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')
    sim.get_grid()

    # Check if the grids are created correctly
    assert sim.pupil_grid is not None, "Pupil grid should not be None"
    assert sim.focal_grid is not None, "Focal grid should not be None"
    assert sim.dark_zone_mask is not None, "Dark zone mask should not be None"

def test_get_prop(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')
    sim.get_grid()
    sim.get_prop()

    # Check if the propagators are created correctly
    assert sim.prop_ptf is not None, "Fraunhofer propagator should not be None"
    assert sim.prop_dm is not None, "Fresnel propagator should not be None"

def test_get_actuator(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')
    sim.get_actuator()

    # Check if the actuator is set correctly
    assert sim.actuator == 2 * len(sim.influence_function), "Actuator value is incorrect"

def test_get_system(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')
    influence_function = np.ones((48, 48))  # Mock influence function
    sim.get_system(influence_function, check=False)

    # Check if the system components are initialized correctly
    assert sim.jacobian is not None, "Jacobian should not be None"
    assert sim.aperture is not None, "Aperture should not be None"
    assert sim.lyot_mask is not None, "Lyot mask should not be None"
    assert sim.ppm is not None, "Pupil plane mask should not be None"
    assert sim.fpm is not None, "Focal plane mask should not be None"
    assert sim.lyot_coronagraph is not None, "Lyot coronagraph should not be None"
    assert sim.field_stop_mask is not None, "Field stop mask should not be None"
    assert sim.field_stop is not None, "Field stop should not be None"
    assert sim.dm1 is not None, "DM1 should not be None"
    assert sim.dm2 is not None, "DM2 should not be None"
    assert sim.wf_ref is not None, "Wavefront reference should not be None"

def test_get_reference_image(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')
    sim.get_grid()
    sim.get_prop()
    sim.get_actuator()
    influence_function = np.ones((48, 48))  # Mock influence function
    sim.get_system(influence_function, check=False)

    # Mock static aberration function
    def static_aberration_func(wf):
        return wf

    sim.get_reference_image(wvl=param_rst['efc']['wvl'] * 1e-9, static_aberration_func=static_aberration_func)

    # Check if the reference image is created correctly
    assert sim.ref_img is not None, "Reference image should not be None"
    assert sim.ref_img.shape == sim.focal_grid.shape, "Reference image dimensions should match the focal grid"

def test_propagate_wavefront(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')
    sim.get_grid()
    sim.get_prop()
    sim.get_actuator()
    influence_function = np.ones((48, 48))  # Mock influence function
    sim.get_system(influence_function, check=False)

    # Mock wavefront
    wf = sim.wf_ref

    sim.propagate_wavefront(wf)

    # Check if the wavefront is propagated correctly
    assert sim.wf_post_dms is not None, "Wavefront after DMs should not be None"
    assert sim.wf_post_lyot is not None, "Wavefront after Lyot should not be None"
    assert sim.wf_post_field_stop is not None, "Wavefront after field stop should not be None"

def test_get_wf_post_lyot_field_stop(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')
    sim.get_grid()
    sim.get_prop()
    sim.get_actuator()
    influence_function = np.ones((48, 48))  # Mock influence function
    sim.get_system(influence_function, check=False)

    # Mock current aberration
    current_aberration = np.zeros(sim.pupil_grid.shape)

    sim.get_wf_post_lyot_field_stop(num_modes=21, current_aberration=current_aberration, wvl=param_rst['efc']['wvl'] * 1e-9)

    # Check if the wavefront after Lyot and field stop is created correctly
    assert sim.wf_post_field_stop is not None, "Wavefront after field stop should not be None"

def test_get_coronagraphic_image(setup_sim):
    sim, param_rst = setup_sim
    sim.get_param(param_rst, sequence='efc')
    sim.get_grid()
    sim.get_prop()
    sim.get_actuator()
    influence_function = np.ones((48, 48))  # Mock influence function
    sim.get_system(influence_function, check=False)

    # Mock wavefront
    wf = sim.wf_ref

    sim.propagate_wavefront(wf)
    sim.get_coronagrphic_image(sim.wf_post_field_stop)

    # Check if the coronagraphic image is created correctly
    assert sim._img_final is not None, "Coronagraphic image should not be None"
    assert sim._img_final.shape == sim.focal_grid.shape, "Coronagraphic image dimensions should match the focal grid"
