from hcipy.aperture import make_obstructed_circular_aperture, make_spider
from hcipy.field import make_pupil_grid, evaluate_supersampled, Field
import numpy as np
import pytest 

@pytest.fixture
def rst_param ():
    param = {
        'pupil_diameter': 2.36,  #m
        'central_obstruction_ratio': 0.303, 
        'central_obstruction_ratio_lyot_coronagraph': 0.6, 
        'radius_pupil_pixels': 336
    }
    return param

def test_lyot_stop(rst_param):
    pupil_grid=make_pupil_grid(rst_param['radius_pupil_pixels']*2, diameter=rst_param['pupil_diameter']*1.1)
    lyot_mask_generator= rst_design.make_lyot_mask(normalized=False)

    lyot_mask=lyot_mask_generator(pupil_grid) #lyot_grid)

    _lyot_mask_generator= _rst_design._make_lyot_mask(normalized=False)
    _lyot_mask=lyot_mask_generator(pupil_grid) #lyot_grid)

    np.testing.assert_allclose(lyot_mask, _lyot_mask, atol = 1e-7, rtol=1e-6)

def test_lyot_stop(rst_param):
    pupil_grid=make_pupil_grid(rst_param['radius_pupil_pixels']*2, diameter=rst_param['pupil_diameter']*1.1)
    lyot_mask_generator= rst_design.make_lyot_mask(normalized=False)

    lyot_mask=lyot_mask_generator(pupil_grid) #lyot_grid)

    _lyot_mask_generator= _rst_design._make_lyot_mask(normalized=False)
    _lyot_mask=lyot_mask_generator(pupil_grid) #lyot_grid)

    np.testing.assert_allclose(lyot_mask, _lyot_mask, atol = 1e-7, rtol=1e-6)

def test_aperture(rst_param):  
    pupil_grid=make_pupil_grid(rst_param['radius_pupil_pixels']*2, diameter=rst_param['pupil_diameter']*1.1)
    
    aperture = evaluate_supersampled(rst_design.make_rst_aperture(normalized=False), pupil_grid, 4)
    _aperture = evaluate_supersampled(_rst_design._make_rst_aperture(normalized=False), pupil_grid, 4)

    np.testing.assert_allclose(aperture, _aperture, atol = 1e-7, rtol=1e-6)
