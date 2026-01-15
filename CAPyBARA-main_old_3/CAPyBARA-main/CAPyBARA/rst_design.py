from hcipy.aperture import make_obstructed_circular_aperture, make_spider
from hcipy.field import make_pupil_grid, evaluate_supersampled, Field
import numpy as np
import matplotlib.pyplot as plt
import os

#  TODO clean this

def _make_spider_positions(D, c, is_lyot = False):
    """
    Calculate spider positions for the telescope aperture. 


    Args:
        D (np.int): Diameter of the Roman Telescope
        c (float): _description_

    Returns:
        _type_: _description_
    """
    if is_lyot == False:
        return {
        'width': 0.65 * D / c,
        'width2': 1.5 * D / c,
        'offset': 2.45 * D / c,
        'outer_radius': 10 * D / c,
        'outer_radius2': 3.25 * D / c,
        'D': D
    }
    else: 
        return {
        'width': 0.7 * D / c,
        'width2': 1.5 * D / c,
        'offset': 2.45 * D / c,
        'outer_radius': 10 * D / c,
        'outer_radius2': 3.25 * D / c,
        'D': D
    } 

def _normalize_positions(positions):
    """ Normalize the positions based on the diameter. """
    for key in positions:
        if key != 'D':
            positions[key] /= positions['D']
    positions['D'] = 1.0

def _create_spiders(positions, with_spiders=True):
    """ Create the spider structures. """
    if not with_spiders:
        return lambda grid: 1

    spider_width, spider_width2 = positions['width'], positions['width2']
    spider_inner_radius, spider_outer_radius, spider_outer_radius2 = positions['offset'], positions['outer_radius'], positions['outer_radius2']
    pi = np.pi

    spider_configs = [
        (pi / 6, 8 * pi / 180, spider_width, spider_outer_radius, False, False, False, True),
        (pi / 6, pi / 6 + 37.71 * pi / 180, spider_width, spider_outer_radius, False, False, False, False),
        (5 * pi / 6, pi / 6 + 37.71 * pi / 180, spider_width, spider_outer_radius, False, False, True, False),
        (5 * pi / 6, 8 * pi / 180, spider_width, spider_outer_radius, False, False, True, True),
        (3 * pi / 2, 3 * pi / 2 - 8 * pi / 180 - pi / 6, spider_width, spider_outer_radius, False, False, False, False),
        (3 * pi / 2, 3 * pi / 2 + 8 * pi / 180 + pi / 6, spider_width, spider_outer_radius, False, False, False, False),
        (pi / 6, pi / 6, spider_width2, spider_outer_radius2, False, False, False, False),
        (5 * pi / 6, pi / 6, spider_width2, spider_outer_radius2, False, False, True, False),
        (3 * pi / 2, 3 * pi / 2, spider_width2, spider_outer_radius2, False, False, False, False)
    ]

    spiders = []
    for start_angle, end_angle, width, radius, invert_cos_start, invert_sin_start, invert_cos_end, invert_sin_end in spider_configs:
        spider_start = spider_inner_radius * np.array([
            np.cos(start_angle) if not invert_cos_start else -np.cos(start_angle),
            np.sin(start_angle) if not invert_sin_start else -np.sin(start_angle)
        ])
        spider_end = radius * np.array([
            np.cos(end_angle) if not invert_cos_end else -np.cos(end_angle),
            np.sin(end_angle) if not invert_sin_end else -np.sin(end_angle)
        ])
        spiders.append(make_spider(spider_start, spider_end, width))


    def _combined_spiders(grid):
        product = 1
        for spider in spiders:
            product *= spider(grid)
        return product

    return _combined_spiders

def _make_obstructed_aperture_with_spiders(D, central_obscuration_ratio ,with_spiders=True, normalized=True, is_lyot= False):
    """ Create an obstructed aperture with optional spider structures. """
    c = 19.6
    positions = _make_spider_positions(D, c, is_lyot)

    if normalized:
        _normalize_positions(positions)
        D = 1.0
    
    if is_lyot is True:
        D *= 0.8
    
    obstructed_aperture = make_obstructed_circular_aperture(D, central_obscuration_ratio)
    spider_structure = _create_spiders(positions, with_spiders)

    def aperture_function(grid):
        return Field(obstructed_aperture(grid) * spider_structure(grid), grid)

    return aperture_function

def _make_rst_aperture(normalized=True, with_spiders=True):
    """ Create the RST aperture function. """
    return _make_obstructed_aperture_with_spiders(2.36, 0.303, with_spiders, normalized,  is_lyot= False)


def _make_lyot_mask(normalized=True, with_spiders=True):
    """ Create the Lyot mask function. """
    return _make_obstructed_aperture_with_spiders(2.36, 0.6, with_spiders, normalized,is_lyot=True)
