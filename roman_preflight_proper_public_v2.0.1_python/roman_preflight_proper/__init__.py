import os
import os.path as _osp
import numpy as np

__version__ = '2.0.1'     # software version

dm_version = '8.0'      # default DM data version (not software version)     

lib_dir = os.path.abspath(os.path.dirname(__file__))

data_dir = lib_dir + '/preflight_data'
map_dir =  data_dir + '/maps/'
polfile = data_dir + '/pol/preflight_pol'
dm_files_dir = data_dir + '/dm/'

# CGI-related codes
from .copy_examples_here import copy_examples_here
from .copy_here import copy_here
from .fft_rotate import fft_rotate
from .ffts import ffts
from .mft_rotate import mft_rotate
from .mft1 import mft1
from .mft2 import mft2
from .polmap import polmap
from .shift_image import shift_image
from .transform_image import transform_image
from .trim import trim

# DM-related codes
from .backout_coupling import backout_coupling
from .cgi_dm import cgi_dm
from .constrain_dm import constrain_dm
from .load_cgi_dm_files import load_cgi_dm_files
from .shiftrot import shiftrot
from .stroke_to_volts import stroke_to_volts
from .volts_to_stroke import volts_to_stroke

