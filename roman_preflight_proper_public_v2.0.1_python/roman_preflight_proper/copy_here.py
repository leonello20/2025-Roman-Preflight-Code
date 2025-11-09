#   Copyright 2019 California Institute of Technology
# ------------------------------------------------------------------

import shutil
import roman_preflight_proper

def copy_here( ):

    # copy PROPER prescriptions from roman_preflight_proper package into local directory

    prescription_file = roman_preflight_proper.lib_dir + '/roman_preflight.py'
    try:
        shutil.copy( prescription_file, './.' )
    except IOError as e:
        raise IOError( "Unable to copy prescription to current directory. %s" % e )

    prescription_file = roman_preflight_proper.lib_dir + '/roman_preflight_compact.py'
    try:
        shutil.copy( prescription_file, './.' )
    except IOError as e:
        raise IOError( "Unable to copy prescription to current directory. %s" % e )

