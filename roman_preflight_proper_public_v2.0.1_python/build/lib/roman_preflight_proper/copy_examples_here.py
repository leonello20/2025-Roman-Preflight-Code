#   Copyright 2020 California Institute of Technology
# ------------------------------------------------------------------

import shutil
import roman_preflight_proper

def copy_examples_here( ):

    # copy Phase C PROPER examples into local directory

    files = ['run_flatten.py', 'run_hlc_input_fields.py', 'run_hlc.py', 'run_spc_spec.py', 'run_spc_wide.py' ]

    for f in files:
        filename = roman_preflight_proper.lib_dir + '/examples/' + f
        try:
            print( "Copying " + f + " to current directory" )
            shutil.copy( filename, './.' )
        except IOError as e:
            raise IOError( "Unable to copy prescription to current directory. %s" % e )

