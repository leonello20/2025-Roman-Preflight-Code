#   Copyright 2019 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  28 August 2019


import shutil
import cgisim as cgisim

def copy_examples_here( ):

    # copy CGISIM examples into local directory

    files = ['testsim_defocus.py', 'testsim_lowfs.py', 'testsim_lowfs_emccd.py', 'testsim_hlc.py', 'testsim_spc_wide.py', 
             'testsim_spc_excam.py', 'testsim_spc_spec.py', 'testsim_mask_shift.py', 'testsim_zwfs.py', 
             'testsim_hlc_iterations.py', 'testsim_spc_spec_iterations.py', 'testsim_spc_wide_iterations.py']

    for f in files:
        filename = cgisim.lib_dir + '/examples/' + f
        try:
            print( "Copying " + f + " to current directory" )
            shutil.copy( filename, './.' )
        except IOError as e:
            raise IOError( "Unable to copy prescription to current directory. %s" % e )

