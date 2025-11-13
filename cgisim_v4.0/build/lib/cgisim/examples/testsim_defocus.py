#   Copyright 2020 California Institute of Technology
#   Users must agree to abide by the restrictions listed in the 
#   file "LegalStuff.txt" in the CGISim library directory.
#
#  Written by John Krist
#  Jet Propulsion Laboratory, California Institute of Technology
#  23 June 2020


import cgisim as cgisim
import proper
import matplotlib.pylab as plt
from roman_preflight_proper import trim
import roman_preflight_proper
import numpy as np

def testsim_defocus():
    cgi_mode = 'excam'
    cor_type = 'hlc'
    bandpass = '1a'
    polaxis = -10       # compute images for mean X+Y polarization (don't compute at each polarization)


    # read in flat wavefront pattern, no HLC pattern 

    dm1 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/hlc_flat_wfe_dm1_v.fits' ) 
    dm2 = proper.prop_fits_read( roman_preflight_proper.lib_dir + '/examples/hlc_flat_wfe_dm2_v.fits' ) 

    fig, ax = plt.subplots( nrows=1, ncols=3, figsize=(9,3), constrained_layout=True )

    print( "Computing pupil lens image..." )
    params = {'use_errors':1, 'use_fpm':0, 'use_lyot_stop':0, 'use_field_stop':0, 'use_pupil_lens':1}
    a0_pupil, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
             star_spectrum='a0v', star_vmag=2.0, output_file='testresult_a0v_pupil_image.fits' )
    a0_pupil = trim( a0_pupil, 400 )

    im = ax[0].imshow( a0_pupil, cmap='gray' )
    ax[0].set_title('Pupil lens image')


    # compute the field at the FPM exit pupil and show its phase 
    # (note: not in detector pixel sampling; HLC pupil is 309 pixels across here)

    lamc = 0.575        # central wavelength of band

    print( "Computing front end phase map" )
    params = {'end_at_fpm_exit_pupil':1, 'use_errors':1, 'polaxis':np.abs(polaxis), 'use_fpm':0, 'use_lyot_stop':0, 'use_field_stop':0 }
    field, dx = proper.prop_run( 'roman_preflight', lamc, 512, QUIET=True, PASSVALUE=params )
    pupil = np.abs(field)
    pupil = pupil > 0.5*np.max(pupil)
    frontend_phase = np.angle(field) * pupil
    frontend_phase = trim( frontend_phase, 400 )
    proper.prop_fits_write( "frontend_phase.fits", frontend_phase )

    im = ax[1].imshow( frontend_phase, cmap='gray' )
    ax[1].set_title('Front end phase')


    # compute the field at the exit pupil of the final focus with the pinhole, to get the backend phase
    # (note: not in detector pixel sampling; HLC pupil is 309 pixels across here, backend pupil is greater due to larger stop)

    print( "Computing back end phase map" )
    params = {'end_at_exit_pupil':1, 'pinhole_diam_m':3.0e-6, 'use_errors':1, 'polaxis':np.abs(polaxis), 'use_fpm':0, 'use_lyot_stop':0, 'use_field_stop':0}
    field, dx = proper.prop_run( 'roman_preflight', lamc, 512, QUIET=True, PASSVALUE=params )
    pupil = np.abs(field)
    pupil = pupil > 0.5*np.max(pupil)
    backend_phase = np.angle(field) * pupil
    backend_phase = trim( backend_phase, 400 )
    proper.prop_fits_write( "backend_phase.fits", backend_phase )

    im = ax[2].imshow( backend_phase, cmap='gray' )
    ax[2].set_title('Back end phase')


    # compute images for each defocus lens

    fig, ax = plt.subplots( nrows=3, ncols=4, figsize=(10,10*3/5), constrained_layout=True )

    for i in range(4):
        lens = i + 1

        print( "Defocus lens #"+str(lens) )

        print( "  Computing defocused image without DM flattening..." )
        params = {'use_errors':1, 'use_dm1':0, 'dm1_v':dm1, 'use_dm2':0, 'dm2_v':dm2, 'use_fpm':0, 'use_lyot_stop':0, 
            'use_field_stop':0, 'use_defocus_lens':lens}
        a0_defocus, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
             star_spectrum='a0v', star_vmag=2.0, output_file='testresult_a0v_defocus'+str(lens)+'_image.fits' )
        a0_defocus = trim( a0_defocus, 400 )

        print( "  Computing defocused image with DM flattening..." )
        params = {'use_errors':1, 'use_dm1':1, 'dm1_v':dm1, 'use_dm2':1, 'dm2_v':dm2, 'use_fpm':0, 'use_lyot_stop':0, 
            'use_field_stop':0, 'use_defocus_lens':lens}
        a0_defocus_flat, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
             star_spectrum='a0v', star_vmag=2.0, output_file='testresult_a0v_defocus'+str(lens)+'_flattened_image.fits' )
        a0_defocus_flat = trim( a0_defocus_flat, 400 )

        print( "  Computing defocused image with pinhole..." )
        params = {'use_errors':1, 'use_dm1':0, 'dm1_v':dm1, 'use_dm2':0, 'dm2_v':dm2, 'use_fpm':0, 'use_lyot_stop':0, 
            'use_field_stop':0, 'use_defocus_lens':lens, 'pinhole_diam_m':3.0e-6}
        a0_defocus_pinhole, a0_counts = cgisim.rcgisim( cgi_mode, cor_type, bandpass, polaxis, params, 
             star_spectrum='a0v', star_vmag=2.0, output_file='testresult_a0v_defocus'+str(lens)+'_pinhole_image.fits' )
        a0_defocus_pinhole = trim( a0_defocus_pinhole, 400 )


        im = ax[0,i].imshow( a0_defocus, cmap='gray' )
        ax[0,i].set_title('Lens '+str(lens)+' before flattening')

        im = ax[1,i].imshow( a0_defocus_flat, cmap='gray' )
        ax[1,i].set_title('Lens '+str(lens)+' after flattening')

        im = ax[2,i].imshow( a0_defocus_pinhole, cmap='gray' )
        ax[2,i].set_title('Lens '+str(lens)+' with pinhole')

    plt.show()

if __name__ == '__main__':
    testsim_defocus()

