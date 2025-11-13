import os
import time
import shutil
import numpy as np
import astropy.io.fits as pyfits
import proper
from emccd_detect.emccd_detect import EMCCDDetectBase
#from arcticpy.traps import TrapInstantCapture
#from arcticpy.roe import ROE
#from arcticpy.ccd import CCD
import cgisim as cgisim
import roman_preflight_proper
from .lowfs import lowfs

# Created by John Krist (JPL)
#
# 8 Oct 2020 - v2.0.4 - J.Krist - fixed bug when adding EMCCD to LOWFS image
# 2 Nov 2020 - v2.0.5 - J.Krist - increased output image size when using fpm_array or field_stop_array options to PROPER model
# 2 Apr 2021 - v3.0 - J.Krist - added optional coronagraph modes; updated bandpasses; single
#     frame images are now returned rather than broadband and subband images, due to incorporation
#     of specified FWHMs for each filter
# 17 May 2021 - v3.0.1 - J.Krist - updated call to emccd_detect to use v2.2
# 29 June 2021 - v3.0.2 - J.Krist - fixed backwards logic for create_big_image 
# 30 June 2021 - v3.0.3 - J.Krist - added hlc_band1 and spc-spec to list of valid modes 
# 2 Aug 2021 - V3.0.4 - J.Krist - edits to manual only
# 20 Sept 2021 - v3.0.5 - J.Krist - added option to return or write out E-fields instead of intensities (polaxis=0 or -10, only)
# 7 Oct 2021 - v3.0.6 - J.Krist - added wavefront reference radii to FITS headers (only for 'excam_efield' mode)
# 16 Nov 2021 - v3.0.7 - J.Krist - added spc-spec_band2_rotated and spc-spec_band3_rotated options; added filters 3a and 3b to
#                                   valid filter list for spc-spec_band2 and spc-spec_band2_rotated
# Sept 2022 - v3.1 - J.Krist - added zwfs to list of valid names; added option to add traps in call to emccd_detect
# March 2023 - v3.2 - J.Krist - modified neutral density specification, use measured ND curves
# August 2024 - v4.0 - J.Krist - changed prescription to roman_preflight; removed support for traps due to problems running arcticpy
#########################################################################################
# LOWFS fast-readout smearing code originally from Nikta Amiri

def smear( image ):

    n = image.shape[0]
    frame = np.zeros((n,n))

    # only ratio of these times is important
    exposureTime = 500.0
    reverseSmearTime = 75.0
    readOutTime = 325.

    # reverse smear
    for i in range(n):
        frame[0:i+1,:] += image[n-i-1:n,:] * reverseSmearTime/n

    # image integration 
    frame += image * exposureTime

    # read out
    for i in range(n):
        frame[0:n-(i+1),:] += image[i+1:n,:] * readOutTime/n

    frame = frame / np.sum(frame) * np.sum(image)

    return frame

#########################################################################################
def model_for_Roman( date ):
    """
    Return arcticpy parameters for preset CTI model for the Roman Space
    Telescope Main Camera.
    The returned objects are ready to be passed to add_cti() or remove_cti()
    for parallel clocking.

    Parameters
    ----------
    date : float
        The date, in year+decimal format. Should not be before assumed launch date of May 2027.
    Returns
        Lists of trap densities and release times for 5 trap species.
    """

    # Dates are in year + decimal
    launch_date = 2027.35

    serial = 10     # the serial readout frequency in MHz
    temp = 170      # camera temp in K

    # Trap densities
    # Primary reference is Bush (2020), Tables 5 and 7; mission duration = 5.25 yrs
    # Species: VV--, Si-E, Si-U, VV-, Si-A
    trap_initial_density = np.array([5.11e-4,1.4e-3,8.7e-5,4.2e-4,2.6e-3]) # um^-3
    # trap_growth_rate units are trap density (um^-3) per year
    trap_growth_rate = np.array([1.6e-4,8.9e-4,3.9e-5,1.3e-4,1.7e-3])
    trap_densities = trap_initial_density + trap_growth_rate * (date - launch_date)
    # trap densities in traps per pixel unit, assuming 13 um^3 charge cloud volume
    trap_densities_pix = 13 * trap_densities

    # Trap release times
    k = 8.617343e-5     # eV / K
    kb = 1.381e-23      # mks units
    hconst = 6.626e-34  # mks units
    Eg = 1.1692 - (4.9e-4)*temp*temp/(temp+655)
    me = 9.109e-31      # kg
    mlstar = 0.1963 * me
    mtstar = 0.1905 * 1.1692 * me / Eg
    mstardc = 3.302 * (mtstar*mtstar*mlstar)**0.333333
    Nc = 2*(2*3.1416*mstardc*kb*temp/(hconst**2))**1.5
    vth = np.sqrt(3*kb*temp/mstardc) # m/s
    # species: VV--, Si-E, Si-U, VV-, Si-A
    cross_secs_cm2 = np.array([2.6e-15,3.7e-14,8.7e-15,2.0e-15,6.1e-15]) # cm^2
    cross_secs_m2 = (1e-4) * cross_secs_cm2
    energies = np.array([0.235, 0.475, 0.37,0.42,0.165]) # eV
    trap_release_times = 1/(cross_secs_m2 * vth * Nc)
    trap_release_times = trap_release_times * np.exp(energies/(k*temp)) # seconds
    # trap release times in pixel units
    # the parallel readout time interval is 1.25e-6 s
    trap_release_times_pix = trap_release_times/(1.25e-6 + 1024/(serial * 1e6))

    # Assemble variables to pass to add_cti()
    traps = [
        TrapInstantCapture(
            density=trap_densities_pix[0], release_timescale=trap_release_times_pix[0]
        ),
        TrapInstantCapture(
            density=trap_densities_pix[1], release_timescale=trap_release_times_pix[1]
        ),
        TrapInstantCapture(
            density=trap_densities_pix[2], release_timescale=trap_release_times_pix[2]
        ),
        TrapInstantCapture(
            density=trap_densities_pix[3], release_timescale=trap_release_times_pix[3]
        ),
        TrapInstantCapture(
            density=trap_densities_pix[4], release_timescale=trap_release_times_pix[4]
        ),
    ]

    return traps

#########################################################################################
def rcgisim( cgi_mode='excam', cor_type='hlc', bandpass='1', polaxis=0, param_struct={}, **kwargs ): 

    # copy PROPER prescription from roman_preflight_proper package into local directory

    prescription = 'roman_preflight'
    prescription_file = roman_preflight_proper.lib_dir + '/' + prescription + '.py'

    try: 
        shutil.copy( prescription_file, './.' )
    except IOError as e:
        raise IOError( "Unable to copy prescription to current directory. %s" % e )
 
    info_dir = cgisim.lib_dir + '/cgisim_info_dir/'
    star_spectrum = 'a0v'
    star_vmag = 0.0
    nd = 0      # integer: 1, 3, or 4 (0 = no ND, the default); this is the ND filter identifier, NOT the amount of ND
    input_save_file = ""
    output_save_file = ""
    output_file = ""
    ccd = {}
    sampling_m = 13.0e-6
    no_integrate_pixels = False
    create_big_image = False

    use_pupil_lens = 0
    use_defocus_lens = 0

    if "info_dir" in kwargs: info_dir = kwargs.get("info_dir")
    if "star_spectrum" in kwargs: star_spectrum = kwargs.get("star_spectrum")
    if "star_vmag" in kwargs: star_vmag = kwargs.get("star_vmag")
    if "nd" in kwargs: nd = kwargs.get("nd")
    if "output_save_file" in kwargs: output_save_file = kwargs.get("output_save_file")
    if "input_save_file" in kwargs: input_save_file = kwargs.get("input_save_file")
    if "output_file" in kwargs: output_file = kwargs.get("output_file")
    if "ccd" in kwargs: ccd = kwargs.get("ccd")
    if "sampling_m" in kwargs: sampling_m = kwargs.get("sampling_m")
    if "no_integrate_pixels" in kwargs: no_integrate_pixels = kwargs.get("no_integrate_pixels")

    # cgi_mode includes 'excam', 'spec', 'lowfs', 'excam_efield'
    # cor_type includes 'hlc', 'spc-spec_band2', 'spc-spec_band3', 'spc-wide', 'zwfs', and others

    if cgi_mode == 'excam_efield':
        if input_save_file != "" or output_save_file != "":
            raise Exception('ERROR: cannot specify input_save_file or output_save_file when cgi_mode==excam_efield')
        if ccd != {}:
            raise Exception('ERROR: cannot specify CCD parameters when cgi_mode==excam_efield')

    phase_retrieval = 0    # gets set to 1 if pupil imaging lens or defocus lens is used

    # some parameters to the PROPER prescription are not allowed when
    # calling it from cgisim; also, check to see if any phase retrieval
    # modes are being used (pupil lens, defocus lens)

    if param_struct:
        if 'cor_type' in param_struct: 
            raise Exception('ERROR: cor_type must be specified in call, not in parameter structure')
        if 'end_at_fpm_exit_pupil' in param_struct: 
            raise Exception('ERROR: end_at_fpm_exit_pupil not allowed')
        if 'end_at_fsm' in param_struct: 
            raise Exception('ERROR: end_at_fsm not allowed')
        if 'polaxis' in param_struct:
            raise Exception('ERROR: polaxis must be specified in call, not in parameter structure')
        if 'final_sampling_lam0' in param_struct:
            raise Exception('ERROR: final_sampling_lam0 is not allowed in parameter structure')
        if 'final_sampling_m' in param_struct:
            raise Exception('ERROR: final_sampling_m is not allowed in parameter structure')
        if 'use_pupil_lens' in param_struct:
            if param_struct["use_pupil_lens"] != 0:
                phase_retrieval = 1
                use_pupil_lens = param_struct["use_pupil_lens"]
        if 'use_defocus_lens' in param_struct:
            if param_struct["use_defocus_lens"] != 0:
                phase_retrieval = 1
                use_defocus_lens = param_struct["use_defocus_lens"]
        if 'fpm_array' in param_struct or 'field_stop_array' in param_struct:
            create_big_image = True        

    valid_cgi_modes = ['excam', 'spec', 'lowfs', 'excam_efield']
    valid_cor_types = ['hlc', 'hlc_band1', 'spc-spec', 'spc-spec_band2', 'spc-spec_band3', 'spc-wide', 'spc-wide_band4', 
                       'spc-wide_band1', 'spc-mswc', 'spc-mswc_band4','spc-mswc_band1', 'zwfs',
                       'hlc_band2', 'hlc_band3', 'hlc_band4', 'spc-spec_rotated', 'spc-spec_band2_rotated', 'spc-spec_band3_rotated']

    # if user provided the name of an input file, use the fields in that
    # rather than generating new fields with PROPER; use particular
    # parameters stored in the file header

    original_bandpass = bandpass

    if input_save_file != "":
        print( "Reading previously generated fields from " + input_save_file )
        try:
            image, h = pyfits.getdata(input_save_file, header = True, ignore_missing_end = True)
        except IOError:
            raise IOError("Unable to read FITS image %s. Stopping" %(input_save_file))
        cgi_mode = h["cgi_mode"]
        cor_type = h["cor_type"]
        phase_retrieval = h["phaseret"]
        bandpass = h["bandpass"]
        if bandpass == '2_spec': original_bandpass = '2'
        if bandpass == '3_spec': original_bandpass = '3'
        if bandpass == '1_wide': original_bandpass = '1'
        if bandpass == '4_wide': original_bandpass = '4'
        if bandpass == '1_mswc': original_bandpass = '1'
        if bandpass == '4_mswc': original_bandpass = '4'
        polaxis = h["polaxis"]
        sampling_m = h["samp_m"]
    elif cgi_mode == 'spec':
        bandpass = bandpass + '_spec'

    if cgi_mode not in valid_cgi_modes:
        raise Exception('ERROR: Requested mode does not match any available mode')

    if cor_type not in valid_cor_types:
        raise Exception('ERROR: Requested coronagraph does not match any available types')

    if cgi_mode != 'excam' and cgi_mode != 'excam_efield' and phase_retrieval != 0:
        raise Exception("ERROR: Phase retrieval supported only in 'excam' or 'excam_efield' mode")

    # get bandpass parameters (min, max wavelength, number of wavelengths, etc)

    mode_data, bandpass_data = cgisim.cgisim_read_mode( cgi_mode, cor_type, bandpass, info_dir )

    lam0_um = bandpass_data["lam0_um"]
    nlam = bandpass_data["nlam"]
    lam_um = np.linspace( bandpass_data["minlam_um"], bandpass_data["maxlam_um"], nlam ) 
    sampling_lamref_div_D = mode_data['sampling_lamref_div_D']
    lamref_um = mode_data['lamref_um']
    owa_lamref = mode_data['owa_lamref']

    max_threads = 15

    if cgi_mode == 'excam' or cgi_mode == 'excam_efield':
        # detector pixel sampling in lam0/D
        detector_sampling_lam0_div_D = sampling_lamref_div_D * lamref_um / lam0_um    

        if phase_retrieval != 0:
            grid_dim_out = 511    # must be odd
        else:
            if create_big_image:
                grid_dim_out = 511
            else:
                grid_dim_out = 201

        grid_dim_out0 = grid_dim_out
    
        sampling_um = mode_data['sampling_um']

        if not no_integrate_pixels:
            # in imaging mode, integrate over finite pixels by subsampling 
            # the E-field by 7x the detector pixel size, converting 
            # to intensity, and then binning down to detector pixels
            oversampling_factor = 7
            grid_dim_out = grid_dim_out * oversampling_factor
            field_sampling_um = sampling_um / oversampling_factor 
        else:
            # don't integrate over pixels, just use the interpolated values
            field_sampling_um = sampling_um
    elif cgi_mode == 'spec':
        # in spectral mode, the images feeding the dispersion model are sampled 
        # by 0.1 lam0/D and are NOT integrated over area 
        spec_sampling_lam0_div_D = 0.1
        owa = owa_lamref * lamref_um / lam0_um
        grid_dim_out = int(np.round((2*owa + 9) / spec_sampling_lam0_div_D))
        grid_dim_out = grid_dim_out + (grid_dim_out % 2 == 0)      # force odd
        grid_dim_out0 = grid_dim_out 
    else:   # LOWFS
        # the output of the PROPER prescription is a wavefront at a pupil;
        # that will be transformed to focus and the reflective FPM applied, then
        # transformed to the LOWFS pupil image, which will be resampled, integrating
        # over pixels by oversampling then binning
        grid_dim_out0 = 51          # size of LOWFS image 
        sampling_um = mode_data['sampling_um']
 
    # polarization

    if cgi_mode == 'excam_efield' and polaxis != 0 and polaxis != -10:
        raise Exception("ERROR: Can only use polaxis=0 or polaxis=-10 for mode excam_efield.")
    
    if polaxis == 5:
        tpol = [-1,1]       # X polarizer
        polaxis_name = 'X'
    elif polaxis == -5:
        tpol = [5]          # X polarizer mean
        polaxis_name = 'X(mean)'
    elif polaxis == 6:
        tpol = [-2,2]       # Y polarizer
        polaxis_name = 'Y'
    elif polaxis == -6:
        tpol = [6]          # Y polarizer mean
        polaxis_name = 'Y(mean)'
    elif polaxis == 10: 
        tpol = [-2,-1,1,2]  # No polarizer (all polarizations)
        polaxis_name = 'X+Y'
    elif polaxis == -10: 
        tpol = [10]         # No polarizer (mean of all polarizations)
        polaxis_name = 'X+Y(mean)'
    else: 
        tpol = [polaxis]
        polaxis_name = "none"

    npol = len(tpol)

    if polaxis != 10 and polaxis != -10 and polaxis != 0:
        polarizer_transmission = 0.45
    else:
        polarizer_transmission = 1.0

    if cgi_mode == 'excam':
        params = { 'cor_type':cor_type, 'output_dim':grid_dim_out, 'final_sampling_m':field_sampling_um*1e-6, 'lam0':lam0_um, 'polaxis':0 }
    elif cgi_mode == 'excam_efield':
        params = { 'cor_type':cor_type, 'output_dim':grid_dim_out, 'final_sampling_m':field_sampling_um*1e-6, 'lam0':lam0_um, 'polaxis':0, 'save_ref_radius':0 }
    elif cgi_mode == 'spec':
        params = { 'cor_type':cor_type, 'output_dim':grid_dim_out, 'final_sampling_lam0':spec_sampling_lam0_div_D, 'lam0':lam0_um, 'polaxis':0 }
    else:  # LOWFS
        params = { 'cor_type':cor_type, 'lam0':lam0_um, 'polaxis':0, 'use_fpm':0, 'end_at_fpm_exit_pupil':1 }

    if param_struct:
        # combine default and optional structures
        params.update( param_struct )

    nfields = npol * nlam
    ilam_array = np.zeros( (nfields), dtype=int )
    lam_array = np.zeros( (nfields), dtype=np.float64 )
    params_array = []
    sampling_m_lam = np.zeros( nlam, dtype=np.float64 )

    # create copies of parameter structure, one for each wavelength & polarization pair
    k = 0
    for ipol in tpol:
        for ilam in range(nlam):
            lam_array[k] = lam_um[ilam]
            ilam_array[k] = ilam
            params_array.append( params.copy() )
            params_array[k]['polaxis'] = ipol
            if cgi_mode == 'excam_efield':
                params_array[k]['save_ref_radius'] = ilam + 1
            k = k + 1

    # compute number of fields in each prop_run_multi batch; note that when cgi_mode == excam_efield, only one 
    # polarization is used, so there will be only one batch (one set of wavelengths)

    nbatch = nfields // max_threads + (nfields % max_threads != 0)
    nfields_in_batch = np.zeros( nbatch, dtype=int )
    t = nfields % nbatch
    nfields_in_batch[:] = nfields // nbatch + 1
    nfields_in_batch[t:] = nfields // nbatch


    total_time = 0

    if input_save_file == "":    # no pre-generated fields, so generate them
        if cgi_mode != 'excam_efield':
            image = np.zeros( (nlam, grid_dim_out0, grid_dim_out0), dtype=float )
        else:
            image = np.zeros( (nlam, grid_dim_out0, grid_dim_out0), dtype=complex )

        k = 0

        for ibatch in range(nbatch):
            nfields_i = nfields_in_batch[ibatch]
            lam_i = lam_array[k:k+nfields_i]
            print( 'Computing fields ' + str(k+1) + ' to ' + str(k+nfields_i) + ' of ' + str(nfields) )
            t1 = time.time()

            fields_i, sampling_i = proper.prop_run_multi( prescription, lam_i, 1024, PASSVALUE=params_array[k:k+nfields_i], QUIET=True ) 
            if cgi_mode == 'lowfs':
                # fields_i are E-fields at pupil plane
                fields_i, oversampling_factor = lowfs( cor_type, fields_i, lam_i, grid_dim_out0 )   
            else:
                # fields_i are E-fields at final image plane
                if cgi_mode != 'excam_efield':
                    fields_i = np.abs(fields_i)**2
 
            # integrate over pixel area unless spectroscopic mode 
            if no_integrate_pixels == 0 and cgi_mode != 'spec':
                for ifld in range(nfields_i):
                    if cgi_mode != 'excam_efield':
                        # bin in intensity
                        image[ilam_array[k+ifld],:,:] += fields_i[ifld,:,:].reshape((grid_dim_out0,oversampling_factor,grid_dim_out0,oversampling_factor)).mean(3).mean(1) * oversampling_factor**2
                    else:
                        # bin real & imaginary parts separately
                        rval = fields_i[ifld,:,:].real.reshape((grid_dim_out0,oversampling_factor,grid_dim_out0,oversampling_factor)).mean(3).mean(1) * oversampling_factor
                        ival = fields_i[ifld,:,:].imag.reshape((grid_dim_out0,oversampling_factor,grid_dim_out0,oversampling_factor)).mean(3).mean(1) * oversampling_factor
                        image[ilam_array[k+ifld],:,:] = rval + 1j * ival
            else:
                image[ilam_array[k:k+nfields_i],:,:] += fields_i[:,:,:]

            fields_i = 0

            t2 = time.time()
            print( 'Time to compute %d fields = %6.1f minutes.' % (nfields_i, (t2-t1)/60.0) )
            total_time = total_time + (t2 - t1)
            k = k + nfields_i

        if cgi_mode != 'excam_efield':
            image /= npol        # [nlam,n,n]
        else:
            # read in wavefront reference radii, one for each wavelength
            ref_radius = np.zeros( (nlam), dtype=float )
            for ilam in range(nlam):
                fname = 'ref_radius_' + str(ilam+1) + '.dat'
                with open( fname, 'rb' ) as file1:
                    ref_radius[ilam] = np.fromfile( file1, dtype=float, count=1 )
                os.remove( fname )

        print( 'Total time to compute = %6.1f minutes.' % (total_time/60.0) )

        # save results without spectral weighting for use later, if output_save_file specified
        if output_save_file != "":
            h = pyfits.PrimaryHDU( image )
            h.header.set( 'cgi_mode', cgi_mode )
            h.header.set( 'cor_type', cor_type )
            h.header.set( 'phaseret', phase_retrieval )
            h.header.set( 'bandpass', bandpass )
            h.header.set( 'polaxis', polaxis )
            if cgi_mode != 'spec':
                h.header.set( 'samp_m', sampling_um*1e-6 )
            else: 
                h.header_set( 'sampling', spec_sampling_lam0_div_D, 'sampling in lam0/D' )
            h.writeto( output_save_file, overwrite=True )

    # If EXCAM: 
    #   7 wavelengths are used for band 1 to span 10% bandpass.
    #   11 wavelengths are used for band 2 to span 15% bandpass.
    #   11 wavelengths are used for band 3 to span 15% bandpass.
    #   7 wavelengths are used for band 4 to span 10% bandpass.
    #   5 wavelengths are used for a subband

    total_counts = 0.0

    if cgi_mode == 'excam':
        dlam_um = lam_um[1] - lam_um[0]
        images = 0
        exptime = 1.0   # 1.0 sec exposure (overridden if ccd info specified)
        for ilam in range(nlam):
            lam_start_um = lam_um[ilam] - 0.5 * dlam_um
            lam_end_um = lam_um[ilam] + 0.5 * dlam_um
            bandpass_i = 'lam' + str(lam_start_um*1000) + 'lam' + str(lam_end_um*1000)
            counts = polarizer_transmission * cgisim.cgisim_get_counts( star_spectrum, bandpass, bandpass_i, nd, star_vmag, 'V', 'excam', info_dir )
            images += counts * image[ilam,:,:]
            total_counts += counts
    elif cgi_mode == 'spec':
        exptime = 1.0
        dlam_um = lam_um[1] - lam_um[0]
        total_counts = np.zeros(nlam)
        for ilam in range(nlam):
            lam_start_um = lam_um[ilam] - 0.5 * dlam_um
            lam_end_um = lam_um[ilam] + 0.5 * dlam_um
            bandpass_i = 'lam' + str(lam_start_um*1000) + 'lam' + str(lam_end_um*1000)
            counts = polarizer_transmission * cgisim.cgisim_get_counts( star_spectrum, bandpass, bandpass_i, nd, star_vmag, 'V', 'spec', info_dir )
            image[ilam,:,:] *= counts 
            total_counts[ilam] = counts
        images = image
    elif cgi_mode == 'lowfs':
        exptime = 1.0
        dlam_um = lam_um[1] - lam_um[0]
        total_counts = 0.0
        images = 0.0
        lowfs_counts = np.zeros( (nlam) )
        im_counts = np.zeros( (nlam) )
        for ilam in range(nlam):
            lam_start_um = lam_um[ilam] - 0.5 * dlam_um
            lam_end_um = lam_um[ilam] + 0.5 * dlam_um
            bandpass_i = 'lam' + str(lam_start_um*1000) + 'lam' + str(lam_end_um*1000)
            counts = cgisim.cgisim_get_counts( star_spectrum, bandpass, bandpass_i, nd, star_vmag, 'V', 'lowfs', info_dir )
            im = image[ilam,:,:] * counts
            images += im
            im_counts[ilam] = np.sum(im)
            lowfs_counts[ilam] = counts 
            total_counts += counts

    if ccd != {}:
        # default values match requirements, except QE, which is year 0 curve (already accounted for in counts)
        full_well_serial = 100000.0         # full well for serial register; 90K is requirement, 100K is CBE
        full_well = 60000.0                 # image full well; 50K is requirement, 60K is CBE
        dark_rate = 0.00056                 # e-/pix/s; 1.0 is requirement, 0.00042/0.00056 is CBE for 0/5 years
        cic_noise = 0.01                    # e-/pix/frame; 0.1 is requirement, 0.01 is CBE
        read_noise = 100.0                  # e-/pix/frame; 125 is requirement, 100 is CBE
        cr_rate = 0                         # hits/cm^2/s (0 for none, 5 for L2) 
        gain = 1000.0                       # EM gain
        bias = 0
        qe = 1.0                            # qe already applied in count rates
        pixel_pitch = 13e-6                 # detector pixel size in meters
        apply_smear = True                  # (LOWFS only) Apply fast readout smear?  
        e_per_dn = 1.0                      # post-multiplied electrons per data unit
        nbits = 14                          # ADC bits
        numel_gain_register = 604           # Number of gain register elements 
        #use_traps = 0                       # include CTI impact of traps
        date = 2028.0                       # decimal year of observation

        if 'full_well_serial' in ccd: full_well_serial = ccd['full_well_serial'] 
        if 'full_well' in ccd: full_well = ccd['full_well'] 
        if 'dark_rate' in ccd: dark_rate = ccd['dark_rate']
        if 'cic_noise' in ccd: cic_noise = ccd['cic_noise']
        if 'read_noise' in ccd: read_noise = ccd['read_noise']
        if 'cr_rate' in ccd: cr_rate = ccd['cr_rate']
        if 'gain' in ccd: gain = ccd['gain']
        if 'exptime' in ccd: exptime = ccd['exptime']
        if 'bias' in ccd: bias = ccd['bias']
        if 'apply_smear' in ccd: apply_smear = ccd['apply_smear']
        if 'e_per_dn' in ccd: e_per_dn = ccd['e_per_dn']
        if 'nbits' in ccd: nbits = ccd['nbits']
        if 'numel_gain_register' in ccd: numel_gain_register = ccd['numel_gain_register']
        if 'use_traps' in ccd:      #use_traps = ccd['use_traps']
            raise Exception('ERROR: use_traps not supported in this version of cgisim')
        if 'date' in ccd: date = ccd['date']

        emccd = EMCCDDetectBase( em_gain=gain, full_well_image=full_well, full_well_serial=full_well_serial,
                             dark_current=dark_rate, cic=cic_noise, read_noise=read_noise, bias=bias,
                             qe=qe, cr_rate=cr_rate, pixel_pitch=pixel_pitch, eperdn=e_per_dn,
                             numel_gain_register=numel_gain_register, nbits=nbits )

        if cgi_mode == 'lowfs':
            print( "Adding CCD to LOWFS image" )
            if apply_smear:
                images[:,:] = smear( images )
        elif cgi_mode == 'excam':
            print( "Adding CCD to EXCAM image" )

        #if use_traps != 0: 
        #    traps = model_for_Roman( date )  
        #    ccd = CCD(well_fill_power=0.58, full_well_depth=full_well)
        #    roe = ROE()
        #    emccd.update_cti( ccd=ccd, roe=roe, traps=traps, express=1 )

        images[:,:] = emccd.sim_sub_frame( images, exptime ).astype(float)

    # output final images to FITS file, if output_file specified
    if output_file != "":
        if cgi_mode == 'excam_efield':
            ncomp = 2
        else:
            ncomp = 1

        for icomp in range(ncomp):
            if cgi_mode != 'excam_efield':
                output_filename = output_file
                print( "Writing result to " + output_filename )
                h = pyfits.PrimaryHDU( images )
            else:
                if icomp == 0:
                    output_filename = output_file + '_real.fits'
                    print( "Writing result to " + output_filename )
                    h = pyfits.PrimaryHDU( image.real )
                else:
                    output_filename = output_file + '_imag.fits'
                    print( "Writing result to " + output_filename )
                    h = pyfits.PrimaryHDU( image.imag )

            h.header.set( 'cgi_mode', cgi_mode )
            h.header.set( 'cor_type', cor_type )

            if phase_retrieval != 0: 
                h.header.set( 'phaseret', phase_retrieval, "In phase retrieval mode (0=none)" )
                if use_pupil_lens != 0: h.header.set( 'pup_lens', use_pupil_lens, "Use pupil imaging lens" )
                if use_defocus_lens != 0: h.header.set( 'dfo_lens', use_defocus_lens, "Defocusing lens number (1-4)" )

            h.header.set( 'bandpass', original_bandpass )
            h.header.set( 'polariz', polaxis_name )

            if cgi_mode != 'spec':
                h.header.set( 'samp_m', sampling_m, "image sampling in meters" )
            else:
                h.header.set( 'sampling', spec_sampling_lam0_div_D, 'image sampling in lam0/D' )

            if cgi_mode == 'spec' or cgi_mode == 'excam_efield':
                h.header.set( 'nlam', nlam, 'number of wavelengths' )
                h.header.set( 'lam0_um', lam0_um, 'center wavelength in microns' )
                h.header.set( 'minlam', bandpass_data["minlam_um"], 'min wavelength in microns' )
                h.header.set( 'maxlam', bandpass_data["maxlam_um"], 'max wavelength in microns' )
                h.header.set( 'dlam', (bandpass_data["maxlam_um"] - bandpass_data["minlam_um"])/(nlam-1), 'spacing between wavelengths in microns' )

            if ccd != {}:  # only supported in EXCAM and LOWFS
                h.header.set( 'fwserial', full_well_serial, "Serial register full well e- capacity" )
                h.header.set( 'fwimage', full_well, "Image full well e- capacity" )
                h.header.set( 'darkrate', dark_rate, "Dark rate in e-/pix/s" )
                h.header.set( 'cicnoise', cic_noise, "CIC noise in e-/pix/frame" )
                h.header.set( 'readnois', read_noise, "Read noise in e-/pix/frame" )
                h.header.set( 'bias', bias, "bias in e-/frame" )
                h.header.set( 'gain', gain, "multiplier gain in e-/photon" )
                h.header.set( 'e_per_dn', e_per_dn, "electrons per data unit" )
                h.header.set( 'nbits', nbits, "ADC bits" )
                h.header.set( 'nelgnreg', numel_gain_register, "Number of gain register elements" )

            if cgi_mode != 'excam_efield':
                h.header.set( 'starspec', star_spectrum, "Star spectral type" )
                h.header.set( 'starvmag', star_vmag, "Star V magnitude" )
            else:
                for ilam in range(nlam):
                    h.header.set( 'refrad'+str(ilam), ref_radius[ilam], "Wavefront reference radius (m), wavelength "+str(ilam) )
       
            if cgi_mode == 'excam' or cgi_mode == 'lowfs':
                h.header.set( 'exptime', exptime, "Exposure time (sec)" )
            else:
                h.header.set( 'exptime', 1.0, "Exposure time (sec)" )

            if cgi_mode != 'lowfs' and cgi_mode != 'excam_efield':
                if nd != 0:
                    h.header.set( 'ND', nd, "Neutral density filter (0, 1, 3, or 4)" )
                if cgi_mode == 'excam':
                    h.header.set( 'counts', total_counts, "phot/sec at primary in bandpass" ) 
                else:
                    for i in range(len(total_counts)):
                        h.header.set( 'counts'+str(i), total_counts[i], "phot/sec at primary for each band" ) 
            elif cgi_mode != 'excam_efield':
                h.header.set( 'counts', total_counts, "phot/sec at primary" ) 
                for i in range(nlam):
                    h.header.set( 'counts'+str(i), lowfs_counts[i], "phot/sec at primary for each component subband" ) 
                for i in range(nlam):
                    h.header.set( 'iphot'+str(i), im_counts[i], "phot/sec in image at each component subband" ) 
            
            h.writeto( output_filename, overwrite=True )

    if cgi_mode != 'excam_efield': 
        return images, total_counts
    else:
        return image, total_counts

