----------------------------------------------------------------------
* Description of and Mask Files for Roman CGI Multi-Star Wavefront 
  Correction (MSWC) Coronagraph Design "SPC-20200623" *
----------------------------------------------------------------------
Author: A.J. Riggs (Jet Propulsion Laboratory, California Institute of Technology)
Written on 2020-06-23 by A.J. Riggs.
Updated on 2020-07-08 by A.J. Riggs to complete the details.

Copyright 2020 California Institute of Technology. Government sponsorship acknowledged.

----------------------------------------------------------------------
This file package contains :

README_SPC-20200623.txt (this file)
pupil_SPC-20200623_982.fits
spot_grid_13_N982.fits
SPM_SPC-20200623_982_rounded9_gray.fits 
SPM_SPC-20200623_982upsamp3x_binary.fits
FPM_SPC-20200623_res6.fits
LS_SPC-20200623_982.fits
LS_SPC-20200623_491.fits
LS_SPC-20200623_120.fits
 
----------------------------------------------------------------------
Specifications of the masks

NOTE: All masks except for the shaped pupil mask have the same specs as the SPC-WFOV design SPC-20200610. In other words, this SPM re-uses the FPM and Lyot stop from the SPC-WFOV design.

- Design is an SPLC (shaped pupil + opaque annular-opening focal plane mask + Lyot stop) for the WFIRST CGI.
  - Wide field of view (FOV) imaging mode 
  - 360-degree field of view
  - 10% spectral bandwidth
  - 825nm center wavelength of bandpass (although the design is the same for any center wavelength since the amplitude masks are achromatic; the FPM would just have to be scaled up or down for larger or smaller wavelengths)

- pupil at CGI FSM file (pupil_SPC-20200623_982.fits)
  - Pixel-centered (FFT convention) in 983x983 array. Max beam diameter is 982 pixel widths.
  - Generated with PROPER (rectangles and ellipses) using a fit by A.J. Riggs and Dwight Moody to the pupil "CGI-20200513" from Scott Rohrbach at GSFC.
  - Anti-aliased (“gray”) edges.

- Underlying grid of spots applied to pupil before optimizing SPM (spot_grid_13_N982.fits)

- Shaped pupil apodizer files 
  - File at MODELING resolution (982x982): SPM_SPC-20200623_982_rounded9_gray.fits
    - Pixel-centered (FFT convention) in 983x983 array. Beam diameter is 982 pixel widths. (Same resolution as pupil file.)
    - Nearly binary (0 or 1) values. Final apodizer solution was rounded to nearest 1/9. 99.65% of values are binary; the remaining 0.35% of pixels are each converted to a 3x3 sub-array with binary values.
  - File at MANUFACTURING resolution (2946x2946): SPM_SPC-20200623_982upsamp3x_binary.fits
      - Pixel-centered (FFT convention) in 2949x2949 array. Beam diameter is 2946 pixel widths. 
      - Binary (0 or 1) values. Upsampled 3x in a smart way from the 982x982 SPM file to make non-binary pixels into binary 3x3 sub-arrays that avoid free-floating sub-pixels.
  - Starting input pupil padded (eroded) by these amounts:
    -Symmetrized obscurations about the vertical axis (as defined by stored data in the file)
    - Bulk padding:
      - +/- 0.1% of pupil diameter (D) normal to each pupil obstruction (as an example the struts are 0.2% D thicker)
      - +/- 6.0 milliradians of clocking
      - +0.75%/-1.05% D pupil magnification
    - Additional outer diameter (OD) padding: -0.5% in pupil radius (= 1.0% D total loss in overall diameter) to block the primary mirror edge rolloff.
    - Additional central obscuration (COBS) and COBS tabs' padding: +0.14% in pupil radius
    - Additional strut width padding: +/-0.13%
    - Thresholded at an amplitude of 0.99 to be binary. 

- Focal plane mask (FPM) file (FPM_SPC-20200623_res6.fits)
  - This is mask is generated with two concentric circles. You may want just to generate it yourself.
  - Resolution = 6 pixels per lambda_central/D.
  - Pixel-centered on 245x245 array. Mask opening diameter is 2*20.4*6=244.8 pixel widths.
  - Annular-opening occulting mask (opaque metal on glass, transmissive)
  - Opening angle = 360 degrees
  - Inner Radius = 5.60 lambda_central/D
  - Outer Radius = 20.40   lambda_central/D

- Lyot stop mask files
  - LS_SPC-20200623_982.fits: Pixel-centered (FFT convention) in 983x983 array. Beam diameter is 982 pixel widths. (Same resolution as pupil file.)
  - LS_SPC-20200623_491.fits: Pixel-centered (FFT convention) in 492x492 array. Beam diameter is 491 pixel widths.
  - LS_SPC-20200623_120.fits: Pixel-centered (FFT convention) in 121x121 array. Beam diameter is 120 pixel widths.
  - Amplitude-only mask with anti-aliased edges
  - Obscurations symmetrized about the vertical axis (as given in the file)
  - Inner circle diameter = 36% of telescope diameter, centered on origin
  - Outer circle diameter = 91% of telescope diameter,  centered on origin
  - Strut widths = 3.2% of telescope diameter. Because the SPM's image is spatially filtered at the Lyot (pupil) plane, the struts do not have to be oversized or aligned exactly. The Lyot stop struts for SPC-20200623 are primarily there to give better structural support.
  - Fillet radii = 2.0% telescope diameter. Fillets are added at every joint for structural stability. 

- Dark Hole
  - Same size of FPM opening (5.6-20.4 lambda_central/D)
  - IWA = 5.9 lambda_central/D
  - OWA = 20.1 lambda_central/D
  - 360 degrees

----------------------------------------------------------------------
Ideal (No-Aberration) Performance:
  - (FWHM) Core throughput = 4.43% (calculated within the half-max isophote for a source 12 lambda/D off axis).
  - 1.0e-9 average raw contrast from IWA to OWA
