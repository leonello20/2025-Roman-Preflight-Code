----------------------------------------------------------------------
* Description of and Mask Files for Roman CGI Wide-FOV Coronagraph Design "SPC-20200610" *
----------------------------------------------------------------------
Author: A.J. Riggs (Jet Propulsion Laboratory, California Institute of Technology)
Written on 2020-06-15 by A.J. Riggs.

Copyright 2020 California Institute of Technology. Government sponsorship acknowledged.

----------------------------------------------------------------------
This file package contains :

readme_SPC-20200610.txt (this file)
pupil_SPC-20200610_1000.fits
SPM_SPC-20200610_1000_rounded9_gray.fits 
SPM_SPC-20200610_1000upsamp3x_binary.fits
FPM_SPC-20200610_res6.fits
LS_SPC-20200610_1000.fits
 
----------------------------------------------------------------------
Specifications of the masks

- Design is an SPLC (shaped pupil + opaque annular-opening focal plane mask + Lyot stop) for the WFIRST CGI.
  - Wide field of view (FOV) imaging mode 
  - 360-degree field of view
  - 10% spectral bandwidth
  - 825nm center wavelength of bandpass (although the design is the same for any center wavelength since the amplitude masks are achromatic; the FPM would just have to be scaled up or down for larger or smaller wavelengths)

- pupil at CGI FSM file (pupil_SPC-20200610_1000.fits)
  - Pixel-centered (FFT convention) in 1001x1001 array. Max beam diameter is 1000 pixel widths.
  - Generated with PROPER (rectangles and ellipses) using a fit by A.J. Riggs and Dwight Moody to the pupil "CGI-20200513" from Scott Rohrbach at GSFC.
  - Anti-aliased (“gray”) edges.

- Shaped pupil apodizer files 
  - File at MODELING resolution (1000x1000): SPM_SPC-20200610_1000_rounded9_gray.fits
    - Pixel-centered (FFT convention) in 1001x1001 array. Beam diameter is 1000 pixel widths. (Same resolution as pupil file.)
    - Nearly binary (0 or 1) values. Final apodizer solution was rounded to nearest 1/9. 99.65% of values are binary; the remaining 0.35% of pixels are each converted to a 3x3 sub-array with binary values.
  - File at MANUFACTURING resolution (3000x3000): SPM_SPC-20200610_1000upsamp3x_binary.fits
      - Pixel-centered (FFT convention) in 3003x3003 array. Beam diameter is 3000 pixel widths. 
      - Binary (0 or 1) values. Upsampled 3x in a smart way from the 1000x1000 SPM file to make non-binary pixels into binary 3x3 sub-arrays that avoid free-floating sub-pixels.
  - Starting input pupil padded (eroded) by these amounts:
    -Symmetrized obscurations about the vertical axis (as defined by stored data in the file)
    - Bulk padding:
      - +/- 0.1% of pupil diameter (D) normal to each pupil obstruction (as an example the struts are 0.4% D thicker)
      - +/- 6.0 milliradians of clocking
      - +0.75%/-1.05% D pupil magnification
    - Additional outer diameter (OD) padding: -0.5% in pupil radius (= 1.0% D total loss in overall diameter) to block the primary mirror edge rolloff.
    - Additional central obscuration (COBS) and COBS tabs' padding: +0.14% in pupil radius
    - Additional strut width padding: +/-0.13%
    - Thresholded at an amplitude of 0.99 to be binary. 

- Focal plane mask (FPM) file (FPM_SPC-20200610_res6.fits)
  - This is mask is generated with two concentric circles. You may want just to generate it yourself.
  - Resolution = 6 pixels per lambda_central/D.
  - Pixel-centered on 245x245 array. Mask opening diameter is 2*20.4*6=244.8 pixel widths.
  - Annular-opening occulting mask (opaque metal on glass, transmissive)
  - Opening angle = 360 degrees
  - Inner Radius = 5.60 lambda_central/D
  - Outer Radius = 20.40   lambda_central/D

- Lyot stop mask file (LS_SPC-20200610_1000.fits)
  - Pixel-centered (FFT convention) in 1001x1001 array. Beam diameter is 1000 pixel widths. (Same resolution as pupil file.)
  - Amplitude-only mask with anti-aliased edges
  - Obscurations symmetrized about the vertical axis (as given in the file)
  - Inner circle diameter = 36% of telescope diameter, centered on origin
  - Outer circle diameter = 91% of telescope diameter,  centered on origin
  - Strut widths = 3.2% of telescope diameter. Because the SPM's image is spatially filtered at the Lyot (pupil) plane, the struts do not have to be oversized or aligned exactly. The Lyot stop struts for SPC-20200610 are primarily there to give better structural support.
  - Fillet radii = 2.0% telescope diameter. Fillets are added at every joint for structural stability. 

- Dark Hole
  - Same size of FPM opening (5.6-20.4 lambda_central/D)
  - IWA = 5.9 lambda_central/D
  - OWA = 20.1 lambda_central/D
  - 360 degrees

----------------------------------------------------------------------
Ideal (No-Aberration) Performance:
  - (FWHM) Core throughput = 4.25% (calculated within the half-max isophote for a source 12 lambda/D off axis).
  - 1.0e-9 average raw contrast from IWA to OWA
