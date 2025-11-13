----------------------------------------------------------------------
* Description of and Mask Files for Roman CGI Spectroscopy Coronagraph Design "SPC-20200617" *
----------------------------------------------------------------------
Author: A.J. Riggs (Jet Propulsion Laboratory, California Institute of Technology)
Written by A.J. Riggs on June 18, 2020.

Copyright 2020 California Institute of Technology. Government sponsorship acknowledged.

----------------------------------------------------------------------
This file package contains :

README_Spec_SPC-20200617.txt (this file)
pupil_SPC-20200617.fits
SPM_SPC-20200617_1000_rounded9.fits
SPM_SPC-20200617_1000us3x_binary.fits
FPM_res100_SPC-20200617.fits
LS_SPC-20200617_1000.fits
LS_SPC-20200617_500.fits

----------------------------------------------------------------------
Specifications of the masks

- Design is an SPLC (shaped pupil + bowtie-opening focal plane mask + Lyot stop) for the Roman CGI.
  - Spectroscopy mode 
  - 2x65-degree field of view
  - 15% spectral bandwidth
  - 730nm center wavelength of bandpass (although the design is the same for any center wavelength since the amplitude masks are achromatic. the FPM would just have to be scaled up or down for larger or smaller wavelengths)

- pupil at CGI FSM file (pupil_SPC-20200617_1000.fits)
  - Pixel-centered (FFT convention) in 1001x1001 array. Max beam diameter is 1000 pixel widths.
  - Generated with PROPER (rectangles and ellipses) using a fit by A.J. Riggs and Dwight Moody to the pupil "CGI-20200513" from Scott Rohrbach at GSFC.
  - Anti-aliased (“gray”) edges.

- Shaped pupil apodizer files 
  - File at MODELING resolution (1000x1000): SPM_SPC-20200617_1000_rounded9_gray.fits
    - Pixel-centered (FFT convention) in 1001x1001 array. Beam diameter is 1000 pixel widths. (Same resolution as pupil file.)
    - Nearly binary (0 or 1) values. Final apodizer solution was rounded to nearest 1/9. 99.98% of values are binary; the remaining 0.02% of pixels are each converted to a 3x3 sub-array with binary values.
  - File at MANUFACTURING resolution (3000x3000): SPM_SPC-20200617_1000upsamp3x_binary.fits
      - Pixel-centered (FFT convention) in 3003x3003 array. Beam diameter is 3000 pixel widths. 
      - Binary (0 or 1) values. Upsampled 3x in a smart way from the 1000x1000 SPM file to make non-binary pixels into binary 3x3 sub-arrays that avoid free-floating sub-pixels if at all possible.

  - Starting input pupil padded (eroded) by these amounts:
    -Symmetrized obscurations about the vertical axis (as defined by stored data in the file)
    - Bulk padding:
      - +/- 0.1% of pupil diameter (D) normal to each pupil obstruction (as an example the struts are 0.2% D thicker)
      - +/- 6.0 milliradians of clocking
      - +0.75%/-1.05% D pupil magnification
    - Additional outer diameter (OD) padding: -0.5% in pupil radius (= 1.0% D total loss in overall diameter) to block the primary mirror edge rolloff.
    - Additional central obscuration (COBS) and COBS tabs' padding: +0.14% D in pupil radius
    - Additional strut width padding per side: +/-0.13% D 
    - Thresholded at an amplitude of 0.99 to be binary. 

- Focal plane mask (FPM) file (FPM_SPC-20200617_res100.fits)
  - Resolution = 100 pixels per lambda_central/D. To allow user to downsample to any desired resolution.
  - Pixel-centered on 1881x1881 array. Beam diameter is 2*9.4*100 = 1880 pixel widths.
  - Bowtie-opening occulting mask (metal on glass, transmissive)
  - Opening angle(s) = 2x65 degrees
  - Inner Radius = 2.60 lambda_central/D
  - Outer Radius = 9.40 lambda_central/D
  - Fillet radius of curvature = 0.25 lambda_central/D at each joint

- Lyot stop mask files (LS_SPC-20200617_1000.fits and LS_SPC-20200617_500.fits)
  - LS_SPC-20200617_1000.fits
    - Pixel-centered (FFT convention) in 1001x1001 array. Beam diameter is 1000 pixel widths. (Same resolution as pupil file.)
  - LS_SPC-20200617_500.fits
    - Pixel-centered (FFT convention) in 501x501 array. Beam diameter is 500 pixel widths.
  - Vertically-oriented "bowtie" amplitude mask
  - Opening angle(s) = 2x88 degrees
  - Inner Radius = 41% of telescope diameter
  - Inner Radius = 89% of telescope diameter
  - Fillet radius of curvature = 3.0% of beam diameter at each joint

- Dark Hole
  - same size as FPM opening
  - IWA = 3.0 lambda_central/D
  - OWA = 9.1 lambda_central/D
  - 2x65 degrees

----------------------------------------------------------------------
Ideal (No-Aberration) Performance:
  - Core throughput = 5.34% (calculated within the half-max isophote(s) for a source 6 lambda/D off axis.
  - 1.6e-9 normalized intensity in dark hole for ideal design

