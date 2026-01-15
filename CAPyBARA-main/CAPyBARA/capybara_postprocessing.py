#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 17:58:41 2024

@author: echoquet
"""

import os
import numpy as np
from astropy.io import fits
from astropy.modeling.functional_models import AiryDisk2D
import scipy.ndimage as ndimage
from scipy.stats import t
import matplotlib.pyplot as plt
import imutils
plt.rcParams['image.origin'] = 'lower'

def create_annulus(rIn, rOut, shape, cent=None):
    """ Creates a boolean array with the pixels within rIn and rOut at True.

    Parameters
    ----------
    rIn : the inner radius of the annulus in pixels.
    rOut : the outer radius of the annulus in pixels.
    shape : shape of the output array [pix, pix].

    Returns
    -------
    mask : 2D Array of booleans, True within rIn and rOut, False outside.

    """
    if len(shape) != 2:
        raise TypeError('dim should be list of 2 elements')

    if cent is None:
        cent = np.array(shape) // 2
    x, y = np.indices(shape)
    rads = np.sqrt((x-cent[1])**2+(y-cent[0])**2)
    mask = (rOut > rads) & (rads >= rIn)
    return mask



def frame_rotate_interp(array, angle, center=None, mode='constant', cval=0, order=3):
    ''' Rotates a frame or 2D array.

        Parameters
        ----------
        array : Input image, 2d array.
        angle : Rotation angle (deg).
        center : Coordinates X,Y  of the point with respect to which the rotation will be
                    performed. By default the rotation is done with respect to the center
                    of the frame; central pixel if frame has odd size.
        interp_order: Interpolation order for the rotation. See skimage rotate function.
        border_mode : Pixel extrapolation method for handling the borders.
                    See skimage rotate function.

        Returns
        -------
        rotated_array : Resulting frame.

    '''
    dtype = array.dtype
    dims = array.shape
    angle_rad = -np.deg2rad(angle)

    if center is None:
        center = (np.array(dims)-1) / 2  # The minus 1 is because of python indexation at 0

    x, y = np.meshgrid(np.arange(dims[1], dtype=dtype), np.arange(dims[0], dtype=dtype))

    xp = (x-center[1])*np.cos(angle_rad) + (y-center[0])*np.sin(angle_rad) + center[1]
    yp = -(x-center[1])*np.sin(angle_rad) + (y-center[0])*np.cos(angle_rad) + center[0]

    rotated_array = ndimage.map_coordinates(array, [yp, xp], mode=mode, cval=cval, order=order)

    return rotated_array

def classical_subtraction(sci_cube, psf_cube, opt_zone=None):
    """
    Computes the mean image of the psf_cube, scales it and subtract it from 
    each sci_cube image.

    Parameters
    ----------
    sci_cube : 3D np.array
        Raw image cube.
    psf_cube : 3D np.array
        PSF image cube
    opti_zone : 2D boolean image
        Area to optimize the PSF scalling coefficients
        
    Returns
    -------
    res_cube: 3D np.array
        Residual image cube

    """
    if sci_cube.ndim != 3:
        raise TypeError('sci_cube should be an image cube')
    if psf_cube.ndim != 3:
        raise TypeError('psf_cube should be an image cube')
    if opt_zone is None:
        opt_zone = np.ones_like(psf_cube[0], dtype=bool)
    
    mean_psf = np.mean(psf_cube, axis=0)
    
    # scaling coefficients computed analytically (least squares)
    scaling_coefs = np.sum(mean_psf[None, opt_zone] * sci_cube[:, opt_zone], axis=1) / np.sum(mean_psf[opt_zone]**2)
    res_cube = sci_cube - scaling_coefs[:, None, None] * mean_psf[None, :, :]
    
    return res_cube



def compute_PC(array):
    """ Computes the principal components of a cube of images.

    Parameters
    ----------
    array : 2D NP ARRAY. Array of vectorized images, mean removed

    Returns
    -------
    pc_vec : the list of vectorized principal components
    e_values: the list of eigen values of the covariance matrix

    """
    if array.ndim != 2:
        raise TypeError('Input cube is not a 2d array')

    # Computes the covariance matrix and its Eigen values and vectors:
    covariance = np.dot(array, array.T)
    e_values, e_vectors = np.linalg.eigh(covariance)
    # e_values: sorted in ascending order.
    # e_vectors[:,i] are normalized.

    # Computes the PCs, normalizes them, and sorts them from the strongest to the weakest:
    pc_tmp = np.dot(e_vectors.T, array)
    pc_norm = pc_tmp / np.reshape(np.sqrt(np.abs(e_values)), (len(array), 1))  # Soummer2011
    # pc_norm = pc_tmp / np.reshape(np.sum(pc_tmp* np.conjugate(pc_tmp),axis=1), (len(array),1)) #KLIP demo
    # The version with sqrt(e_value) and sqrt(pc_tmp**2) are exactly equal (down to machine precision)
    pc_vec = pc_norm[::-1]

    return pc_vec, e_values[::-1]




def pca_subtraction(sci_cube, psf_cube, kl_value, opt_zone=None):
    
    
    if sci_cube.ndim != 3:
        raise TypeError('sci_cube should be an image cube')
    if psf_cube.ndim != 3:
        raise TypeError('psf_cube should be an image cube')
    if opt_zone is None:
        opt_zone = np.ones_like(psf_cube[0], dtype=bool)
    
    psf_cube_prepped = psf_cube[:, opt_zone] - np.mean(psf_cube[:, opt_zone], axis=1, keepdims=True)
    sci_cube_prepped = sci_cube[:, opt_zone] - np.mean(sci_cube[:, opt_zone], axis=1, keepdims=True)
    
    # Computes the PC modes
    pc_vec, e_values = compute_PC(psf_cube_prepped)
    pc_vec_trunc = pc_vec[:kl_value]

    res_cube = np.zeros_like(sci_cube)
    
    # Subtract the PSF
    for i, sci_im in enumerate(sci_cube_prepped):
        coefs = np.dot(sci_im, pc_vec_trunc.T)
        modelPSF = np.dot(coefs, pc_vec_trunc)
        res_cube[i, opt_zone] = sci_im - modelPSF
     
    return res_cube


def compute_pca_throughput(seps, contrasts, psf_model, psf_cube, kl_value, opt_zone=None):
    
    if opt_zone is None:
        opt_zone = np.ones_like(psf_cube[0], dtype=bool)
    
    imSize = psf_model.shape
    imCent = np.array(imSize) // 2
    
    psf_cube_prepped = psf_cube[:, opt_zone] - np.mean(psf_cube[:, opt_zone], axis=1, keepdims=True)
    pc_vec, e_values = compute_PC(psf_cube_prepped)
    pc_vec_trunc = pc_vec[:kl_value]
    
    position_angles = np.linspace(0, 2*np.pi, num=10, endpoint=False)
    throughput = np.zeros_like(seps)
    for i, sep in enumerate(seps):
        throughput_angles = np.zeros_like(position_angles)
        for j, angle in enumerate(position_angles):
            sep = seps[i]
            angle = position_angles[j]
            # Shift fake planet to 6 position angles at each separation:
            pose_from_cent = sep * np.array((-np.sin(angle), np.cos(angle)))
            model_shifted = contrasts[i] * imutils.shift(psf_model, pose_from_cent)
            
            # Forward model, affected by PCA:
            model_prepped = model_shifted[opt_zone] - np.mean(model_shifted[opt_zone])
            coefs = np.dot(model_prepped, pc_vec_trunc.T)
            modelPSF = np.dot(coefs, pc_vec_trunc)
            forward_model = np.zeros_like(model_shifted)
            forward_model[opt_zone] = model_prepped - modelPSF
            
            # Measure the planet flux before and after PCA
            ap_rad = 4
            aperture = create_annulus(0, ap_rad, psf_model.shape, cent=imCent+pose_from_cent)
            flux_in = np.sum(model_shifted[aperture * opt_zone])
            flux_out = np.sum(forward_model[aperture * opt_zone])
            throughput_angles[j] = flux_out / flux_in
            
            # fig0, ax0 = plt.subplots(1, 2, figsize=(16, 8))
            # ax0[0].imshow(model_shifted*aperture, vmax=vmax, vmin=-0.5*vmax)
            # ax0[1].imshow(forward_model*aperture, vmax=vmax, vmin=-0.5*vmax)
        throughput[i] = np.mean(throughput_angles)
        
    return throughput


def derotate_and_combine(cube, angles, weights=None):
    """ Derotates a cube of images then mean-combine them.

    Parameters
    ----------
    cube : the input cube, 3D array.
    angles : the list of parallactic angles corresponding to the cube.
    weights : Optional list of weights for each image of the combined cube.

    Returns
    -------
    image_out : the mean-combined image

    """
    if cube.ndim != 3:
        raise TypeError('Input cube is not a cube or 3d array')
    if angles.ndim != 1:
        raise TypeError('Input angles must be a 1D array')
    if len(cube) != len(angles):
        raise TypeError('Input cube and input angle list must have the same length')
    if weights is None:
        weights = np.ones(len(cube))

    shape = cube.shape
    cube_out = np.zeros(shape)
    for im in range(shape[0]):
        # cube_out[im] = frame_rotate(cube[im], -angles[im]) * weights[im]
        cube_out[im] = frame_rotate_interp(cube[im], -angles[im]) * weights[im]

    image_out = np.nanmean(cube_out, axis=0)
    return image_out


def uncalibrated_contrast_curve(image, fwhm, rmin, rmax, mask=None):
    """ Computes the uncalibrated contrast curbe from a normalized-intensity image

    Parameters
    ----------
    image : 2D image
    fwhm : fwhm in pixel
    rmin : inner radial boudary of the contrast curve in pixel
    rmax: outer radial bounday of the contrast curve in pixel
    mask: optional boolean 2D mask to hide a planet or other specific features

    Returns
    -------
    seps : separations sampled in the contrast curve
    contrasts : 5-sigma contrasts measured at the sampled separations

    """
    shape = image.shape
    
    if mask is None:
        mask = np.ones_like(image, dtype=bool)
    
    step = fwhm / 2.
    num = int(np.ceil((rmax - rmin - step) / step))
    seps = np.linspace(rmin+step/2., rmax-step/2., num)
    contrasts = np.zeros_like(seps)
    
    for i, sep in enumerate(seps):
        annulus = create_annulus(sep-step/2., sep+step/2., shape)
        res_mean = np.nanmean(image[annulus * mask])
        res_std = np.nanstd(image[annulus * mask])
        
        # Small sample statistics correction for 5sigma statistics:
        nsamples = int(np.nansum(annulus * mask) / (np.pi * fwhm**2 / 4))
        if (nsamples != 0):
            tau = t.ppf(0.99999971334, nsamples-1, scale=res_std)
        else:
            tau = np.nan
        
        detection_limit = tau * np.sqrt(1 + 1./nsamples) + res_mean
        contrasts[i] = detection_limit
    
    return seps, contrasts

#%% Get the raw images

path = # --- IGNORE ---
filename_ref = 'reference_star_acquisition_50.fits'
filename_sci_A = 'planetA_acquisition_50.fits'
filename_sci_B = 'planetB_acquisition_50.fits'
filename_psf = 'non_coronagraphic_psf_shaped.fits'

data_ref = fits.getdata(os.path.join(path, filename_ref))
data_sci_A = fits.getdata(os.path.join(path, filename_sci_A))
data_sci_B = fits.getdata(os.path.join(path, filename_sci_B))
data_sci = np.concatenate((data_sci_A, data_sci_B))
unocculted_psf = fits.getdata(os.path.join(path, filename_psf))
unocculted_psf /= np.max(unocculted_psf)

nRef = len(data_ref)
nSci_A = len(data_sci_A)
nSci_B = len(data_sci_B)
nSci = len(data_sci)
imSize = data_sci_A[0].shape
star_cent = np.array(imSize) // 2

angle_A = float(fits.getheader(os.path.join(path, filename_sci_A))['angle'])
angle_B = float(fits.getheader(os.path.join(path, filename_sci_B))['angle'])
angles_sci = np.concatenate((np.full((nSci_A), angle_A), np.full((nSci_B), angle_B)))

# planet_sep = float(fits.getheader(os.path.join(path, filename_sci_A))['RADIUS'])
planet_sep = 13.4 # pix, TBC with Lisa
planet_PA = -90   # deg, planet position angle from North
planet_pos_A = star_cent + planet_sep * np.array((-np.sin(np.deg2rad(angle_A+planet_PA)), np.cos(np.deg2rad(angle_A+planet_PA))))
planet_pos_B = star_cent + planet_sep * np.array((-np.sin(np.deg2rad(angle_B+planet_PA)), np.cos(np.deg2rad(angle_B+planet_PA))))
planet_pos = star_cent + planet_sep * np.array((-np.sin(np.deg2rad(planet_PA)), np.cos(np.deg2rad(planet_PA))))


print('Ref file shape: ', data_ref.shape)
print('Sci_A file shape: ', data_sci_A.shape)
print('Sci_B file shape: ', data_sci_B.shape)
print('Rolls available: ', set(angles_sci))


# Combined image and contrast curve without PSF subtraction
data_sci_comb = derotate_and_combine(data_sci, -angles_sci)

vmax = 10e4
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 8))
ax2.imshow(data_sci_comb, vmin=-vmax/2, vmax=vmax)

# Parameters for the contract curves
fwhm = 2.5  # pix TBC with Lisa
rmin = 4    # pix
rmax = 20   # pix

planet_mask_rad = 2.1*fwhm
planet_mask_A = ~create_annulus(0, planet_mask_rad, imSize, cent=planet_pos_A)
planet_mask_B = ~create_annulus(0, planet_mask_rad, imSize, cent=planet_pos_B)
planet_mask = ~create_annulus(0, planet_mask_rad, imSize, cent=planet_pos)

seps0, contrast0 = uncalibrated_contrast_curve(data_sci_comb, fwhm, rmin, rmax, mask=planet_mask)


#%% Classical RDI
rIn = 4
rOut = 20
opt_zone = create_annulus(rIn, rOut, imSize)
crdi_sci = classical_subtraction(data_sci, data_ref, opt_zone=opt_zone)

# crdi_sci_A = classical_subtraction(data_sci_A, data_ref, opt_zone=opt_zone*planet_mask_A)
# crdi_sci_B = classical_subtraction(data_sci_B, data_ref, opt_zone=opt_zone*planet_mask_B)
# crdi_sci = np.concatenate((crdi_sci_A, crdi_sci_B))

vmax = 1e4
fig3_titles = np.array([['C-RDI SCI_A, first frame', 'C-RDI SCI_A, last frame', 'C-RDI SCI_A, mean'],
                       ['C-RDI SCI_B, first frame', 'C-RDI SCI_B, last frame', 'C-RDI SCI_B, mean']])
fig3, ax3 = plt.subplots(2, 3, figsize=(12, 8))
ax3[0, 0].imshow(crdi_sci[0], vmin=-vmax/2, vmax=vmax)
ax3[0, 1].imshow(crdi_sci[nSci_A-1], vmin=-vmax/2, vmax=vmax)
ax3[0, 2].imshow(np.mean(crdi_sci[:nSci_A], axis=0), vmin=-vmax/2, vmax=vmax)
ax3[1, 0].imshow(crdi_sci[nSci_A+1], vmin=-vmax/2, vmax=vmax)
ax3[1, 1].imshow(crdi_sci[-1], vmin=-vmax/2, vmax=vmax)
ax3[1, 2].imshow(np.mean(crdi_sci[nSci_A:], axis=0), vmin=-vmax/2, vmax=vmax)
for i in range(2):
    for j in range(3):
        ax3[i, j].set_title(fig3_titles[i, j])


crdi_sci_comb = derotate_and_combine(crdi_sci, -angles_sci)

vmax = 10e4
fig4, ax4 = plt.subplots(1, 1, figsize=(8, 8))
ax4.imshow(crdi_sci_comb, vmin=-vmax/2, vmax=vmax)

seps_crdi, contrast_crdi = uncalibrated_contrast_curve(crdi_sci_comb, fwhm, rmin, rmax, mask=planet_mask)


#%% PCA RDI

kl = len(data_ref) // 3
# kl = len(data_ref) // 2
# kl = 2* len(data_ref) // 3
# kl = len(data_ref)

pca_sci = pca_subtraction(data_sci, data_ref, kl, opt_zone=opt_zone)

vmax = 1e4
fig6_titles = np.array([['PCA SCI_A, first frame', 'PCA SCI_A, last frame', 'PCA SCI_A, mean'],
                       ['PCA SCI_B, first frame', 'PCA SCI_B, last frame', 'PCA SCI_B, mean']])
fig6, ax6 = plt.subplots(2, 3, figsize=(12, 8))
ax6[0, 0].imshow(pca_sci[0], vmin=-vmax/2, vmax=vmax)
ax6[0, 1].imshow(pca_sci[nSci_A-1], vmin=-vmax/2, vmax=vmax)
ax6[0, 2].imshow(np.mean(pca_sci[:nSci_A], axis=0), vmin=-vmax/2, vmax=vmax)
ax6[1, 0].imshow(pca_sci[nSci_A+1], vmin=-vmax/2, vmax=vmax)
ax6[1, 1].imshow(pca_sci[-1], vmin=-vmax/2, vmax=vmax)
ax6[1, 2].imshow(np.mean(pca_sci[nSci_A:], axis=0), vmin=-vmax/2, vmax=vmax)
for i in range(2):
    for j in range(3):
        ax6[i, j].set_title(fig6_titles[i, j])

pca_sci_comb = derotate_and_combine(pca_sci, -angles_sci)

vmax = 10e4
fig7, ax7 = plt.subplots(1, 1, figsize=(8, 8))
ax7.imshow(pca_sci_comb, vmin=-vmax/2, vmax=vmax)

seps_pca, contrast_pca = uncalibrated_contrast_curve(pca_sci_comb, fwhm, rmin, rmax, mask=planet_mask)
pca_throughput = compute_pca_throughput(seps_pca, contrast_pca, unocculted_psf, data_ref, kl, opt_zone=opt_zone)

fig5, ax5 = plt.subplots(1, 1, figsize=(6, 4))
ax5.plot(seps0, contrast0, label='Raw image')
ax5.plot(seps_crdi, contrast_crdi, label='Classical RDI')
ax5.plot(seps_pca, contrast_pca/pca_throughput, label='PCA RDI')
ax5.set_xlabel('Spearation (pix)')
ax5.set_ylabel('5\sigma detection limit (NI)')
ax5.set_yscale('log')
ax5.legend()
