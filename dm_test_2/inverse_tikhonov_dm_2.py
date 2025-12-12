# Inverse Tikhonov with Dual DMs
# This script implements the inverse Tikhonov regularization method for two deformable mirrors (DMs)
# to achieve high-contrast imaging in a coronagraphic system. The code calculates the optimal
# actuator commands for both DMs to minimize the intensity in a defined dark zone of the image plane.
from ast import Lambda
import numpy as np
import matplotlib.pyplot as plt
from scipy.differentiate import jacobian
from scipy.linalg import svd
# Regularization parameters for DM1 and DM2

def inverse_tikhonov_dual_dm(J, influence_functions, rcond_dm1, rcond_dm2):
    """
    Compute the inverse Tikhonov regularization for dual DMs.

    Parameters:
    jacobian (ndarray): The Jacobian matrix mapping DM actuator commands to image plane intensities.
    rcond_dm1 (float): Regularization parameter for DM1.
    rcond_dm2 (float): Regularization parameter for DM2.

    Returns:
    ndarray: The pseudo-inverse of the Jacobian matrix with Tikhonov regularization.
    """

    # Create the block-diagonal regularization matrix Lambda^2
    num_modes_dm1 = len(influence_functions)
    num_modes_total = 2*num_modes_dm1
    num_error_states = J.shape[0]

    # Calculate the Hermitian Transpose of J
    J_H = J.conj().T
    J_H_J = J_H @ J

    # 1. Initialize the block-diagonal regularization matrix Lambda^2
    lambda_matrix = np.zeros((num_modes_total, num_modes_total), dtype=J.dtype) # Use complex just in case

    # 2. Fill DM1 block (top-left) with rcond_dm1 squared
    np.fill_diagonal(lambda_matrix[:num_modes_dm1, :num_modes_dm1], rcond_dm1**2)

    # 3. Fill DM2 block (bottom-right) with rcond_dm2 squared
    np.fill_diagonal(lambda_matrix[num_modes_dm1:, num_modes_dm1:], rcond_dm2**2)

    # 4. Compute the pseudo-inverse with Tikhonov regularization
    A = J_H_J + lambda_matrix
    B = J_H
    efc_matrix = np.linalg.solve(A, B)
    # efc_matrix = np.linalg.inv(J_H_J + lambda_matrix) @ J_H

    M_aug = np.concatenate((J, lambda_matrix), axis=0)

    M_aug_pinv = np.linalg.pinv(M_aug)

    efc_matrix = M_aug_pinv[:, :num_error_states]

    return efc_matrix