import numpy as np
def get_jacobian_matrix(get_image, dark_zone, num_modes, epsilon):
    responses = []
    amps = np.linspace(-epsilon, epsilon, 2)

    for i, mode in enumerate(np.eye(num_modes)):
        response = 0

        for amp in amps:
            response += amp * get_image(mode * amp, include_aberration=False).electric_field

        response /= np.var(amps)
        response = response[dark_zone]

        responses.append(np.concatenate((response.real, response.imag)))

    jacobian = np.array(responses).T
    return jacobian

"""
import numpy as np

def get_jacobian_matrix(get_image, dark_zone, num_modes, epsilon):    
    # The number of actuators per DM, assuming equal DMs for Jacobian shape.
    num_modes_per_dm = num_modes // 2 
    
    # The total number of actuators being poked (DM1 + DM2).
    total_modes_to_poke = num_modes 

    # We will build the responses list for all DM modes
    responses = []

    for i in range(total_modes_to_poke):
        # 1. Poke Positive
        # We poke the i-th mode with +epsilon
        poke_plus = np.zeros(total_modes_to_poke)
        poke_plus[i] = epsilon
        
        # 2. Poke Negative
        # We poke the i-th mode with -epsilon
        poke_minus = np.zeros(total_modes_to_poke)
        poke_minus[i] = -epsilon

        # Get the electric field response for the positive and negative pokes.
        # This is where the initial aberration MUST be included in the simulation.
        E_plus = get_image(poke_plus).electric_field
        E_minus = get_image(poke_minus).electric_field
        
        # 3. Calculate the Jacobian element J_i using the Centered Difference formula
        # Response = (E_plus - E_minus) / (2 * epsilon)
        response_complex = (E_plus - E_minus) / (2 * epsilon)

        # 4. Extract Real and Imaginary components (ensures Real/Imag Block order)
        response_dark_zone = response_complex[dark_zone]
        
        # J rows are [Real_1, Real_2, ..., Imag_1, Imag_2, ...]
        responses.append(np.concatenate((response_dark_zone.real, response_dark_zone.imag)))

    # The Jacobian is constructed with E-field components in rows and Actuators in columns.
    jacobian = np.array(responses).T
    return jacobian
"""