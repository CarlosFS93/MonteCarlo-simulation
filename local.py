import numpy as np

#Potential electron-electron
def electron_electron_potential(electron_positions, num_electrons):
    v_ee = 0.0  # Initializes the total potential to 0
    
    # Go through all the single pairs of electrons.
    for i in range(num_electrons):
        for j in range(i + 1, num_electrons):
            r_ij = np.linalg.norm(electron_positions[i] - electron_positions[j])  # Calculate the distance between the electrons
            v_ee += 1 / r_ij  # Add the potential of this pair to the total
    
    return v_ee

#Potential nucleus-nucleus
def nucleus_nucleus_potential(nucleus_positions, nucleus_charges, num_nuclei):    
    v_nn = 0.0  # Initializes the total potential to 0
    nucleus_positions = np.array(nucleus_positions)
    # Loop through all unique pairs of nuclei
    for i in range(num_nuclei):
        for j in range(i + 1, num_nuclei):
            r_ij = np.linalg.norm(nucleus_positions[i] - nucleus_positions[j])  # Calculate the distance between the nuclei
            v_nn += nucleus_charges[i] * nucleus_charges[j] / r_ij  # Add the potential of this pair to the total
    
    return v_nn

#Potential electron-nucleus
def electron_nucleus_potential(electron_positions, nucleus_positions, nucleus_charges, num_electrons, num_nuclei):
    nucleus_positions = np.array(nucleus_positions)
    v_en = 0.0  # Initializes the total potential to 0
    
    # Go through all the electrons and nuclei to calculate the sum of their interactions
    for i in range(num_electrons):
        for j in range(num_nuclei):
            r_ij = np.linalg.norm(electron_positions[i] - nucleus_positions[j])  
            v_en -= nucleus_charges[j] / r_ij  # Add the potential of this interaction to the total
    
    return v_en

def total_potential(electron_positions, nucleus_positions, nucleus_charges, num_electrons, num_nuclei):
    # Calculate individual potentials
    v_ee = electron_electron_potential(electron_positions, num_electrons)
    v_nn = nucleus_nucleus_potential(nucleus_positions, nucleus_charges, num_nuclei)
    v_en = electron_nucleus_potential(electron_positions, nucleus_positions, nucleus_charges, num_electrons, num_nuclei)
    
    # Calculate the total potential by adding the three potentials
    v_total = v_ee + v_nn + v_en
    
    return v_total
   
# the wave function is calculated without taking into account  
# the interaction between the electrons to make it simpler
def psi(electron_positions, nucleus_positions, a):
    # Calculate the sum of the exponential contributions of each electron to each nucleus
    total_psi = 1.0
    for electron in electron_positions:
        electron_contribution = 0.0
        for nucleus in nucleus_positions:
            r = np.linalg.norm(electron - nucleus)
            electron_contribution += np.exp(-a * r)
        total_psi *= electron_contribution
    return total_psi

#Calculation of the kinetica energy iterating for every electron
def total_kinetic_energy(a, electron_positions):
    kinetic_energy = 0.0
    
    for r in electron_positions:
        distance = np.linalg.norm(r)
        if distance == 0:
            print('Warning: Electron found at nucleus position, energy is infinite.')
            return float('inf')
        kinetic_value = -1/2 * (a**2 - 2 * a / distance) # Kinetic energy equation
        kinetic_energy += kinetic_value
    
    return kinetic_energy

# Local energy is simply the sum of kinetic and potential energies
def local_energy(kinetic_energy, v_total):
    e_loc = kinetic_energy + v_total
    return e_loc

