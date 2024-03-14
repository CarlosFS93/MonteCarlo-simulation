import numpy as np
from math import sqrt
from local import *

# This function ask the user for the parameter requiered to do the MonteCarlo simulation
def get_user_input():
    
    num_electrons = int(input("Introduce number of electrons: "))
    num_nuclei = int(input("Introduce number of nucleis: "))
    a = float(input("Introduce parameter a: "))
    atomic_weight = int(input("Introduce atomic weight (ex. 1 for H, 2 for He): "))
    total_charge = int(input("Introduce the total charge: "))

    # This function calculate the charge of the nucleis splitting the total charge between the nucleus
    nucleus_charges = [atomic_weight for _ in range(num_nuclei)]

    charge_difference = total_charge - sum(nucleus_charges)
    if charge_difference != 0 and num_nuclei > 0:
        nucleus_charges[0] += charge_difference

    # This part transform angstron into atomic units
    angstrom_to_atomic_units = 1.0 / 0.529177
    nucleus_positions = []
    for i in range(num_nuclei):
        print(f"Introduce coordinates of nuclei {i+1} in Angstrom (format: x y z): ")
        coordinates = input().split()
        coordinates_atomic_units = [float(coord) * angstrom_to_atomic_units for coord in coordinates]
        nucleus_positions.append(coordinates_atomic_units)

    nmax = int(input("How many MonteCarlo steps do you want?"))
    #Ask the user to select the type of Monte Carlo Simulation
    print("What type of Monte Carlo simulation do you want to run?")
    print("1: Variational Monte Carlo (VMC)")
    print("2: Pure Diffusion Monte Carlo (PDMC)")
    choice = input()

    return choice, num_electrons, num_nuclei, a, nucleus_positions, nucleus_charges, nmax

if __name__ == "__main__":
    choice, num_electrons, num_nuclei, a, nucleus_positions, nucleus_charges, nmax = get_user_input()
    print(f"Number of electrons: {num_electrons}, Number of nucleis: {num_nuclei}, Parameter a is: {a}")
    print("Coordinates of nuclei (atomic units):")
    for idx, pos in enumerate(nucleus_positions):
        print(f"Nuclei {idx+1}: {pos}")

# Randomization of the coordinates of the electrons. 
# This is just to calculate the local energy but in the MonteCarlo new 
# Random coordinates for the electrons will be generated so all this part could be deleted
def generate_electron_positions(num_electrons):
    electron_positions = np.random.uniform(-5, 5, size=(num_electrons, 3))

    return electron_positions

electron_positions = generate_electron_positions(num_electrons)
print("Coordinates of the electrons:")
print(electron_positions)

# We calculate here the local energy with the inputs given by the user
v_total = total_potential(electron_positions, nucleus_positions, nucleus_charges, num_electrons, num_nuclei) 
print(f"Total potential of the system is: {v_total}")
total_psi = psi(electron_positions, nucleus_positions, a)
print("wave funtion psi:",total_psi)
kinetic_energy = total_kinetic_energy(a, electron_positions)
print("kinetic energy of the system:", kinetic_energy)
e_loc = local_energy(kinetic_energy, v_total)
print("Local energy of the system:", e_loc)

#Function for average error
def ave_error(arr):
    # Determine the number of observations in the array.
    M = len(arr)
    # Ensure that there is at least one observation.
    assert(M > 0)

    # If there's only one observation, the average is the observation itself and error is 0.
    if M == 1:
        average = arr[0]
        error = 0.
    else:
        average = sum(arr) / M
        variance = sum((x - average) ** 2 for x in arr) / (M - 1)
        # The standard error of the mean (SEM) is the square root of the
        # variance divided by the number of observations.
        error = sqrt(variance / M)

    return average, error

#Function for the drift vector
def drift(electron_positions, nucleus_positions, a):
    # Initialize an array to hold the drift vectors for each electron,
    # with the same shape as the electron_positions array.
    drift_vectors = np.zeros_like(electron_positions)
    # Loop over each electron position vector.
    for i, r in enumerate(electron_positions):
        # Initialize a gradient sum for the current electron.
        gradient_sum = np.zeros_like(r)
        # Sum the gradients due to all nuclei for this electron.
        for nucleus in nucleus_positions:
            # Calculate the vector from the nucleus to the electron.
            r_minus_R = r - nucleus            
            distance = np.linalg.norm(r_minus_R)            
            # Avoid division by zero in case the electron is very close to the nucleus.
            if distance != 0:
                # Compute the gradient of the exponential part of the wave function
                # with respect to the electron position.
                gradient = -a * r_minus_R / distance               
                # Add the gradient contribution from this nucleus to the total gradient_sum.
                gradient_sum += gradient        
        drift_vectors[i] = gradient_sum
    
    return drift_vectors

#MonteCarlo with the drift diffusion scheme
def VMC(a, num_electrons, nucleus_positions, nucleus_charges, nmax, num_nuclei):    
    dt = 1.0
    energy = 0.
    N_accep = 0

    sq_dt = np.sqrt(dt)
    # Initializes random positions for all electrons.
    r_old = np.random.normal(loc=0., scale=1.0, size=(num_electrons, 3))
    d_old = drift(r_old, nucleus_positions, a)
    psi_old = psi(r_old, nucleus_positions, a)

    for istep in range(nmax):
        energy += local_energy(total_kinetic_energy(a, r_old), total_potential(r_old, nucleus_positions, nucleus_charges, num_electrons, num_nuclei))
        
        # Proposes new movements for each electron
        r_new = r_old + dt * d_old + sq_dt * np.random.normal(loc=0., scale=1.0, size=(num_electrons, 3))
        d_new = drift(r_new, nucleus_positions, a)
        psi_new = psi(r_new, nucleus_positions, a)

        # Metropolis criteria for full configuration
        q = (psi_new / psi_old) ** 2
        if np.random.uniform(0, 1) <= q:
            N_accep += 1
            r_old = r_new
            d_old = d_new
            psi_old = psi_new

    acceptance_rate = N_accep / nmax
    energy_average = energy / nmax

    return energy_average, acceptance_rate

def PDMC(a, num_electrons, nucleus_positions, nucleus_charges, nmax, num_nuclei):
    dt = 0.05
    sq_dt = np.sqrt(dt)
    energy_accum = 0.
    normalization = 0.
    acceptance = 0.
    tau_current = 0.
    tau = 100
    Eref = -0.5

    # Initialize random positions for all electrons
    r_old = np.random.normal(loc=0., scale=sq_dt, size=(num_electrons, 3))
    psi_old = psi(r_old, nucleus_positions, a)
    weight = 1.0

    for istep in range(nmax):
        e_local = local_energy(total_kinetic_energy(a, r_old), total_potential(r_old, nucleus_positions, nucleus_charges, num_electrons, num_nuclei))
        weight *= np.exp(-dt * (e_local - Eref))
        energy_accum += weight * e_local
        normalization += weight
        tau_current += dt

        # Reset weights and imaginary time after tau
        if tau_current >= tau:
            weight = 1.0
            tau_current = 0.0

        r_new = r_old + dt * drift(r_old, nucleus_positions, a) + sq_dt * np.random.normal(loc=0., scale=1.0, size=(num_electrons, 3))
        psi_new = psi(r_new, nucleus_positions, a)

        # Metropolis acceptance ratio
        q = (psi_new / psi_old) ** 2
        if np.random.uniform(0, 1) <= q:
            acceptance += 1
            r_old = r_new
            psi_old = psi_new

    energy_average = energy_accum / normalization if normalization != 0 else float('inf')
    acceptance_rate = acceptance / nmax

    return energy_average, acceptance_rate

def run_simulation(choice, a, num_electrons, nucleus_positions, nucleus_charges, nmax, num_nuclei):
    if choice == '1':
        # Run VMC
        vmc_results = [VMC(a, num_electrons, nucleus_positions, nucleus_charges, nmax, num_nuclei) for _ in range(30)]
        energies = [result[0] for result in vmc_results]
        E, deltaE = ave_error(energies)
        print("Variational Monte Carlo simulation:")
        print(f"Energy = {E} +/- {deltaE}")
        
        # Acceptance rate
        acceptance_rates = [result[1] for result in vmc_results]
        A, deltaA = ave_error(acceptance_rates)
        print(f"Acceptance rate = {A} +/- {deltaA}")
        
        return E, deltaE, A, deltaA 
        
    elif choice == '2':
        # Run PDMC
        pdmc_results = [PDMC(a, num_electrons, nucleus_positions, nucleus_charges, nmax, num_nuclei) for _ in range(30)]
        energies = [result[0] for result in pdmc_results]
        E, deltaE = ave_error(energies)
        print("Pure diffusion Monte Carlo simulation:")
        print(f"Energy = {E} +/- {deltaE}")
        
        acceptance_rates = [result[1] for result in pdmc_results]
        A, deltaA = ave_error(acceptance_rates)
        print(f"Acceptance rate = {A} +/- {deltaA}")
        
        return E, deltaE, A, deltaA 
        
    else:
        print("Election not allowed. Please, introduce 1 or 2.")
        return None
    
results = run_simulation(choice, a, num_electrons, nucleus_positions, nucleus_charges, nmax, num_nuclei)

