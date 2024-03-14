# Quantum Monte Carlo Simulations

This repository contains a Python program that performs Quantum Monte Carlo simulations. It includes two distinct types of Monte Carlo methods: Variational Monte Carlo (VMC) and Pure Diffusion Monte Carlo (PDMC), both used to estimate the ground state energy of quantum systems.

## Description

The program is designed to simulate electron-nucleus and electron-electron interactions within a quantum system. By leveraging Monte Carlo methods, it can handle multiple electrons and nuclei, making it a versatile tool for studying various atomic and molecular systems.

### Variational Monte Carlo (VMC)

VMC utilizes a trial wave function with variational parameters to estimate the lower bounds of the ground state energy of a quantum system. It involves a stochastic process to sample the probability distribution and optimize the parameters to minimize the energy.

### Pure Diffusion Monte Carlo (PDMC)

PDMC is a simplified version of the standard Diffusion Monte Carlo approach. It focuses solely on the diffusion process and does not include the branching process typical of full DMC methods. PDMC allows for a straightforward implementation and provides an estimate of the ground state energy through a series of iterative steps and weights adjustments based on a reference energy.

## Installation

To run this program, ensure that you have Python installed on your system. No additional libraries are required beyond what is standard with Python's scientific stack (NumPy and Math).

## Usage

To start a simulation, run the `montecarlo.py` file in your Python environment:

Follow the on-screen prompts to input the necessary parameters for the simulation:

- Number of electrons
- Number of nuclei
- Variational parameter `a`
- Atomic weight
    - The program only allow molecules with all the same atoms, so the same weight will be use for or your atoms
- Total charge
- Coordinates of your nuclei 
    - The coordinates must be in Angstrom and with a space between each of xyz axis, for example: 1 1 1
- Number of Monte Carlo steps that you want to perform in the simulation

After entering the parameters, you'll be asked to choose the type of Monte Carlo simulation to run:

- Enter `1` for Variational Monte Carlo (VMC).
- Enter `2` for Pure Diffusion Monte Carlo (PDMC).

The program will output the estimated ground state energy and the acceptance rate of the Monte Carlo steps.


