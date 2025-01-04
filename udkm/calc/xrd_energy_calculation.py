# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:59:25 2024

@author: aleks
"""

import math


def calculate_xray_energy(lattice_constant, angle_deg, hkl):
    # Constants
    h = 6.626e-34  # Planck's constant (J·s)
    c = 3e8        # Speed of light (m/s)
    eV_to_J = 1.602e-19  # 1 electron volt in Joules

    # Convert angle from degrees to radians
    angle_rad = math.radians(angle_deg)

    # Calculate d-spacing for given (hkl)
    d_spacing = lattice_constant / math.sqrt(hkl[0]**2 + hkl[1]**2 + hkl[2]**2)

    # Use Bragg's law to calculate wavelength
    n = 1  # First-order diffraction
    wavelength = 2 * d_spacing * math.sin(angle_rad) / n  # in meters

    # Calculate energy in Joules and convert to electron volts
    energy_J = h * c / wavelength
    energy_eV = energy_J / eV_to_J

    # Convert to keV
    energy_keV = energy_eV / 1000
    return energy_keV


# Input lattice constant for Gold (111) in meters
lattice_constant_gold = 4.078e-10  # meters (convert Å to m if needed)
diffraction_angle = 41.206  # degrees

# Define the reflexes to calculate (111, 222, 333, 444, 555)
reflexes = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]

# Calculate and print energy for each reflex
print("Reflex\tEnergy (keV)")
for reflex in reflexes:
    energy = calculate_xray_energy(lattice_constant_gold, diffraction_angle, reflex)
    print(f"{reflex}\t{energy:.4f} keV")
