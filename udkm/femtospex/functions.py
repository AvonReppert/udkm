# -*- coding: utf-8 -*-
"""
This module contains basic functions used to evaluate data from the femtospex beamline

-- Further information --

"""
import numpy as np
import udkm.tools.constants as u


def get_q(data):
    """
    Calculate the scattering vector q for a given dataset.

    Parameters
    ----------
    data : dict or DataFrame
        A data container that should have at least the following keys:
        - "Energy": The energy of the incident beam in electronvolts (eV).
        - "TwoTheta": The scattering angle (2θ) in degrees.

    Returns
    -------
    q : ndarray
        The scattering vector q in inverse angstroms (1/Å).

    Notes
    -----
    This function calculates the scattering vector q based on the provided
    energy and scattering angle (2θ) using the following formula:

    q = (4 * π / λ) * sin(2θ/2)

    where:
    - λ is the wavelength of the incident beam calculated from energy and
      the speed of light using the Planck constant.
    - Energy is provided in electronvolts (eV).

    Constants
    ---------
    - u.EV : float
        The elementary charge in electronvolts (eV).
    - u.H_PLANCK : float
        The Planck constant in joule-seconds (J·s).
    - u.C_0 : float
        The speed of light in meters per second (m/s).

    Examples
    --------
    >>> data = {"Energy": 10, "TwoTheta": 60}
    >>> get_q(data)
    1.2010773166797805
    """
    E = np.mean(data["Energy"]) * u.EV
    wl = u.H_PLANCK * u.C_0 / E
    q = (4 * np.pi / wl) * np.sin(data["TwoTheta"] / 2 * np.pi / 180) * 1e-10
    return q


def get_turn_angle_dy(q_pos):
    """
    Calculate the interlayer turn angle (in degrees) for the magnetic moment in the Dysprosium spin spiral(Dy) 
    based on the provided q-position of the resonant scattering peak given in (1/Å).

    Parameters
    ----------
    q_pos : float or ndarray
        The q-position (scattering vector) in inverse angstroms (1/Å).

    Returns
    -------
    phi : float or ndarray
        The interlayer turn angle

    Notes
    -----
    This function calculates the interlayer turn angle of the spin spiral for Dysprosium (Dy) using the provided
    q-position of the scattering peak using the following formula:

    angle = (180 * c_Dy * q_pos) / (2 * π)

    where:
    - `c_Dy` is the lattice constant for Dysprosium in angstroms (Å).
    - The scattering angle is given in degrees.

    Constants
    ---------
    - c_Dy : float
        The lattice constant for Dysprosium (Dy) in angstroms (Å).

    Examples
    --------
    >>> q_pos = 0.18
    >>> get_turn_angle_dy(q_pos)
    29.1534677149646
    """
    c_Dy = 5.6536  # Lattice constant for Dysprosium (Dy) in angstroms (Å)
    phi = (180 * c_Dy * q_pos) / (2 * np.pi)
    return phi
