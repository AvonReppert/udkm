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
        - "Theta": The scattering angle in degrees.

    Returns
    -------
    q : ndarray
        The scattering vector q in inverse angstroms (1/Å).

    Notes
    -----
    This function calculates the scattering vector q based on the provided
    energy and scattering angle using the following formula:

    q = (4 * π / λ) * sin(θ)

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
    >>> data = {"Energy": 10, "Theta": 30}
    >>> get_q(data)
    2.402154633359561
    """
    E = np.mean(data["Energy"]) * u.EV
    wl = u.H_PLANCK * u.C_0 / E
    q = (4 * np.pi / wl) * np.sin(data["Theta"] * np.pi / 180) * 1e-10
    return q
