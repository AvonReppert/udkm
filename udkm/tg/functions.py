# -*- coding: utf-8 -*-

import numpy as np
import os as os
import pandas as pd
import udkm.tools.functions as tools
import shutil


import udkm.tools.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_scan_parameter(ref_file_name, line):
    '''creates a dictionary from the information on a measurement noted in a reference file

    Reads in 'ref_file_name.txt' in the folder 'reference_data/'. Information that are not stated in that file are
    set by default. The reference file needs to contain the columns 'date', 'time' and 'power' for each
    measurement.


    Parameters
    ----------
    ref_file_name : str
        File name of the reference file storing information about measurement.
    line : int
        The number of the measurement equivalent to line in the reference file.

    Returns
    -------
    params : dictionary
        - 'date': (str) date of measurement using the format yyyymmdd,
        - 'time': (str) time of measurement using the format hhmmss
        - 'power': (float) laser power without chopper [mW]
        - 'time_zero': (float, optional) delay of pump-probe overlap in [ps] - default 0
        - 'temperature': (float, optional) start temperature of measurement [K] - default 300
        - 'pump_angle': (float, optional) angle between surface normal and pump beam [deg] - default 0
        - 'field_angle': (float, optional) angle of the external field in respect to surface normal [deg] - default 0
        - 'double_pulse': (int, optional) first (0), second (1) or both pulses (2) in double pulse experiment -default 0
        - 'frontside': (int, optional) whether (1) or not (0) the excitation happens at from the surface - default 1
        - 'field': (float, optional) external magnetic field strength [mT] - default -1
        - 'fwhm_x': (float, optional) horizontal pump width [microns] - default 1000
        - 'fwhm_y': (float, optional) vertical pump width [microns] - default 1000
        - 'wavelength': (float, optional) wavelength of pumppulse [nm] - default 800
        - 'pulse_length': (float, optional) duration of pump pulse [fs] - default 120
        - 'fluence': (float, optional) calculated from 'power', 'fwhm_x' and 'fwhm_y' in top-hat approximation [mJ/cm^2]
        - 'voltage': (float, optional) set voltage that equals current in the electromagnet [A] default - 5
        - 'rotation': (float, optional) position of rotatable stage (power or field angle) - default -1

    Example
    -------
    The following code creates the parameter dictionary corresponding to line 1 in the reference file

    >>> params = get_scan_parameter('reference.txt', 1)
    {'date': '20220813',
     'time': '181818',
     'id': 20220813_181818
     'rotation': -1.0,
     'voltage': 5.0,
     'power': 50,
     'fwhm_x': 1100,
     'fwhm_y': 1100,
     'temperature': 300,
     'field': 525,
     'pump_angle': 0,
     'field_angle': 0,
     'time_zero': 0,
     'double_pulse': 0,
     'frontside': 0,
     'wavelength': 800,
     'pulse_length': 120,
     'fluence': 5.32
     }

    '''
    params = {'line': line}
    param_file = pd.read_csv('reference_data/' + ref_file_name, delimiter="\t", header=0, comment="#")
    header = list(param_file.columns.values)

    for i, entry in enumerate(header):
        if entry == 'date' or entry == 'time':
            params[entry] = tools.timestring(int(param_file[entry][line]))
        else:
            params[entry] = param_file[entry][line]

    # set default parameters
    params["id"] = params["date"] + "_" + params["time"]
    params["rep_rate"] = 1000

    if not('rotation') in header:
        params['rotation'] = -1.0

    if not('voltage') in header:
        params['voltage'] = 5

    if not('fwhm_x') in header:
        params['fwhm_x'] = 1000

    if not('fwhm_y') in header:
        params['fwhm_y'] = 1000

    if not('temperature') in header:
        params['temperature'] = 300

    if not('field') in header:
        params['field'] = -1

    if not('pump_angle') in header:
        params['pump_angle'] = 0

    if not('field_angle') in header:
        params['field_angle'] = 0

    if not('time_zero') in header:
        params['time_zero'] = 0

    if not('double_pulse') in header:
        params['double_pulse'] = 0

    if not('frontside') in header:
        params['frontside'] = 1

    if not('wavelength') in header:
        params['wavelength'] = 800

    if not('pulse_length') in header:
        params['pulse_length'] = 120

    if not('fluence') in header:
        params['fluence'] = tools.calc_fluence(params["power"], params["fwhm_x"], params["fwhm_y"],
                                               params["pump_angle"], params["rep_rate"])
    return params
