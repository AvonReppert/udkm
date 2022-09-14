# -*- coding: utf-8 -*-
import numpy as np
import udkm.tools.functions as tools
import udkm.tools.colors as colors
import h5py
import lmfit as lm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def initialize_series():
    '''initializes an empty dictionary that can be filled with data from a series of similar scans

    Returns
    -------
    series : dictionary
        - 'qx': (list of numpy arrays) list of qx grids
        - 'qy': (list of numpy array) list of qy grids
        - 'qz': (list of numpy array) list of qz grids
        - 'intensity_qx': (list of numpy arrays) list of qx intensities
        - 'intensity_qy': (list of numpy array) list of qy intensities
        - 'intensity_qz': (list of numpy array) list of qz intensities
        - 'scan': (list of integers) scan numbers
        - 'temperature': (list of numpy array) temperatures for each scan
        - 'delay': (list of numpy array) delays for each scan
        - 'com_qz': (list of numpy array) list of evaluated com
        - 'std_qz': (list of numpy array) list of evaluated standard deviation
        - 'integral_qz' (list of numpy array) list of evaluated integral values
        - 'fit_qz': (numpy array) list of evaluated fit positions
        - 'width_qz': (numpy array) list of evaluated fit widths
        - 'area_qz' (list of numpy array) list of evaluated areas
        - 'slope_qz': (numpy array) list of evaluated slopes
        - 'offset_qz': (numpy array) list of evaluated offset
        - 'c_min_lin': (float) colormap minimum for linear scaling - default 0
        - 'c_max_lin':  (float) colormap maximum for linear scaling - default 1000
        - 'c_min_log': (float) colormap minimum for logarithmic scaling - default 1
        - 'c_max_log':  (float) colormap maximum for logarithmic scaling - default 1000

    Example
    -------

    >>> series = initialize_series()
    '''
    key = ["qx", "qy", "qz", "rsm",
           "intensity_qx", "intensity_qy", "intensity_qz",
           "scan", "temperature", "delay",
           "com_qz", "std_qz", "integral_qz",
           "fit_qz", "position_qz", "width_qz", "area_qz", "slope_qz", "offset_qz",
           "com_qy", "std_qy", "integral_qy",
           "fit_qy", "position_qy", "width_qy", "area_qy", "slope_qy", "offset_qy",
           "com_qx", "std_qx", "integral_qx",
           "fit_qx", "position_qx", "width_qx", "area_qx", "slope_qx", "offset_qx"]
    series = {k: [] for k in key}
    series["c_min_log"] = 1
    series["c_max_log"] = 1000
    series["c_min_lin"] = 1
    series["c_min_lin"] = 1000
    return series
