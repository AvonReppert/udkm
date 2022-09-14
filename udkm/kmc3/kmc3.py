# -*- coding: utf-8 -*-
import numpy as np
import udkm.tools.functions as tools
import udkm.tools.colors as colors
import h5py
import lmfit as lm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def get_q_data(h5_file, scan_nr, tau_offset=300.4):
    '''
    Parameters
    ----------
    h5_file : str
        name of the h5 file to load
    scan_nr : integer
        number of the scan to extract from the file
    tau_offset : str
        offset value of tau_APD

    Returns
    -------
    scan : dictionary
        - 'qx': (numpy array) qx grid
        - 'intensity_qx': (numpy array) intensity projected on the qx grid
        - 'qy': (numpy array) qy grid
        - 'intensity_qy': (numpy array) intensity projected on the qy grid
        - 'qz': (numpy array) qz grid
        - 'intensity_qz': (numpy array) intensity projected on the qz grid
        - 'delay_measured': (numpy array) measured delay
        - 'delay_set': (numpy array) set delay
        - 'temperature' (numpy array) temperature
        - 'q_grid': (numpy array) user definded grid
        - 'intensity': (numpy array) projection on the user defined grid

    Example
    -------
    Loads scan number 1196 from the given .h5 file

    >>> scan = get_q_data('au_mgo_333_static_temp_scans1196_1294_2.h5', 1196)

    '''

    h5_file = h5_file
    h5_data = h5py.File(h5_file, 'r')
    spec_name_key = list(h5_data.keys())[0]  # get key (aka spec filename) of the HDF5 file

    file_name_qx = spec_name_key + '/scan_' + str(scan_nr) + '/ReducedData/IntQx'
    file_name_qy = spec_name_key + '/scan_' + str(scan_nr) + '/ReducedData/IntQy'
    file_name_qz = spec_name_key + '/scan_' + str(scan_nr) + '/ReducedData/IntQz'
    file_name_grid = spec_name_key + '/scan_' + str(scan_nr) + '/ReducedData/grid'
    file_name_intensity = spec_name_key + '/scan_' + str(scan_nr) + '/ReducedData/QMapData'
    file_name_qx_x = spec_name_key + '/scan_' + str(scan_nr) + '/ReducedData/qx'
    file_name_qx_y = spec_name_key + '/scan_' + str(scan_nr) + '/ReducedData/qy'
    file_name_qx_z = spec_name_key + '/scan_' + str(scan_nr) + '/ReducedData/qz'
    spec_data = spec_name_key + '/scan_' + str(scan_nr) + '/data'

    qx = np.array(h5_data[file_name_qx_x])
    qy = np.array(h5_data[file_name_qx_y])
    qz = np.array(h5_data[file_name_qx_z])

    intensity_qx = np.array(h5_data[file_name_qx])
    intensity_qy = np.array(h5_data[file_name_qy])
    intensity_qz = np.array(h5_data[file_name_qz])

    intensity = np.array(h5_data[file_name_intensity])
    grid = np.array(h5_data[file_name_grid])

    temperature = np.mean(h5_data[spec_data]['ls_t1'][0])
    delay_measured = -np.mean(h5_data[spec_data]['PH_average'][1:]) + \
        tau_offset  # measured delay corrected for tau_apd_0
    delay_set = h5_data[spec_data]['delay'][1]  # set delay

    scan = {'qx': qx, 'intensity_qx': intensity_qx, 'qy': qy, 'intensity_qy': intensity_qy,
            'qz': qz, 'intensity_qz': intensity_qz, 'delay_measured': delay_measured,
            'delay_set': delay_set, 'temperature': temperature, 'q_grid': grid, 'intensity': intensity}

    return scan
