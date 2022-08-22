# -*- coding: utf-8 -*-
"""
Created on Thu May 19 09:27:21 2022

@author: aleks
"""

import numpy as np
import udkm.tools.functions as tools
import h5py
import lmfit as lm


def get_q_data(h5File, scanNr, tauOffset=300.4):
    '''Read an h5 file after converting the measured Pilatus images into a q_x -- q_z grid.

    args
        h5File          name of the h5 file to load
        scanNr          number of the scan to extract from the file
        tauOffset       offset value of tau_APD
    returns
        dictionary with q_x, q_y, q_z, delay, temperature, grid of q vectors, and measured intensity     
    '''
    h5File = h5File
    h5Data = h5py.File(h5File, 'r')
    # get key (aka spec filename) of the HDF5 file
    specNameKey = list(h5Data.keys())[0]

    FileNameQx = specNameKey + '/scan_' + str(scanNr) + '/ReducedData/IntQx'
    FileNameQy = specNameKey + '/scan_' + str(scanNr) + '/ReducedData/IntQy'
    FileNameQz = specNameKey + '/scan_' + str(scanNr) + '/ReducedData/IntQz'
    FileNameGrid = specNameKey + '/scan_' + str(scanNr) + '/ReducedData/grid'
    FileNameIntensity = specNameKey + '/scan_' + str(scanNr) + '/ReducedData/QMapData'
    FileNameQx_X = specNameKey + '/scan_' + str(scanNr) + '/ReducedData/qx'
    FileNameQx_Y = specNameKey + '/scan_' + str(scanNr) + '/ReducedData/qy'
    FileNameQx_Z = specNameKey + '/scan_' + str(scanNr) + '/ReducedData/qz'
    SpecData = specNameKey + '/scan_' + str(scanNr) + '/data'

    qx_X = h5Data[FileNameQx_X]
    qy_X = h5Data[FileNameQx_Y]
    qz_X = h5Data[FileNameQx_Z]

    qx = h5Data[FileNameQx]
    qy = h5Data[FileNameQy]
    qz = h5Data[FileNameQz]

    Intensity = np.array(h5Data[FileNameIntensity])
    grid = np.array(h5Data[FileNameGrid])

    Temp = np.mean(h5Data[SpecData]['ls_t1'][0])
    tauOffset = tauOffset
    delayMeas = -np.mean(h5Data[SpecData]['PH_average'][1:]) + tauOffset  # measured delay value corrected for tau_apd_0
    delaySet = 2  # h5Data[SpecData]['delay'][1]  # set delay
    xQzVec = np.array(qz_X)
    yQzVec = np.array(qz)
    xQyVec = np.array(qy_X)
    yQyVec = np.array(qy)
    xQxVec = np.array(qx_X)
    yQxVec = np.array(qx)
    scanData = {'xQx': xQxVec, 'yQx': yQxVec, 'xQy': xQyVec, 'yQy': yQyVec, 'xQz': xQzVec, 'yQz': yQzVec,
                'delayMeas': delayMeas, 'delaySet': delaySet, 'Temperature': Temp, 'qGrid': grid, 'Intensity': Intensity}

    return scanData


def initialize_series():
    '''initializes an empty dictionary that can be filled with data from a series of similar scans

    returns
       empty dictionary with a set of list that can be populated with data of a scan series
    '''
    key = ["qx", "qz", "rsm",
           "intensity_qx", "intensity_qz",
           "scan", "temperature", "delay",
           "com_qz", "std_qz", "integral_qz",
           "fit_qz", "position_qz", "width_qz", "area_qz", "slope_qz", "offset_qz",
           "com_qx", "std_qx", "integral_qx",
           "fit_qx", "position_qx", "width_qx", "area_qx", "slope_qx", "offset_qx"]
    series = {k: [] for k in key}
    return series


def append_scan(series, scan_data, scan_number):
    series["scan"].append(scan_number)
    series["temperature"].append(scan_data['Temperature'])
    series["delay"].append(scan_data['delayMeas'])

    series["qz"].append(scan_data['xQz'])
    series["qx"].append(scan_data['xQx'])
    series["rsm"].append(np.transpose(np.sum(scan_data['Intensity'], 1)))

    series["intensity_qz"].append(scan_data['yQz'])
    series["intensity_qx"].append(scan_data['yQx'])

    com_qz, std_qz, integral_qz = tools.calc_moments(scan_data['xQz'], scan_data['yQz'])
    com_qx, std_qx, integral_qx = tools.calc_moments(scan_data['xQz'], scan_data['yQz'])

    series["com_qx"].append(com_qx)
    series["std_qx"].append(std_qx)
    series["integral_qx"].append(integral_qx)

    series["com_qz"].append(com_qz)
    series["std_qz"].append(std_qz)
    series["integral_qz"].append(integral_qz)
    return series


def fit_scan_qz(series, model, parameters_qz, scan_number):
    if scan_number == 0:
        fit_qz = model.fit(series["intensity_qz"][scan_number], parameters_qz, x=series["qz"][scan_number])
    else:
        parameters_qz = lm.Parameters()
        parameters_qz.add_many(('center', series["position_qz"][scan_number-1], True),
                               ('sigma', series["width_qz"][scan_number-1], True, 0),
                               ('amplitude', series["area_qz"][scan_number-1], True, 0),
                               ('slope', series["slope_qz"][scan_number-1], False),
                               ('intercept', series["offset_qz"][scan_number-1], True))
        fit_qz = model.fit(series["intensity_qz"][scan_number], parameters_qz, x=series["qz"][scan_number])

    series["fit_qz"].append(fit_qz)
    series["position_qz"].append(fit_qz.best_values["center"])
    series["width_qz"].append(fit_qz.best_values["sigma"])
    series["area_qz"].append(fit_qz.best_values["amplitude"])
    series["slope_qz"].append(fit_qz.best_values["sigma"])
    series["offset_qz"].append(fit_qz.best_values["intercept"])
    return series


def fit_scan_qx(series, model, parameters_qx, scan_number):

    if scan_number == 0:
        fit_qx = model.fit(series["intensity_qx"][scan_number], parameters_qx, x=series["qx"][scan_number])
    else:
        parameters_qx = lm.Parameters()
        parameters_qx.add_many(('center', series["position_qx"][scan_number-1], True),
                               ('sigma', series["width_qx"][scan_number-1], True, 0),
                               ('amplitude', series["area_qx"][scan_number-1], True, 0),
                               ('slope', series["slope_qx"][scan_number-1], False),
                               ('intercept', series["offset_qx"][scan_number-1], True))
        fit_qx = model.fit(series["intensity_qx"][scan_number], parameters_qx, x=series["qx"][scan_number])

    series["fit_qx"].append(fit_qx)
    series["position_qx"].append(fit_qx.best_values["center"])
    series["width_qx"].append(fit_qx.best_values["sigma"])
    series["area_qx"].append(fit_qx.best_values["amplitude"])
    series["slope_qx"].append(fit_qx.best_values["sigma"])
    series["offset_qx"].append(fit_qx.best_values["intercept"])
    return series


def calc_strain(qz, reference_point):
    strain = tools.rel_change(1/np.array(qz), 1/reference_point)
    return strain*1e3


def calc_changes_qz(series, index):
    series["qz_strain_fit"] = calc_strain(series["position_qz"], series["position_qz"][index])
    series["qz_width_change_fit"] = tools.rel_change(series["width_qz"], series["width_qz"][index])*100
    series["qz_area_change_fit"] = tools.rel_change(series["area_qz"], series["area_qz"][index])*100

    series["qz_strain_com"] = calc_strain(series["com_qz"], series["com_qz"][index])
    series["qz_width_change_com"] = tools.rel_change(series["std_qz"], series["std_qz"][index])*100
    series["qz_area_change_com"] = tools.rel_change(series["integral_qz"], series["integral_qz"][index])*100
    return series


def calc_changes_qx(series, index):
    series["qx_strain_fit"] = calc_strain(series["position_qx"], series["position_qx"][index])
    series["qx_width_change_fit"] = tools.rel_change(series["width_qx"], series["width_qx"][index])*100
    series["qx_area_change_fit"] = tools.rel_change(series["area_qx"], series["area_qx"][index])*100

    series["qx_strain_com"] = calc_strain(series["com_qx"], series["com_qx"][index])
    series["qx_width_change_com"] = tools.rel_change(series["std_qx"], series["std_qx"][index])*100
    series["qx_area_change_com"] = tools.rel_change(series["integral_qx"], series["integral_qx"][index])*100
    return series
