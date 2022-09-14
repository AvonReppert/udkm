# -*- coding: utf-8 -*-

import numpy as np
import os as os
import pandas as pd
import udkm.tools.functions as tools
import shutil
import lmfit as lm
import h5py
from matplotlib.colors import LogNorm

import udkm.tools.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def get_q_data(h5_file, scan_nr, tau_offset=300.4):
    '''loads data from a .h5 file into a scan dictionary

    Parameters
    -----------
    h5_file : str
        name of the h5 file to load
    scan_nr : integer
        number of the scan to extract from the file
    tau_offset : str
        offset value of tau_APD

    Returns
    --------
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
    --------
    initializes an analysis parameter dictionary with the sample name "P28b"

    >>> get_q_data()

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


def append_scan(series, scan, scan_number):
    '''appends a "scan" to an existing dictionary of scans ("series") using the appropriate scan number

    Example
    -------
    the following line appends "scan_data" to "series"
    >>> series = append_scan(series, scan_data, scan_number)

    '''
    series["scan"].append(scan_number)
    series["temperature"].append(scan['temperature'])
    series["delay"].append(scan['delay_measured'])

    series["qz"].append(scan['qz'])
    series["qx"].append(scan['qx'])
    series["rsm"].append(np.transpose(np.sum(scan['intensity'], 1)))

    series["intensity_qz"].append(scan['intensity_qz'])
    series["intensity_qx"].append(scan['intensity_qx'])

    com_qz, std_qz, integral_qz = tools.calc_moments(scan['qz'], scan['intensity_qz'])
    com_qx, std_qx, integral_qx = tools.calc_moments(scan['qx'], scan['intensity_qx'])

    series["com_qx"].append(com_qx)
    series["std_qx"].append(std_qx)
    series["integral_qx"].append(integral_qx)

    series["com_qz"].append(com_qz)
    series["std_qz"].append(std_qz)
    series["integral_qz"].append(integral_qz)
    return series


def fit_scan_qz(series, model, parameters_qz, scan_number):
    '''fits a series qz peaks using a simple lmfit model (gaussian + linear background)  and adds fitresult to "series"

    Suitable for a delay series or temperature series alike.
    If scan_number is "0" then it uses the provided parameters_qx otherwise it uses the best fit values from the
    previous fit.

    Example
    -------
    >>> series = fit_scan_qz(series, model, parameters_qx, i)

    '''

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
    '''fits a series qx peaks using a simple lmfit model (gaussian + linear background)  and adds fitresult to "series"

    Suitable for a delay series or temperature series alike.
    If scan_number is "0" then it uses the provided parameters_qx otherwise it uses the best fit values from the
    previous fit.

    Example
    -------
    >>> series = fit_scan_qx(series, model, parameters_qx, i)

    '''

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
    ''' returns the strain in 10^(-3) of a vector of q_z positions relative to a reference value '''
    strain = tools.rel_change(1/np.array(qz), 1/reference_point)
    return strain*1e3


def calc_changes_qz(series, index):
    ''' calculates the q_z strain (10^-3), rel. width and area change (%) relative to a reference scan given by index

    The returned dictionary then inclues the additional keys
        - "qz_strain_fit"
        - "qz_width_change_fit"
        - "qz_area_change_fit"
        - "qz_strain_com"
        - "qz_width_change_com"
        - "qz_area_change_com"

    Example
    -------
    >>> series = calc_changes_qz(series, i_ref)

    '''

    series["qz_strain_fit"] = calc_strain(series["position_qz"], series["position_qz"][index])
    series["qz_width_change_fit"] = tools.rel_change(series["width_qz"], series["width_qz"][index])*100
    series["qz_area_change_fit"] = tools.rel_change(series["area_qz"], series["area_qz"][index])*100

    series["qz_strain_com"] = calc_strain(series["com_qz"], series["com_qz"][index])
    series["qz_width_change_com"] = tools.rel_change(series["std_qz"], series["std_qz"][index])*100
    series["qz_area_change_com"] = tools.rel_change(series["integral_qz"], series["integral_qz"][index])*100
    return series


def calc_changes_qx(series, index):
    ''' calculates the q_x strain (10^-3), rel. width and area change (%) relative to a reference scan given by index

    The returned dictionary then inclues the additional keys
        - "qx_strain_fit"
        - "qx_width_change_fit"
        - "qx_area_change_fit"
        - "qx_strain_com"
        - "qx_width_change_com"
        - "qx_area_change_com"

    Example
    -------
    >>> series = calc_changes_qx(series, i_ref)

    '''
    series["qx_strain_fit"] = calc_strain(series["position_qx"], series["position_qx"][index])
    series["qx_width_change_fit"] = tools.rel_change(series["width_qx"], series["width_qx"][index])*100
    series["qx_area_change_fit"] = tools.rel_change(series["area_qx"], series["area_qx"][index])*100

    series["qx_strain_com"] = calc_strain(series["com_qx"], series["com_qx"][index])
    series["qx_width_change_com"] = tools.rel_change(series["std_qx"], series["std_qx"][index])*100
    series["qx_area_change_com"] = tools.rel_change(series["integral_qx"], series["integral_qx"][index])*100
    return series


def plot_rsm_overview(series, index, plot_log, i_ref):
    ''' returns an overview plot of the reciprocal spacemap including the projection on the qx and qz axis

    standard plot layout, where a reference fit of the scan with the number "i_ref" is added in grey.
    it is required that  fit_scan_qx and fit_scan_qz have been executed befor

    Example
    -------
    >>> f, ax0, ax1, ax2 = plot_rsm_overview(series, i, plot_log, i_ref)

    '''

    f = plt.figure(figsize=(5.2, 5.2))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 0.32], height_ratios=[0.5, 1], wspace=0.0, hspace=0.0)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[2])
    ax2 = plt.subplot(gs[3])

    X, Y = np.meshgrid(series["qx"][index], series["qz"][index])

    if plot_log:
        plotted = ax1.pcolormesh(X, Y,  series["rsm"][index], cmap=colors.cmap_1,
                                 norm=LogNorm(vmin=series["c_min_log"], vmax=series["c_max_log"]), shading="auto")
    else:
        plotted = ax1.pcolormesh(X, Y,  series["rsm"][index], cmap=colors.cmap_1,
                                 vmin=series["c_min_lin"], vmax=series["c_max_lin"], shading="auto")

    ax0.plot(series["qx"][i_ref], series["intensity_qx"][i_ref],  'o', color=colors.grey_3,
             label=str(int(np.round(series["temperature"][i_ref], 0)))+r"$\,$K")
    ax0.plot(series["qx"][i_ref], series["fit_qx"][i_ref].best_fit, '-', color=colors.grey_1, label="fit", lw=2)

    ax0.plot(series["qx"][index], series["intensity_qx"][index], 'o', color=colors.blue_1, label="sum")
    ax0.plot(series["qx"][index], series["fit_qx"][index].best_fit, '-', color=colors.red_1, label="fit", lw=2)

    ax2.plot(series["intensity_qz"][i_ref], series["qz"][i_ref], 'o', color=colors.grey_3)
    ax2.plot(series["fit_qz"][i_ref].best_fit, series["qz"][i_ref], '-', color=colors.grey_1, label="fit", lw=2)

    ax2.plot(series["intensity_qz"][index], series["qz"][index], 'o', color=colors.blue_1, label="sum")
    ax2.plot(series["fit_qz"][index].best_fit, series["qz"][index], '-', color=colors.red_1, label="fit", lw=2)

    ax1.set_xlabel(r'$q_\mathrm{x} \,\,\mathrm{\left(\frac{1}{\AA}\right)}$')
    ax1.set_ylabel(r'$q_\mathrm{z} \,\,\mathrm{\left(\frac{1}{\AA}\right)}$')
    cbaxes = f.add_axes([0.5, 0.21, 0.18, 0.027])
    cbar = plt.colorbar(plotted, cax=cbaxes, orientation='horizontal')
    cbar.set_label(r'I (cts)', fontsize=8)
    cbar.ax.tick_params(labelsize=8, direction='in')
    ax0.legend(bbox_to_anchor=(1, 1), loc="upper left")

    ax0.xaxis.tick_top()
    ax2.yaxis.tick_right()
    ax0.axes.xaxis.set_ticklabels([])
    ax2.axes.yaxis.set_ticklabels([])
    ax0.tick_params(axis='x', which='both', bottom='on', top='on', labelbottom='off')
    ax1.tick_params(axis='x', which='both', bottom='on', top='on', labelbottom='on')
    ax2.tick_params(axis='y', which='both', left='on', right='on', labelright='off')
    return f, ax0, ax1, ax2


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


def set_analysis_params(sample_name, **params):
    '''initializes a dictionary of parameters for the data analysis that can be set in the main script

    The resulting dictionary always contains the relevant folder names and the 'sample_name'.

    Additionally it allows to set the option to:
        - force a recalculation of the loop averages, to exclude 'bad_loops',
        - to manually sort the raw data into field up and field down
        - calculate time zero from the measurement results as delay of largest change.
        - copy the raw data from the measurement folder into a specified sample folder


    Parameters
    ----------
    sample_name : str
        Name of the sample used for the title of the generated plots.
    import_path : str
        Folder path of the raw data in the measurement folder. If given the raw data is copied to the samples folder.
    column_calc_t0 : str
        Results 'moke' or 'sum' which is used to caculate time zero. If given it is caculasted and printed.

    Returns
    -------
    analysis_params : dictionary
        - 'data_folder': (str) folder of measured data - default "raw_data"
        - 'export_folder': (str) folder of final analysis results - default "exported_results"
        - 'export_dict_folder': (str) folder of the exported dictionarys  - default "exported_analysis"
        - 'sample_name': (str) name of the sample as used for plot headlines and labeling
        - 'force_recalc': (bool) recalculate analysis if it already exists - default False
        - 'bad_field': (bool) is there a problem with the separation into field up and field down - default False
        - 'bad_loops': (list) list of lists of loops to exclude in the averaged as analysis result - default [[]].
        - 'import_path': (str, optional) path from which the measurement results should be copied to 'data_folder'
        - 'bool_import_data' (str, optional) set to True if 'import_path' exists
        - 'column_calc_t0': (str, optional) column name for time zero determination 'moke' or 'sum' - default 'moke'
        - 'bool_calc_t0': (bool, optional) True if 'column_calc_t0 exists' and forces automatic t0 determination
        - 'bool_plot_overview': (bool, optional) set True if overview plot is shown after analysis - default True

    Example
    -------
    initializes an analysis parameter dictionary with the sample name "P28b"

    >>> analysis_params =  set_analysis_params('P28b')
    {'data_folder': 'raw_data',
    'export_folder': 'exported_results',
    'export_dict_folder': 'exported_analysis'
    'sample_name': 'P28b',
    'force_recalc': False,
    'bad_field': False,
    'bad_loops': [[]],
    'bool_calc_t0': False,
    'bool_import_data': False,
    'bool_plot_overview': True}

    '''
    analysis_params = {}
    analysis_params['data_folder'] = 'raw_data'
    analysis_params['export_folder'] = 'exported_results'
    analysis_params['export_dict_folder'] = 'exported_analysis'
    analysis_params['sample_name'] = sample_name
    analysis_params['force_recalc'] = False
    analysis_params['bad_field'] = False
    analysis_params['bad_loops'] = [[]]
    analysis_params['bool_plot_overview'] = True

    if 'import_path' in params:
        analysis_params['import_path'] = params['import_path']
        analysis_params['bool_import_data'] = True
    else:
        analysis_params['bool_import_data'] = False

    if 'column_calc_t0' in params:
        analysis_params['column_calc_t0'] = params['column_calc_t0']
        analysis_params['bool_calc_t0'] = True
    else:
        analysis_params['bool_calc_t0'] = False

    return analysis_params


def get_export_information(scan_params, analysis_params):
    '''creates a dictionary with with export parameters

    The resulting dictionary contains:
        - plot_title
        - export_folder
        - file_name

    the results are made of  'date', 'time', 'rotation' and 'voltage' of the measurement.

    Parameters
    ----------
    scan_params : dictionary
        Containing the information of the measurement imported from the reference file.
    analysis_params : dictionary
        Set and potentially adjusted analysis paramaters including the 'sample_name'.

    Returns
    -------
    export_params : dictionary
        - 'data_path': folder structure  of the raw data
        - 'identification': same information but separated with whitespace used for printing the progress,
        - 'file_name': name of the exported overview file consiisting of the same information separated by underline
        - 'plot_title': additionally contains the sample name and is used as title for each plot for identification

    Example
    -------
    >>> export_params =  get_export_information(scan, analysis)
    {'data_path': '20220818_181818/Fluence/-1.0/5.0',
    'identification': '20220818 181818 -1.0 5.0',
    'file_name': '20200818_181818_-1.0_5.0'
    'plot_title': 'P28b  20220818  181818  -1.0  5.0V'
    }

    '''
    export_params = {}
    export_params['data_path'] = scan_params['date'] + '_' + scan_params['time'] + \
        '/Fluence/' + str(scan_params['rotation']) + '/' + str(scan_params['voltage'])
    export_params['identification'] = scan_params['date'] + ' ' + scan_params['time'] + \
        ' ' + str(scan_params['rotation']) + ' ' + str(scan_params['voltage'])
    export_params['file_name'] = scan_params['date'] + '_' + scan_params['time'] + \
        '_' + str(scan_params['rotation']) + '_' + str(scan_params['voltage'])
    export_params['plot_title'] = analysis_params['sample_name'] + '  ' + scan_params['date'] + '  ' + scan_params['time'] + \
        '  ' + str(scan_params['rotation']) + '  ' + str(scan_params['voltage']) + 'V'
    return export_params


def load_scan(ref_file_name, analysis_params, line):
    ''' loads the data of a measurement scan into a dictionary

    The resulting scan dictionary contains the parameters of the measurement and the loopwise averaged and
    analysed measurement results.

    The resulting dictionary contains:
      - voltage change for each field direction
      - their difference
      - their sum

    If this scan is not already exported or 'force_recalc' is set True these results are created from the
    'raw_data' by calling the function 'export_raw_data' that also creates a loopwise overview plot and
    enables the possibility to exclude 'bad_loops'. If 'bool_import_data' is True in 'analysis_params' it
    automatically copies of the raw data from the measurement folder to the sample folder.

    Parameters
    ----------
    ref_file_name : str
        File name of the reference file storing information about measurement.
    analysis_params : dictionary
        Set and potentially adjusted analysis paramaters including 'force_recalc' and 'bad_loops'.
    line : int
        The number of the measurement equivalent to line in the reference file.


    Returns
    -------
    scan : dictionary
        - 'date': (str) date of measurement
        - 'time': (str) time of measurement
        - 'power': (float) laser power without chopper [mW].
        - 'time_zero': (float) delay of pump-probe overlap in [ps] - default 0
        - 'temperature': (float) start temperature of measurement [K] - default 300
        - 'pump_angle': (float) angle between surface normal and pump beam [deg] - default 0
        - 'field_angle': (float) angle of the external field in respect to surface normal [deg] - default 0
        - 'double_pulse':  (int) first (0), second (1) or both pulses (2) in double pulse experiment - default 0
        - 'frontside': (int) whether (1) or not (0) the excitation happens at from the surface - default 1
        - 'field': (float) external magnetic field strength [mT] - default -1
        - 'fwhm_x': (float) horizontal pump width [microns] - default 1000
        - 'fwhm_y': (float) vertical pump width [microns] - default 1000
        - 'wavelength': (float) wavelength of pumppulse [nm] - default 800
        - 'pulse_length': (float) duration of pump pulse [fs] - default 120
        - 'fluence': (float) calculated from 'power', 'fwhm_x' and 'fwhm_y' in top-hat approximation [mJ/cm^2].
        - 'voltage': (float) set voltage that equals current in the electromagnet [A] default - 5
        - 'rotation': (float) position of rotatable stage (power or field angle) - default -1
        - 'raw_delay': (numpy array) delay axis as measured without time zero correction
        - 'delay': (numpy array) delays with time zero correction in reference file
        - 'sum': (numpy array) sum of the voltage change for opposite field directions
        - 'moke': (numpy array) difference of the voltage change for opposite field directions,
        - 'field_up': (numpy array) voltage change for field up
        - 'field_down': (numpy array) voltage change of field down

    Example
    -------
    >>> scan = load_scan('reference.txt', analysis, 1)
    {'date': '20220813',
    'time': '181818',
    'id': '20220813_181818',
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
    'fluence': 5.32,
    'raw_delay': np.array([110,112,114]),
    'delay': np.array([-2,0,2]),
    'sum': np.array([0,0.02,-0.02]),
    'moke': np.array([0,0.2,0.4]),
    'field_up': np.array([0,0.11,0.19]),
    'field_down': np.array([0,-0.09,-0.21])}

    '''
    scan_params = get_scan_parameter(ref_file_name, line)
    export_params = get_export_information(scan_params, analysis_params)

    if analysis_params['bool_import_data']:
        tools.make_folder(analysis_params['data_folder'] + '/' + export_params['data_path'])
        shutil.copyfile(analysis_params['import_path'] + '/' + export_params['data_path'] +
                        '/AllData_Reduced.txt', analysis_params['data_folder'] + '/' +
                        export_params['data_path'] + '/AllData_Reduced.txt')
        shutil.copyfile(analysis_params['import_path'] + '/' + export_params['data_path'] +
                        '/MOKE_Average.txt', analysis_params['data_folder'] + '/' +
                        export_params['data_path'] + '/MOKE_Average.txt')
        shutil.copyfile(analysis_params['import_path'] + '/' + export_params['data_path'] +
                        '/Hyteresis_Measurement_Parameters.txt', analysis_params['data_folder'] + '/' +
                        export_params['data_path'] + '/Hyteresis_Measurement_Parameters.txt')
        print('copy raw data from ' + analysis_params['import_path'] + export_params['data_path'] + ' ...')

    tools.make_folder(analysis_params['export_folder'])
    tools.make_folder(analysis_params['export_dict_folder'])

    overview_file = analysis_params['export_folder'] + '/overview_data_' + scan_params['date'] + '_' +\
        scan_params['time'] + '_' + str(scan_params['rotation']) + '_' + str(scan_params['voltage']) + '.txt'

    if os.path.isfile(overview_file) and not analysis_params['force_recalc']:
        print('Read in exported results of ' + export_params['identification'])
        data = np.genfromtxt(overview_file, comments='#', skip_footer=1)
    else:
        print('Calculate and export the final results.')
        export_raw_data(scan_params, analysis_params, export_params, line)
        data = np.genfromtxt(overview_file, comments='#', skip_footer=1)

    print('Time Zero is: ' + str(scan_params['time_zero']) + 'ps.')
    scan = {}
    for key in scan_params:
        scan[key] = scan_params[key]
    scan['raw_delay'] = data[:, 0] + scan_params['time_zero']
    scan['delay'] = data[:, 0]
    scan['sum'] = data[:, 1]
    scan['moke'] = data[:, 2]
    scan['field_up'] = data[:, 3]
    scan['field_down'] = data[:, 4]

    return scan


def export_raw_data(scan_params, analysis_params, export_params, line):
    '''reads in the raw data from the 'data_path' and plots an overview of for each measurement loop

    This function reads in the raw data from the ‘data_path’ as given by the function ‘get_export_information’
    and plots the loopise voltage change between pumped and unpumped for both field directions as well as
    their difference, their sum and loopwise aver . With 'bad_field' it is possible to manually set the field directions
    if this infomation is not given by the raw data file. Afterwards an average excluding 'bad_loops' is calculated
    and the delay axis is shifted according to the time zero given in the refernence file. The results are exported
    in an overview file.

    Parameters
    ----------
    scan_params : dictionary
        Parameters of the measurement
    analysis_params : dictionary
        Analysis paramaters including 'bool_calc_t0' and 'bad_loops'.
    export_params : dictionary
        Folder names and plot title
    line : int
        Number of the measurement in reference file is used as entry of the 'bad_loops' list. For single
        measurement and single 'bad_loops' entry this list is used to exclude bad loops.

    Example
    -------
    This line exports the measurement number 3 in the reference file and thereby excludes loop 1

    >>> export_raw_data(scan_params, {'bad_loops': [[1]], ...}, export_params, 3)
    '''

    tools.make_folder('plot_loopwise')
    tools.make_folder('plot_overview')
    tools.make_folder('data_loopwise')

    # col_index = 0
    col_diff_signal = 1
    # col_diode_a = 2
    # col_diode_b = 3
    # col_reference = 4
    col_chopper = 5
    col_delay = 6
    col_loop = 7
    col_voltage = 8
    print('analyze measurement ' + export_params['identification'])
    data_in = np.genfromtxt(analysis_params['data_folder'] + '/' + export_params['data_path'] +
                            '/AllData_Reduced.txt', skip_header=1)
    data_raw = data_in[np.isfinite(data_in[:, 1]), :]

    if len(data_raw) > 0:
        unique_delays = np.unique(data_raw[:, col_delay])
        loops = int(np.max(data_raw[:, col_loop]))
        data_avg_result = np.zeros((len(unique_delays), 5))
        data_loop_field_up = np.zeros((len(unique_delays), loops))
        data_loop_field_down = np.zeros((len(unique_delays), loops))

        s_pumped = data_raw[:, col_chopper] == 1
        s_not_pumped = data_raw[:, col_chopper] == 0
        s_field_up = data_raw[:, col_voltage] > 0
        s_field_down = data_raw[:, col_voltage] <= 0

        if analysis_params['bad_field']:
            print('Magnetic field is manually sorted.')
            n_all_delays = np.sum(data_raw[:, col_loop] == 1)/2
            field_array = np.zeros(len(data_raw[:, col_loop]))
            for loop in range(loops-1):
                field_array[int(0+(2*loop*n_all_delays)): int(n_all_delays+(2*loop*n_all_delays))] = 1
            s_field_up = field_array == 1
            s_field_down = field_array == 0

        for array in [data_avg_result, data_loop_field_down, data_loop_field_up]:
            array[:, 0] = unique_delays - scan_params['time_zero']

        for loop in range(loops-1):
            s_loop = data_raw[:, col_loop] == loop+1
            for i, t in enumerate(unique_delays):
                s_delay = data_raw[:, col_delay] == t
                s_field_up_pumped = ((s_loop & s_field_up) & s_pumped) & s_delay
                s_field_up_not_pumped = ((s_loop & s_field_up) & s_not_pumped) & s_delay
                s_field_down_pumped = ((s_loop & s_field_down) & s_pumped) & s_delay
                s_field_down_not_pumped = ((s_loop & s_field_down) & s_not_pumped) & s_delay

                data_loop_field_up[i, loop + 1] = np.mean(data_raw[s_field_up_pumped, col_diff_signal]) - \
                    np.mean(data_raw[s_field_up_not_pumped, col_diff_signal])
                data_loop_field_down[i, loop + 1] = np.mean(data_raw[s_field_down_pumped, col_diff_signal]) - \
                    np.mean(data_raw[s_field_down_not_pumped, col_diff_signal])
        tools.write_list_to_file('data_loopwise/field_up_' + export_params['file_name'] + '.txt',
                                 'time (ps)    data plus loopwise', data_loop_field_up)
        tools.write_list_to_file('data_loopwise/field_down_' + export_params['file_name'] + '.txt',
                                 'time (ps)    data minus loopwise', data_loop_field_down)

        for i, t in enumerate(unique_delays):
            s_delay = unique_delays == t
            if line + 1 > len(analysis_params['bad_loops']):
                bad_loops = analysis_params['bad_loops'][0]
                good_loops = []
                for loop in range(loops-1):
                    if loop+1 not in bad_loops:
                        good_loops.append(loop+1)
            else:
                bad_loops = analysis_params['bad_loops'][line]
                good_loops = []
                for loop in range(loops-1):
                    if loop+1 not in bad_loops:
                        good_loops.append(loop+1)

            data_avg_result[i, 1] = np.mean(data_loop_field_up[s_delay, good_loops]) + \
                np.mean(data_loop_field_down[s_delay, good_loops])
            data_avg_result[i, 2] = np.mean(data_loop_field_up[s_delay, good_loops]) - \
                np.mean(data_loop_field_down[s_delay, good_loops])
            data_avg_result[i, 3] = np.mean(data_loop_field_up[s_delay, good_loops])
            data_avg_result[i, 4] = np.mean(data_loop_field_down[s_delay, good_loops])

        tools.write_list_to_file('exported_results/overview_data_' + export_params['file_name'] + '.txt',
                                 u'time (ps)\t sum signal (V)\t MOKE (V)\t field up signal (V)'
                                 + ' \t field down signal (V)', data_avg_result)

        # Plot data loopwise
        c_map = colors.cmap_1

        plt.figure(figsize=(5.2, 5.2 / 0.68))
        gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
        ax1 = plt.subplot(gs[0])
        ax2 = plt.subplot(gs[1])
        ax3 = plt.subplot(gs[2])

        for loop in range(loops-1):
            ax1.plot(data_loop_field_up[:, 0], data_loop_field_up[:, loop + 1], '-',
                     color=c_map(loop / loops), lw=1, label=str(loop + 1))
            ax2.plot(data_loop_field_down[:, 0], data_loop_field_down[:, loop + 1],
                     '-', color=c_map(loop / loops), lw=1, label=str(loop + 1))
            ax3.plot(data_loop_field_down[:, 0],
                     data_loop_field_up[:, loop + 1] - data_loop_field_down[:, loop + 1],
                     '-', color=c_map(loop / loops), lw=1, label=str(loop + 1))

        ax1.plot(data_loop_field_up[:, 0], np.mean(data_loop_field_up[:, 1:], axis=1), '-',
                 color=colors.grey_1, lw=2, label="avg")
        ax2.plot(data_loop_field_up[:, 0], np.mean(data_loop_field_down[:, 1:], axis=1), '-',
                 color=colors.grey_1, lw=2, label="avg.")
        ax3.plot(data_loop_field_up[:, 0], np.mean(data_loop_field_up[:, 1:], axis=1) -
                 np.mean(data_loop_field_down[:, 1:], axis=1), '-', color=colors.grey_1, lw=2, label="avg.")

        ax1.set_ylabel('field up  signal \n' + r'$\mathrm{S_{+}} (\mathrm{V})$')
        ax2.set_ylabel('field down signal \n' + r' $\mathrm{S_{-}} (\mathrm{V})$')
        ax3.set_ylabel('MOKE signal \n' r'$\mathrm{S_{+}\,\,-\,\, S_{-}}$ ($\,\mathrm{V}$)')
        ax3.set_xlabel('time (ps)')

        for ax in [ax1, ax2, ax3]:
            ax.legend(loc=1, fontsize=9, ncol=7, handlelength=1, columnspacing=1.5)
            ax.set_xlim((np.min(unique_delays), np.max(unique_delays)))
        ax1.xaxis.set_ticks_position('top')
        ax2.set_xticklabels([])
        ax1.set_title(export_params['plot_title'], pad=13)

        plt.savefig('plot_loopwise/loopwise_' + export_params['file_name'] + '.png', dpi=150)
        plt.show()


def plot_overview(ref_file_name, scan, analysis_params, line):
    '''Plots an overview of a measurement for a specified line in the reference file

    This function plots and saves the exported results in the dictionary 'scan' for a certain measurement
    in the given line of the reference file.

    Parameters
    ----------
    ref_file_name : str
        Name of the reference file containing the measurement parameters
    scan : dictionary
        Contains the parameters and the result of the measurement that are plotted
    analysis_params : dictionary
        Contains the parameters of the data analysis including the 'sample_name'
    line : int
        Number of the measurement in reference file.

    Example
    -------
    >>> plot_overview('reference.txt', scan, analysis_params, 3)

    '''
    scan_params = get_scan_parameter(ref_file_name, line)
    export_params = get_export_information(scan_params, analysis_params)

    print('Plot the overview over the exported results...')
    plt.figure(figsize=(5.2, 5.2/0.68))
    gs = gridspec.GridSpec(3, 1, wspace=0, hspace=0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    ax1.plot(scan["delay"], scan["field_up"], label="field_up", color=colors.red_1, lw=2)
    ax1.plot(scan["delay"], scan["field_down"], label="field_down", color=colors.blue_1, lw=2)
    ax2.plot(scan["delay"], scan["sum"], label="sum", color=colors.orange_2, lw=2)
    ax3.plot(scan["delay"], scan["moke"], label="moke", color=colors.grey_1, lw=2)

    ax1.set_ylabel(r'single signal  ($\,\mathrm{V}$)' + "\n" + r'$\mathrm{S_{+/-}   = I_{+/-}^{1} - I_{+/-}^{0}}$ ')
    ax2.set_ylabel(r'sum signal  ($\,\mathrm{V}$)' + "\n" + r'$\mathrm{S_{+}\,\, +\,\, S_{-}}$ ')
    ax3.set_ylabel(r'MOKE signal ($\,\mathrm{V}$)' + "\n" + r'$\mathrm{S_{+} \,\,-\,\, S_{-}}$')
    ax3.set_xlabel('time (ps)')

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(min(scan["delay"]), max(scan["delay"]))
        ax.legend(loc=4)
        ax.axhline(y=0, ls="--", color="grey")

    ax1.xaxis.set_ticks_position('top')
    ax2.set_xticklabels([])

    ax1.set_title(export_params['plot_title'], pad=13)
    plt.savefig('plot_overview/overview_' + export_params['file_name'] + '.png', dpi=150)


def load_series(ref_file_name, analysis_params):
    '''creates a dictionary that contains the parameters, data and analysis of a series as lists

    This function creates a dictionary containing the parameters of all measurements and their loopwise averaged
    and analysed measurement results given by the voltage change for each field direction, their difference and sum.
    This function uses the functions ‘load_scan’ and ‘plot_overview’ and iterates over all lines in the reference_file.
    If these measurements are not already exported or ‘force_recalc’ is set True these results are created from the
    ‘raw_data’ by the function ‘export_raw_data’. This alsoo creates a loopwise plot and enables the possibility to
    exclude ‘bad_loops’. If given in ‘analysis_params’ created by the function ‘set_analysis_params’ it  automatically
    copies the raw data from the measurement folder to the respective sample folder.

    Parameters
    ----------
    ref_file_name : str
        File name of the reference file storing information about measurement.
    analysis_params : dictionary
        Set and potentially adjusted analysis paramaters including 'force_recalc' and 'bad_loops'. 'bad_loops' should
        be given as list of lists with the same length as the reference file.

    Returns
    -------
    series : dictionary
        - 'date': (list of str) date of measurement
        - 'time': (list of str) time of measurement
        - 'power': (list of float) laser power without chopper [mW].
        - 'time_zero': (list of float) delay of pump-probe overlap in [ps] - default 0
        - 'temperature': (list of float) start temperature of measurement [K] - default 300
        - 'pump_angle': (list float) angle between surface normal and pump beam [deg] - default 0
        - 'field_angle': (list float) angle of the external field in respect to surface normal [deg] - default 0
        - 'double_pulse': (list of int) first (0), second (1) or both pulses (2) in double pulse experiment - default 0
        - 'frontside': (list of int) whether (1) or not (0) the excitation happens at from the surface - default 1
        - 'field': (list of float) external magnetic field strength [mT] - default -1
        - 'fwhm_x': (list of float) horizontal pump width [microns] - default 1000
        - 'fwhm_y': (list of float) vertical pump width [microns] - default 1000
        - 'wavelength': (list of float) wavelength of pumppulse [nm] - default 800
        - 'pulse_length': (list of float) duration of pump pulse [fs] - default 120
        - 'fluence': (list of float) calculated from 'power', 'fwhm_x' and 'fwhm_y' in top-hat approximation [mJ/cm^2].
        - 'voltage': (list of float) set voltage that equals current in the electromagnet [A] default - 5
        - 'rotation': (list of float) position of rotatable stage (power or field angle) - default -1
        - 'raw_delay': (list of numpy array) delay axis as measured without time zero correction
        - 'delay': (list of numpy array) delays with time zero correction in reference file
        - 'sum': (list of numpy array) sum of the voltage change for opposite field directions
        - 'moke': (list of numpy array) difference of the voltage change for opposite field directions,
        - 'field_up': (list of numpy array) voltage change for field up
        - 'field_down': (lisft of numpy array) voltage change of field down

    Example
    -------
    >>> series = load_series('reference.txt', analysis)
    {'date': ['20220813', 20220813'],
    'time': ['181818', '191919'],
    'id': ['20220813_181818', '20220813_191919'],
    'rotation': [-1.0, 0.0],
    'voltage': [5, 5],
    'power': [50, 50],
    'fwhm_x': [1100, 1100],
    'fwhm_y': [1100, 1100],
    'temperature': [300, 300],
    'field': [525, 525],
    'pump_angle': [0, 0],
    'field_angle': [0, 0],
    'time_zero': [0, 0],
    'double_pulse': [0, 0],
    'frontside': [0, 0],
    'wavelength': [800, 800],
    'pulse_length': [120, 120],
    'fluence': [5.32, 5.32],
    'raw_delay': [np.array([110,112,114]), np.array([110,112,114])],
    'delay': [np.array([-2,0,2]), np.array([-2,0,2])],
    'sum': [np.array([0,0.02,-0.02]), np.array([0,-0.02,0.02])],
    'moke': [np.array([0,0.2,0.4]), np.array([0,0.4,0.8])],
    'field_up': [np.array([0,0.11,0.19]), np.array([0,0.19,0.41])],
    'field_down': [np.array([0,-0.09,-0.21]), np.array([0,-0.21,-0.39])]
    }

    '''
    ref_file = pd.read_csv('reference_data/' + ref_file_name, delimiter="\t", header=0, comment="#")
    measurements = len(ref_file['date'])
    ref_scan_params = get_scan_parameter(ref_file_name, 0)

    print('Initialize the series dictionary...')
    series = {}
    for key in ref_scan_params:
        series[key] = []
    series['raw_delay'] = []
    series['delay'] = []
    series['sum'] = []
    series['moke'] = []
    series['field_up'] = []
    series['field_down'] = []

    print('Analyze results of all measurements...')
    for line in range(measurements):
        scan = load_scan(ref_file_name, analysis_params, line)
        if analysis_params["bool_plot_overview"]:
            plot_overview(ref_file_name, scan, analysis_params, line)
        for key in series:
            series[key].append(scan[key])

    return series
