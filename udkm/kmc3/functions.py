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
