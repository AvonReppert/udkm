import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py as h5
from scipy.interpolate import interp1d
import udkm.tools.functions as tools
import udkm.tools.colors as colors
import os
import matplotlib.gridspec as gridspec


teststring = "Successfully loaded udkm.opp.functions"


def get_scan_parameter(parameter_file_name, line):
    '''Loads the measurement parameters of the given line in the parmeter file to the parameter dictionary'''

    params = {'line': line}
    param_file = pd.read_csv(parameter_file_name, delimiter="\t", header=0, comment="#")
    header = list(param_file.columns.values)

    for i, entry in enumerate(header):
        if entry == 'date' or entry == 'time':
            params[entry] = tools.timestring(int(param_file[entry][line]))
        else:
            params[entry] = param_file[entry][line]

    # set default params
    params["bool_force_reload"] = False

    params["rep_rate"] = 5000
    params["pump_angle"] = 8  # Value from labbook
    params["probe_method"] = "transmission"
    params["symmetric_colormap"] = True

    # default values for dispersion correciton
    params["method"] = "max"
    params["range_wl"] = [450, 740]
    params["degree"] = 3
    params["file"] = False

    params["prefix"] = "Spectro1_"

    params["id"] = params["date"] + "_" + params["time"]

    params["measurements"] = len(param_file[entry])

    params["data_directory"] = "data\\"
    params["scan_directory"] = "scan_export\\"

    params["filename_pump"] = params["prefix"] + "Pump.txt"
    params["filename_probe"] = params["prefix"] + "Probe.txt"
    params["filename_averaged"] = params["prefix"] + "PumpProbe_AveragedData.txt"

    return params


def initialize_scan(params):
    '''initializes the scan dictionary'''
    scan = {}

    # required parameters

    scan["date"] = params["date"]
    scan["time"] = params["time"]
    scan["id"] = params["id"]
    scan["name"] = params["name"]
    scan["parameters_missing"] = False
    scan["filename_averaged"] = params["filename_averaged"]
    scan["data_directory"] = params["data_directory"] + "\\" + scan["id"] + "\\"
    scan["scan_directory"] = params["scan_directory"]
    # values for frog fit
    scan["method"] = params["method"]
    scan["range_wl"] = params["range_wl"]
    scan["degree"] = params["degree"]
    scan["file"] = params["file"]

    # important parameters for title
    entry_list = ["power", "rep_rate", "pump_angle", "T", "symmetric_colormap",
                  "fwhm_x", "fwhm_y", "pump_wl", "probe_method", "filename_pump", "filename_probe"]
    for entry in entry_list:
        if entry in params:
            scan[entry] = params[entry]
        else:
            print("Warning: " + str(entry) + " is missing in parameter file")
            scan["parameters_missing"] = True

    # optional user defined parameters
    entry_list = ["t0", "slice_wl", "slice_wl_width", "slice_delay",
                  "slice_delay_width", "delay_min", "delay_max", "wl_min", "wl_max"]
    for entry in entry_list:
        if entry in params:
            scan[entry] = params[entry]

    scan["fluence"] = tools.calc_fluence(scan["power"], scan["fwhm_x"],
                                         scan["fwhm_y"], scan["pump_angle"], scan["rep_rate"])

    if scan["parameters_missing"]:
        scan["title_text"] = scan["id"]
    else:
        scan["title_text"] = scan["name"] + "  " + scan["date"] + " " + scan["time"] + " " + \
            str(scan["T"]) + r"$\,$K " + " " + str(scan["pump_wl"]) + r"$\,$nm  " + str(np.round(scan["fluence"], 1)) + \
            r"$\,\mathrm{\frac{mJ}{\,cm^2}}$ "
    return scan


def load_data_average(params):
    '''Loads the average data for a given parameter set into a scan dictionary and returns overview plot'''

    scan = initialize_scan(params)

    # load data into scan dictionary
    scan["data_averaged"] = np.genfromtxt(scan["data_directory"]+scan["filename_averaged"], comments="#")

    scan["wavelength"] = scan["data_averaged"][0, 1:]
    scan["delay_raw_unique"] = scan["data_averaged"][1:, 0]
    scan["data_averaged"] = np.transpose(scan["data_averaged"][1:, 1:])

    # overview plot of the data
    print("overview plot " + scan["name"] + " " + scan["date"] + " " +
          scan["time"]+" " + str(scan["T"])+"K " + str(scan["fluence"])+"mJ/cm^2 ")
    if scan["symmetric_colormap"]:
        if "signal_level" in params:
            scan["c_min"] = -1*params["signal_level"]
            scan["c_max"] = params["signal_level"]
        else:
            scan["c_min"] = -1*np.max(np.abs(scan["data_averaged"]))
            scan["c_max"] = 1*np.max(np.abs(scan["data_averaged"]))
    else:
        scan["c_min"] = np.min(scan["data_averaged"])
        scan["c_max"] = np.max(scan["data_averaged"])

    X, Y = np.meshgrid(scan["delay_raw_unique"], scan["wavelength"])

    plt.figure(1, figsize=(5.2/0.68, 5.2))
    plt.pcolormesh(X, Y, 100*scan["data_averaged"],
                   cmap=colors.fireice(), vmin=100*scan["c_min"], vmax=100*scan["c_max"], shading='nearest')
    plt.xlabel(r'delay (ps)')
    plt.ylabel(r'wavelength (nm)')

    plt.title(scan["title_text"])

    cb = plt.colorbar()

    if scan["probe_method"] == "transmission":
        cb.ax.set_title(r'$\Delta T/T$ (%)')
    else:
        cb.ax.set_title(r'$\Delta R/R$ (%)')

    return scan


def load_data(params):
    '''Loads all data given parameter set into a scan dictionary and returns overview plot'''

    if (params["bool_force_reload"] or not(os.path.isfile(params["scan_directory"]+params["id"]+".h5"))):
        print("load data from : " + params["data_directory"]+params["id"])
        # initialize scan dictionary
        scan = initialize_scan(params)

        # load data into scan dictionary
        scan["data_averaged"] = np.genfromtxt(scan["data_directory"]+scan["filename_averaged"], comments="#")
        scan["data_pump"] = np.genfromtxt(scan["data_directory"]+scan["filename_pump"], comments="#")
        scan["data_probe"] = np.genfromtxt(scan["data_directory"]+scan["filename_probe"], comments="#")

        scan["wavelength"] = scan["data_averaged"][0, 1:]
        scan["delay_raw_unique"] = scan["data_averaged"][1:, 0]

        scan["data_averaged"] = np.transpose(scan["data_averaged"][1:, 1:])
        scan["loop_array"] = scan["data_pump"][1:, 0]
        scan["delay_raw_set"] = scan["data_pump"][1:, 1]
        scan["delay_raw_real"] = scan["data_pump"][1:, 2]

        scan["data_pump"] = np.transpose(scan["data_pump"][1:, 3:])
        scan["data_probe"] = np.transpose(scan["data_probe"][1:, 3:])
        # clean data

    else:
        print("reload scan from: " + params["scan_directory"]+params["id"]+".h5")
        scan = load_scan(params["date"], params["time"], params["scan_directory"])

    # overview plot of the data
    print("overview plot " + scan["name"] + " " + scan["date"] + " " +
          scan["time"]+" " + str(scan["T"])+"K " + str(scan["fluence"])+"mJ/cm^2 ")

    if scan["symmetric_colormap"]:
        if "signal_level" in params:
            scan["c_min"] = -1*params["signal_level"]
            scan["c_max"] = params["signal_level"]
        else:
            scan["c_min"] = -1*np.max(np.abs(scan["data_averaged"]))
            scan["c_max"] = 1*np.max(np.abs(scan["data_averaged"]))
    else:
        scan["c_min"] = np.min(scan["data_averaged"])
        scan["c_max"] = np.max(scan["data_averaged"])

    X, Y = np.meshgrid(scan["delay_raw_unique"], scan["wavelength"])

    plt.figure(1, figsize=(5.2/0.68, 5.2))
    plt.pcolormesh(X, Y, 100*scan["data_averaged"],
                   cmap=colors.fireice(), vmin=100*scan["c_min"], vmax=100*scan["c_max"], shading='nearest')
    plt.xlabel(r'delay (ps)')
    plt.ylabel(r'wavelength (nm)')

    plt.title(scan["title_text"])

    cb = plt.colorbar()

    if scan["probe_method"] == "transmission":
        cb.ax.set_title(r'$\Delta T/T$ (%)')
    else:
        cb.ax.set_title(r'$\Delta R/R$ (%)')

    # %% clean data from errors where the spectrometer gave values smaller <= 5
    pump_signal_errors = np.sum(scan["data_pump"], axis=0) <= 0
    probe_signal_errors = np.sum(scan["data_probe"], axis=0) <= 0
    errors = np.logical_or(pump_signal_errors, probe_signal_errors)
    select_data = np.logical_not(errors)

    scan["delay_raw_set"] = scan["delay_raw_set"][select_data]
    scan["delay_raw_real"] = scan["delay_raw_real"][select_data]
    scan["loop_array"] = scan["loop_array"][select_data]

    scan["data_pump"] = scan["data_pump"][:, select_data]
    scan["data_probe"] = scan["data_probe"][:, select_data]

    scan["data_raw"] = (scan["data_pump"] - scan["data_probe"])/scan["data_probe"]

    scan["data"] = np.zeros((len(scan["wavelength"]), len(scan["delay_raw_unique"])))

    for i, t in enumerate(scan["delay_raw_unique"]):
        select_time = scan["delay_raw_set"] == t
        scan["data"][:, i] = np.mean(scan["data_raw"][:, select_time], axis=1)

    # shift t0
    scan["delay_set"] = scan["delay_raw_set"] - scan["t0"]
    scan["delay_real"] = scan["delay_raw_real"] - scan["t0"]
    scan["delay_unique"] = scan["delay_raw_unique"] - scan["t0"]

    scan["wl_min"] = np.min(scan["wavelength"])
    scan["wl_max"] = np.max(scan["wavelength"])

    scan["delay_min"] = np.min(scan["delay_unique"])
    scan["delay_max"] = np.max(scan["delay_unique"])

    entry_list = ["delay_min", "delay_max", "wl_min", "wl_max", "signal_level"]
    for entry in entry_list:
        if entry in params:
            scan[entry] = params[entry]

    scan = generate_slices(scan, data_key="data")
    return scan


def generate_slices(scan, data_key="data"):
    # generate wl slices
    if "slice_wl" in scan:
        wl_slices = len(scan["slice_wl"])
        scan["wl_labels"] = list()
        if wl_slices > 0:
            scan["wl_slices"] = np.zeros((wl_slices, len(scan["delay_unique"])))

            for i, wl in enumerate(scan["slice_wl"]):

                wl_min = scan["slice_wl"][i] - scan["slice_wl_width"][i]
                wl_max = scan["slice_wl"][i] + scan["slice_wl_width"][i]
                t_min = np.min(scan["delay_unique"])
                t_max = np.max(scan["delay_unique"])
                _, _, data_cut, _, _ = tools.set_roi_2d(scan["delay_unique"], scan["wavelength"], scan[data_key],
                                                        t_min, t_max, wl_min, wl_max)
                scan["wl_slices"][i, :] = np.mean(data_cut, axis=0)
                scan["wl_labels"].append(str(int(scan["slice_wl"][i])))

        else:
            scan["wl_slices"] = np.zeros((wl_slices, 1))
            scan["wl_slices"][0, :] = np.sum(scan[data_key], axis=0)
            scan["wl_labels"].append("integral")

    # generate delay slices
    if "slice_delay" in scan:
        t_slices = len(scan["slice_delay"])
        scan["delay_labels"] = list()
        if t_slices > 0:
            scan["delay_slices"] = np.zeros((t_slices, len(scan["wavelength"])))
            for i, t in enumerate(scan["slice_delay"]):

                wl_min = np.min(scan["wavelength"])
                wl_max = np.max(scan["wavelength"])
                t_min = scan["slice_delay"][i] - scan["slice_delay_width"][i]
                t_max = scan["slice_delay"][i] + scan["slice_delay_width"][i]
                _, _, data_cut, _, _ = tools.set_roi_2d(scan["delay_unique"], scan["wavelength"], scan["data"],
                                                        t_min, t_max, wl_min, wl_max)
                scan["delay_slices"][i, :] = np.mean(data_cut, axis=1)
                scan["delay_labels"].append(str(int(scan["slice_delay"][i])))

        else:
            scan["delay_slices"] = np.zeros((t_slices, 1))
            scan["delay_slices"][0, :] = np.sum(scan[data_key], axis=1)
            scan["delay_labels"].append("integral")
    return scan


def save_scan(scan: dict) -> None:
    """
    Saves a scan dictionary to an H5 file in the specified scan directory.

    Parameters
    ----------
    scan : dict
        A dictionary containing the scan data, including the scan directory and ID.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the `scan` dictionary does not contain the keys `"scan_directory"` and `"id"`.
    """
    try:
        filename = scan["scan_directory"] + scan["id"] + ".h5"
    except KeyError:
        raise KeyError("The `scan` dictionary must contain the keys `scan_directory` and `id`.")

    with h5.File(filename, "w") as hfile:
        for key, value in scan.items():
            hfile[key] = value


def load_scan(date: str, time: str, scan_directory: str) -> dict:
    """
    Load a scan from an H5 file into a dictionary.

    Parameters
    ----------
    date : str
        A string encoding the date of the scan in the format "YYYYMMDD".
    time : str
        A string encoding the time of the scan in the format "HHMMSS".
    scan_directory : str
        The path to the directory containing the H5 files.

    Returns
    -------
    dict
        A dictionary containing the scan data.

    Raises
    ------
    FileNotFoundError
        If the specified H5 file cannot be found.
    """
    filename = scan_directory + date + "_" + time + ".h5"
    scan = {}
    with h5.File(filename, 'r') as f:
        for key in f.keys():
            if isinstance(f[key], h5.Dataset):
                value = f[key][()]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                scan[key] = value
            elif isinstance(f[key], h5.Group):
                scan[key] = load_scan(date, time, filename + '/' + key)

    return scan


# def load_scan(date, time, scan_directory):
#     '''Loads a scan from a python pickle into a scan dictionary'''
#     path_string = scan_directory + tools.timestring(date)+"_"+tools.timestring(time)+".pickle"
#     return pickle.load(open(path_string, "rb"))


def plot_overview(scan, data_key="data"):

    scan = generate_slices(scan, data_key=data_key)

    '''yields an overview plot of the data with selected lineouts'''
    fig = plt.figure(figsize=(12, 12*0.68))
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[3, 1],
                           height_ratios=[1, 2],
                           wspace=0.0, hspace=0.0)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    ## Plot 1) Top Left: Horizontal Profile #####################

    for i, intensity in enumerate(scan["wl_slices"]):
        plotted = ax1.plot(scan["delay_unique"], 100*intensity, lw=1,  label=str(scan["wl_labels"][i]) + r'$\,$nm')
        if not(scan["wl_labels"][i] == "integral"):
            ax3.axhline(y=scan["slice_wl"][i], lw=1, ls='--', color=plotted[0].get_color())
        #ax1.axvline(x=0, ls="--", lw=0.5, color=colors.grey_3)

    ax1.axhline(y=0, ls="--", lw=0.5, color=colors.grey_2)

    ax1.set_xlabel(r' ')

    # ax1.set_ylim(signalStrengthSliceMin, signalStrengthSliceMax)

    ax1.set_xlim(scan["delay_min"], scan["delay_max"])

    ax1.legend(loc=1, bbox_to_anchor=(1.175, 1.05))

    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")

    ## Plot 3) Bottom Left: Colormap of the Profile#############

    if scan["symmetric_colormap"]:
        if "signal_level" in scan:
            c_min = -1*scan["signal_level"]
            c_max = scan["signal_level"]
        else:
            c_min = -1*np.max(np.abs(scan["data_averaged"]))
            c_max = 1*np.max(np.abs(scan["data_averaged"]))
    else:
        c_min = np.min(scan["data_averaged"])
        c_max = np.max(scan["data_averaged"])

    X, Y = np.meshgrid(scan["delay_unique"], scan["wavelength"])
    plotted = ax3.pcolormesh(X, Y, 100*scan[data_key], cmap=colors.fireice(), vmin=100*c_min, vmax=100*c_max)

    ax3.axis([scan["delay_min"], scan["delay_max"], scan["wl_min"], scan["wl_max"]])

    ax3.set_xlabel(r'delay $t$ (ps)')
    ax3.set_ylabel(r'wavelength $\mathrm{\lambda}$ (nm)')

    cbaxes3 = fig.add_axes([0.47, 0.65, 0.22, 0.015])
    cbar = plt.colorbar(plotted, cax=cbaxes3, orientation='horizontal')

    if scan["probe_method"] == "transmission":
        cbar.ax.set_title(r'$\Delta T/T$ (%)')
        ax1.set_ylabel(r'$\Delta T/T$ (%)')
        ax4.set_xlabel(r'$\Delta T/T$ (%)')

    else:
        cbar.ax.set_title(r'$\Delta R/R$ (%)')
        ax1.set_ylabel(r'$\Delta R/R$ (%)')
        ax4.set_xlabel(r'$\Delta R/R$ (%)')

    cbar.ax.tick_params('both', length=5, width=0.5, which='major', direction='in')
    cl = plt.getp(cbar.ax, 'xmajorticklabels')
    cbaxes3.xaxis.set_ticks_position("top")
    cbaxes3.xaxis.set_label_position("top")

    ax2.axis('off')

    ## Plot 4) Bottom Right Slice at t = 0 #####################

    for i, intensity in enumerate(scan["delay_slices"]):
        plotted = ax4.plot(100*intensity, scan["wavelength"],  lw=1,  label=str(scan["delay_labels"][i]) + r'$\,$ps')
        if not(scan["delay_labels"][i] == "integral"):
            ax3.axvline(x=scan["slice_delay"][i], lw=1, ls='--', color=plotted[0].get_color())

    ax4.axvline(x=0, ls="--", lw=0.5, color=colors.grey_3)

    ax4.set_ylim([scan["wl_min"], scan["wl_max"]])
    if "signal_level" in scan:
        ax1.set_ylim([-100*scan["signal_level"], 100*scan["signal_level"]])
        ax4.set_xlim([-100*scan["signal_level"], 100*scan["signal_level"]])

    ax4.set_ylabel(r'$\mathrm{\lambda}$ (nm)')

    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")

    ax4.legend(loc=1,  bbox_to_anchor=(1.05, 1.5))
    ax4.axvline(x=0, lw=1, ls='--', color='grey')

    fig.suptitle(scan["title_text"])


def dispersion_fit(scan):
    """
    Calculates the fit function for the data and plots it.

    Parameters
    ----------
    scan : dict
        A dictionary containing the following keys:
        - "delay_unique" (array-like): Array of unique delay values.
        - "wavelength" (array-like): Array of wavelength values.
        - "data" (array-like): 2D array of data.
        - "method" (str): Method for Dispersion determination.
        - "range_wl" (list): List of wavelength ranges.
        - "degree" (int): Degree of polynomial fit.
        - "file" (str or bool): File name or False if no file.

    Returns
    ----------
    dict
        Updated scan dictionary with the "fit_function" key.

    Raises
    ----------
    ValueError
        If an invalid method for FROG determination is provided.
    """

    delays = scan["delay_unique"]
    wl = scan["wavelength"]
    data = pd.DataFrame(scan["data"])
    method = scan["method"]
    range_wl = scan["range_wl"]
    degree = scan["degree"]
    file = scan["file"]
    fit_wl = np.arange(scan["wl_min"], scan["wl_max"]+1)

    if file is False:
        if method == 'maxSlope':
            I0 = np.argmax(np.diff(data, axis=1), axis=1)
        elif method == 'minSlope':
            I0 = np.argmin(np.diff(data, axis=1), axis=1)
        elif method == 'max':
            I0 = np.argmax(data.values, axis=1)
        elif method == 'min':
            I0 = np.argmin(data.values, axis=1)
        elif method == 'absMaxSlope':
            I0 = np.argmax(np.diff(np.abs(data), axis=1), axis=1)
        elif method == 'absMax':
            I0 = np.argmax(np.abs(data.values), axis=1)
        else:
            print('No valid method for FROG determination!')
            return
        delOffset = delays[I0]

        if len(range_wl) == 2:
            indexRange = np.where((wl >= range_wl[0]) & (wl <= range_wl[1]))[0]
        else:
            indexList = []
            for i in range(len(range_wl)):
                indexList.append(np.where((wl >= range_wl[i][0]) & (wl <= range_wl[i][1]))[0])
                indexRange = np.concatenate(indexList)

        x = wl[indexRange]
        y = delOffset[indexRange]

        coefficients = np.polyfit(x, y, deg=degree)
        fit_function = np.poly1d(coefficients)

        plt.figure(2, figsize=(5.2/0.68, 5.2))
        # plt.figure(dpi=180)
        plt.clf()
        plt.plot(delOffset, wl, 'k', label=r'FROG trace', lw=0.5)
        plt.plot(delOffset[indexRange], wl[indexRange], '.r', label=r'fit range', markersize=2)
        plt.plot(fit_function(fit_wl), fit_wl, '-b', label=r'fit')
        plt.axis([np.min(delays)-1.5, np.max(delays)-1, np.min(wl), np.max(wl)])
        plt.xlabel(r'delay (ps)')
        plt.ylabel(r'wavelength (nm)')
        plt.grid(True)
        plt.legend(loc='best')
        plt.title('Fitted Dispersion correction function')

    else:
        if not os.path.isfile(file):
            print('Entered FROG file does not exist!')
        else:
            file_loaded = h5.File(file, 'r')
            fit_function_read = file_loaded["fit_function"]
            fit_function_coefficients = fit_function_read[()]
            fit_function_loaded = np.poly1d(fit_function_coefficients)

            plt.figure(2, figsize=(5.2/0.68, 5.2))
            # plt.figure(dpi=180)
            plt.clf()
            plt.plot(fit_function_loaded(fit_wl), fit_wl, '-b', label=r'fit')
            plt.axis([np.min(delays)-1.5, np.max(delays)-1, np.min(wl), np.max(wl)])
            plt.xlabel(r'delay (ps)')
            plt.ylabel(r'wavelength (nm)')
            plt.grid(True)
            plt.legend(loc='upper left')
            plt.title('Fit of dispersion correction')

            fit_function = fit_function_loaded
            print('Dispersion correction loaded:', file)

    scan["fit_function"] = fit_function

    return scan


def dispersion_corr(scan):
    """
    Interpolates the fit function on the data, saves the FROG-corrected data, and plots it.

    Parameters
    ----------
    scan : dict
        A dictionary containing the following keys:
        - "delay_unique" (array-like): Array of unique delay values.
        - "wavelength" (array-like): Array of wavelength values.
        - "data" (array-like): 2D array of data.
        - "fit_function" (callable): Fit function for dispersion correction.
        - "c_min" (float): Minimum value for color scale.
        - "c_max" (float): Maximum value for color scale.
        - "probe_method" (str): Probe method for color bar label.

    Returns
    ----------
    dict
        Updated scan dictionary with the "dispersion_data" key.
    """

    delays = scan["delay_unique"]
    wl = scan["wavelength"]
    data = pd.DataFrame(scan["data"])
    fit_function = scan["fit_function"]

    data_dispersion_corrected = np.zeros((len(delays), len(wl)))

    for i in range(len(wl)):
        interp_func = interp1d(delays, data.iloc[i, :], bounds_error=False, fill_value=0.0)
        data_dispersion_corrected[:, i] = interp_func(delays + fit_function(wl[i]))

    scan["dispersion_data"] = data_dispersion_corrected.T

    X, Y = np.meshgrid(scan["delay_unique"], scan["wavelength"])

    plt.figure(3, figsize=(5.2/0.68, 5.2))
    plt.gcf().clear()
    plt.pcolormesh(X, Y, 100*scan["dispersion_data"],
                   cmap=colors.fireice(), vmin=100*scan["c_min"], vmax=100*scan["c_max"],
                   shading='nearest')

    plt.title('Dispersion corrected data')
    plt.xlabel(r'delay (ps)')
    plt.ylabel(r'wavelength (nm)')

    cb = plt.colorbar()

    if scan["probe_method"] == "transmission":
        cb.ax.set_title(r'$\Delta T/T$ (%)')
    else:
        cb.ax.set_title(r'$\Delta R/R$ (%)')

    return scan


def save_time_explicit_format(filename, delays, wavelengths, data, comment1, comment2):
    """
    Save data in time-explicit format to a .ascii file.

    Parameters:
    - filename (str): Name of the file to save the data with .ascii extension.
    - delays (list): List of time delays.
    - wavelengths (list): List of wavelengths.
    - data (list or numpy array): 2D list or numpy array containing the spectroscopy data.
    - comment1 (str): Heading line 1 - A comment for the first line in the file.
    - comment2 (str): Heading line 2 - A comment for the second line in the file.

    Example:
    save_time_explicit_format("output_time_explicit", [1.0, 2.0, 3.0], [500, 600, 700],
                              [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
                              "Experiment Title", "Data recorded on 2024-02-03")
    """

    # Open the file in write mode with .ascii extension
    output_filename = "scan_export\\" + filename + ".ascii"
    with open(output_filename, 'w') as file:
        # Write comments
        file.write("#" + comment1 + "\n")
        file.write("#" + comment2 + "\n")

        # Write the format indicator
        file.write("Time explicit\n")

        # Write the number of time points and the delays
        file.write(f"Intervalnr {len(delays)}\n")
        file.write(" ".join(map(str, delays)) + "\n")

        # Write the data
        for i in range(len(wavelengths)):
            row_data = " ".join(map(str, data[i]))
            file.write(f"{wavelengths[i]} {row_data}\n")
        print("data exported to time explicit format in " + output_filename)
