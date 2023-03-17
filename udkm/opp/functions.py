import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import pandas as pd
import matplotlib


import udkm.tools.functions as tools
import udkm.tools.colors as colors

import lmfit as lm
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import os

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
    params["pump_angle"] = 8  # Wert aus dem Lab Book
    params["probe_method"] = "transmission"
    params["symmetric_colormap"] = True

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
        c_min = -1*np.max(np.abs(scan["data_averaged"]))
        c_max = 1*np.max(np.abs(scan["data_averaged"]))
    else:
        c_min = np.min(scan["data_averaged"])
        c_max = np.max(scan["data_averaged"])

    X, Y = np.meshgrid(scan["delay_raw_unique"], scan["wavelength"])

    plt.figure(1, figsize=(5.2/0.68, 5.2))
    plt.pcolormesh(X, Y, 100*scan["data_averaged"],
                   cmap=colors.fireice(), vmin=100*c_min, vmax=100*c_max, shading='nearest')
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

    if (params["bool_force_reload"] or not(os.path.isfile(params["scan_directory"]+params["id"]+".pickle"))):
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
        print("reload scan from: " + params["scan_directory"]+params["id"]+".pickle")
        scan = load_scan(params["date"], params["time"], params["scan_directory"])

    # overview plot of the data
    print("overview plot " + scan["name"] + " " + scan["date"] + " " +
          scan["time"]+" " + str(scan["T"])+"K " + str(scan["fluence"])+"mJ/cm^2 ")
    if scan["symmetric_colormap"]:
        c_min = -1*np.max(np.abs(scan["data_averaged"]))
        c_max = 1*np.max(np.abs(scan["data_averaged"]))
    else:
        c_min = np.min(scan["data_averaged"])
        c_max = np.max(scan["data_averaged"])

    X, Y = np.meshgrid(scan["delay_raw_unique"], scan["wavelength"])

    plt.figure(1, figsize=(5.2/0.68, 5.2))
    plt.pcolormesh(X, Y, 100*scan["data_averaged"],
                   cmap=colors.fireice(), vmin=100*c_min, vmax=100*c_max, shading='nearest')
    plt.xlabel(r'delay (ps)')
    plt.ylabel(r'wavelength (nm)')

    plt.title(scan["title_text"])

    cb = plt.colorbar()

    if scan["probe_method"] == "transmission":
        cb.ax.set_title(r'$\Delta T/T$ (%)')
    else:
        cb.ax.set_title(r'$\Delta R/R$ (%)')

    # %% clean data from errors where the spectrometer gave values smaller <=0
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

    entry_list = ["delay_min", "delay_max", "wl_min", "wl_max"]
    for entry in entry_list:
        if entry in params:
            scan[entry] = params[entry]

    # generate wl slices
    wl_slices = len(scan["slice_wl"])
    scan["wl_labels"] = list()
    if wl_slices > 0:
        scan["wl_slices"] = np.zeros((wl_slices, len(scan["delay_unique"])))

        for i, wl in enumerate(scan["slice_wl"]):

            wl_min = scan["slice_wl"][i] - scan["slice_wl_width"][i]
            wl_max = scan["slice_wl"][i] + scan["slice_wl_width"][i]
            t_min = np.min(scan["delay_unique"])
            t_max = np.max(scan["delay_unique"])
            _, _, data_cut, _, _ = tools.set_roi_2d(scan["delay_unique"], scan["wavelength"], scan["data"],
                                                    t_min, t_max, wl_min, wl_max)
            scan["wl_slices"][i, :] = np.mean(data_cut, axis=0)
            scan["wl_labels"].append(str(int(scan["slice_wl"][i])))

    else:
        scan["wl_slices"] = np.zeros((wl_slices, 1))
        scan["wl_slices"][0, :] = np.sum(scan["data"], axis=0)
        scan["wl_labels"].append("integral")

    # generate delay slices
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
        scan["delay_slices"][0, :] = np.sum(scan["data"], axis=1)
        scan["delay_labels"].append("integral")
    return scan


def save_scan(scan):
    '''Saves a scan dictionary to the scan_directory, given in scan, as python pickle'''
    pickle.dump(scan, open(scan["scan_directory"] + scan["id"] + ".pickle", "wb"))


def load_scan(date, time, scan_directory):
    '''Loads a scan from a python pickle into a scan dictionary'''
    path_string = scan_directory + tools.timestring(date)+"_"+tools.timestring(time)+".pickle"
    return pickle.load(open(path_string, "rb"))


def plot_overview(scan):
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
        plotted = ax1.plot(scan["delay_unique"], intensity, lw=1,  label=str(scan["wl_labels"][i]) + r'$\,$nm')
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
        c_min = -1*np.max(np.abs(scan["data_averaged"]))
        c_max = 1*np.max(np.abs(scan["data_averaged"]))
    else:
        c_min = np.min(scan["data_averaged"])
        c_max = np.max(scan["data_averaged"])

    X, Y = np.meshgrid(scan["delay_unique"], scan["wavelength"])
    plotted = ax3.pcolormesh(X, Y, 100*scan["data"], cmap=colors.fireice(), vmin=100*c_min, vmax=100*c_max)

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
        ax4.plot(intensity, scan["wavelength"],  lw=1,  label=str(scan["delay_labels"][i]) + r'$\,$ps')
        if not(scan["delay_labels"][i] == "integral"):
            ax3.axvline(x=scan["slice_delay"][i], lw=1, ls='--')

    ax4.axvline(x=0, ls="--", lw=0.5, color=colors.grey_3)

    ax4.set_ylim([scan["wl_min"], scan["wl_max"]])
    ax4.set_ylabel(r'$\mathrm{\lambda}$ (nm)')

    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")

    ax4.legend(loc=1,  bbox_to_anchor=(1.05, 1.5))
    ax4.axvline(x=0, lw=1, ls='--', color='grey')

    fig.suptitle(scan["title_text"])
