import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
import pandas as pd


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
    params["pump_angle"] = 0
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


def load_data(params):
    '''Loads the data for a given parameter set into a scan dictionary and returns overview plot'''

    if (params["bool_force_reload"] or not(os.path.isfile(params["scan_directory"]+params["id"]+".pickle"))):
        print("load data from : " + params["data_directory"]+params["id"])
        # initialize scan dictionary
        scan = {}
        scan["date"] = params["date"]
        scan["time"] = params["time"]
        scan["id"] = params["id"]
        scan["name"] = params["name"]
        scan["parameters_missing"] = False

        scan["filename_pump"] = params["filename_pump"]
        scan["filename_probe"] = params["filename_probe"]
        scan["filename_averaged"] = params["filename_averaged"]

        scan["data_directory"] = params["data_directory"] + "\\" + scan["id"] + "\\"

        entry_list = ["power", "rep_rate", "pump_angle", "T",
                      "fwhm_x", "fwhm_y", "pump_wl", "probe_method", "scan_directory"]
        for entry in entry_list:
            if entry in params:
                scan[entry] = params[entry]
            else:
                print("Warning: " + str(entry) + " is missing in parameter file")
                scan["parameters_missing"] = True

        scan["fluence"] = tools.calc_fluence(scan["power"], scan["fwhm_x"],
                                             scan["fwhm_y"], scan["pump_angle"], scan["rep_rate"])

        if scan["parameters_missing"]:
            scan["title_text"] = scan["id"]
        else:
            scan["title_text"] = scan["name"] + "  " + scan["date"] + " " + scan["time"] + " " + \
                str(scan["T"]) + r"$\,$K " + " " + str(scan["pump_wl"]) + r"$\,$nm  " + str(np.round(scan["fluence"], 1)) + \
                r"$\,\mathrm{\frac{mJ}{\,cm^2}}$ "

        # load data into scan dictionary
        scan["data_averaged"] = np.genfromtxt(scan["data_directory"]+scan["filename_averaged"], comments="#")
        scan["data_pump"] = np.genfromtxt(scan["data_directory"]+scan["filename_pump"], comments="#")
        scan["data_probe"] = np.genfromtxt(scan["data_directory"]+scan["filename_probe"], comments="#")

        scan["wavelength"] = scan["data_averaged"][0, 1:]
        scan["delay_raw"] = scan["data_averaged"][1:, 0]
        scan["data_averaged"] = scan["data_averaged"][1:, 1:]
    else:
        print("reload scan from: " + params["scan_directory"]+params["id"]+".pickle")
        scan = load_scan(params["date"], params["time"], params["scan_directory"])

    # overview plot of the data
    print("overview plot " + scan["name"] + " " + scan["date"] + " " +
          scan["time"]+" " + str(scan["T"])+"K " + str(scan["fluence"])+"mJ/cm^2 ")
    if params["symmetric_colormap"]:
        c_min = -1*np.max(np.abs(scan["data_averaged"]))
        c_max = 1*np.max(np.abs(scan["data_averaged"]))
    else:
        c_min = np.min(scan["data_averaged"])
        c_max = np.max(scan["data_averaged"])

    X, Y = np.meshgrid(scan["delay_raw"], scan["wavelength"])

    plt.figure(1, figsize=(5.2/0.68, 5.2))
    plt.pcolormesh(X, Y, 100*np.transpose(scan["data_averaged"]),
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


def save_scan(scan):
    '''Saves a scan dictionary to the scan_directory, given in scan, as python pickle'''
    pickle.dump(scan, open(scan["scan_directory"] + scan["id"] + ".pickle", "wb"))


def load_scan(date, time, scan_directory):
    '''Loads a scan from a python pickle into a scan dictionary'''
    path_string = scan_directory + tools.timestring(date)+"_"+tools.timestring(time)+".pickle"
    return pickle.load(open(path_string, "rb"))
