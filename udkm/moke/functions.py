# -*- coding: utf-8 -*-

import numpy as np
import udkm.tools.helpers as h

import pandas as pd
import udkm.tools.functions as tools

import udkm.tools.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle

teststring = "Successfully loaded udkm.moke.functions"

# initialize some useful functions
t0 = 0    # Estimated value of t0
data_path = "data/"
export_path = "results/"


def load_data(data_path, date, time, voltage, col_to_plot):
    file = data_path+str(date)+"_"+tools.timestring(time)+"/Fluence/-1.0/"+str(voltage)+"/overviewData.txt"
    data = np.genfromtxt(file, comments="#")
    return(data[:, 0], data[:, col_to_plot])


def load_data_A(data_path, date, time, angle, col_to_plot):
    file = data_path+str(date)+"_"+tools.timestring(time)+"/Fluence/"+str(int(angle))+"/1.0"+"/overviewData.txt"
    data = np.genfromtxt(file, comments="#")
    return(data[:, 0], data[:, col_to_plot])


def load_data_B(data_path, date, time, angle, col_to_plot):
    file = data_path+str(date)+"_"+tools.timestring(time)+"/Fluence/"+str(angle)+"/1.0"+"/overviewData.txt"
    data = np.genfromtxt(file, comments="#")
    return(data[:, 0], data[:, col_to_plot])


def load_data_reflectivity(data_path, date, time, voltage, col_to_plot):
    file = data_path+str(date)+"_"+tools.timestring(time)+"/Fluence/-1.0/"+str(voltage)+"/overviewData.txt"
    data = np.genfromtxt(file, comments="#")
    return(data[:, 0], data[:, col_to_plot])


def load_data_hysteresis(data_path, date, time, name):
    file = data_path+str(date)+"_"+tools.timestring(time)+"/Fluence/Static/"+name+"_NoLaser.txt"
    data = np.genfromtxt(file, comments="#")
    return(data[:, 0], data[:, 1], data[:, 2])


def get_scan_parameter(parameter_file_name, line):
    params = {'line': line}
    param_file = pd.read_csv(parameter_file_name, delimiter="\t", header=0, comment="#")
    header = list(param_file.columns.values)

    for i, entry in enumerate(header):
        if entry == 'date' or entry == 'time':
            params[entry] = tools.timestring(int(param_file[entry][line]))
        else:
            params[entry] = param_file[entry][line]

    # set default parameters
    params["bool_t0_shift"] = False
    params["bool_save_plot"] = True
    params["t0_column_name"] = "moke"

    params["pump_angle"] = 0
    params["rep_rate"] = 1000
    params["id"] = params["date"] + "_" + params["time"]

    if not('fluence') in header:
        params['fluence'] = -1.0

    return params


def load_overview_data(params):
    data_path = params["date"]+"_"+params["time"]+"/Fluence/"+str(params["fluence"])+"/"+str(params["voltage"])+"/"
    prefix = "data/"
    file_name = "overviewData.txt"
    data = np.genfromtxt(prefix+data_path+file_name, comments="#")

    scan = {}
    scan["raw_delay"] = data[:, 0]
    scan["delay"] = data[:, 0]
    scan["sum"] = data[:, 1]
    scan["moke"] = data[:, 2]
    scan["field_up"] = data[:, 3]
    scan["field_down"] = data[:, 4]

    scan["date"] = params["date"]
    scan["time"] = params["time"]
    scan["sample"] = params["sample"]
    scan["voltage"] = params["voltage"]
    scan["id"] = params["id"]
    scan["scan_path"] = params["scan_path"]

    if params["bool_t0_shift"]:

        if "t0" in params:
            scan["delay"] = scan["raw_delay"]-params["t0"]
            scan["t0"] = params["t0"]
            print("t0 = " + str(scan["t0"]) + " ps")
        else:
            differences = np.abs(np.diff(scan[params["t0_column_name"]]))
            t0_index = tools.find(differences, np.max(differences))
            t0 = scan["raw_delay"][:-1][t0_index]
            scan["t0"] = t0
            print("t0 = " + str(scan["t0"]) + " ps")
            scan["delay"] = scan["raw_delay"]-t0

    scan["fluence"] = tools.calc_fluence(params["power"], params["fwhm_x"], params["fwhm_y"],
                                         params["pump_angle"], params["rep_rate"])

    title_text = scan["sample"] + "   " + scan["date"]+" "+scan["time"] + "  " + \
        str(np.round(scan["fluence"], 1)) + r"$\mathrm{\,\frac{mJ}{\,cm^2}}$" + "  " + \
        str(scan["voltage"]) + r"$\,$V"

    scan["title_text"] = params["title_text"] = title_text

    return scan


def plot_overview(scan, **params):

    plt.figure(figsize=(5.2, 5.2/0.68))
    gs = gridspec.GridSpec(3, 1, wspace=0, hspace=0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    ax1.plot(scan["delay"], scan["field_up"], label="field_up", color=colors.red_1, lw=2)
    ax1.plot(scan["delay"], scan["field_down"], label="field_down", color=colors.blue_1, lw=2)
    ax2.plot(scan["delay"], scan["sum"], label="sum", color=colors.orange_2, lw=2)
    ax3.plot(scan["delay"], scan["moke"], label="moke", color=colors.grey_1, lw=2)

    ax1.set_ylabel('single signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+/-}   = I_{+/-}^{1} - I_{+/-}^{0}}$ ')
    ax2.set_ylabel('sum signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+}\,\, +\,\, S_{-}}$ ')
    ax3.set_ylabel('MOKE signal ($\,\mathrm{V}$) \n $\mathrm{S_{+} \,\,-\,\, S_{-}}$')
    ax3.set_xlabel('time (ps)')

    if "t_min" in params:
        t_min = params["t_min"]
    else:
        t_min = min(scan["delay"])

    if "t_max" in params:
        t_max = params["t_max"]
    else:
        t_max = max(scan["delay"])

    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(t_min, t_max)
        ax.legend(loc=4)
        ax.axhline(y=0, ls="--", color="grey")

    ax1.xaxis.set_ticks_position('top')
    ax2.set_xticklabels([])

    ax1.set_title(scan["title_text"], pad=13)

    if "bool_save_plot" in params:
        if params["bool_save_plot"]:
            plt.savefig(params["plot_path"] + scan["id"] + ".png", dpi=150)


def save_scan(scan):
    pickle.dump(scan, open(scan["scan_path"] + scan["id"] + ".pickle", "wb"))


def load_scan(date, time, path):
    path_string = path + tools.timestring(date)+"_"+tools.timestring(time)+".pickle"
    return pickle.load(open(path_string, "rb"))
