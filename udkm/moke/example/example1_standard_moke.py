# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:58:07 2022

@author: Aleks
"""
import pickle
import udkm.moke.functions as moke
import udkm.tools.functions as tools
import numpy as np
import matplotlib.pyplot as plt
import udkm.tools.colors as colors
import matplotlib.gridspec as gridspec

plt.style.use("udkm_base.mplstyle")
# print(moke.teststring)

parameter_file_name = "parameters/parameters_example_1.txt"


line = 0


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

    return scan


params = moke.get_scan_parameter(parameter_file_name, line)
params["bool_t0_shift"] = False
params["bool_save_plot"] = True
params["t0_column_name"] = "moke"
params["plot_path"] = "plot_overview//"
params["scan_path"] = "scan_export//"
scan = moke.load_overview_data(params)

# if "fluence" in params:
#     scan["fluence"] = params["fluence"]
# else:
scan["fluence"] = tools.calc_fluence(params["power"], params["fwhm_x"], params["fwhm_y"],
                                     params["pump_angle"], params["rep_rate"])

moke.plot_overview(scan, params)

# Save a dictionary into a pickle file.
# %%

scan1 = moke.load_scan(20211119, 92027, "scan_export/")
