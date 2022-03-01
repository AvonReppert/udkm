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
scan = load_overview_data(params)

# if "fluence" in params:
#     scan["fluence"] = params["fluence"]
# else:
scan["fluence"] = tools.calc_fluence(params["power"], params["fwhm_x"], params["fwhm_y"],
                                     params["pump_angle"], params["rep_rate"])

moke.plot_overview(scan, params)

# Save a dictionary into a pickle file.
# %%

scan1 = moke.load_scan(20211119, 92027, "scan_export/")
# # %%
# plt.figure(figsize=(5.2, 5.2/0.68))
# gs = gridspec.GridSpec(3, 1, wspace=0, hspace=0)
# ax1 = plt.subplot(gs[0])
# ax2 = plt.subplot(gs[1])
# ax3 = plt.subplot(gs[2])

# ax1.plot(scan["delay"], scan["field_up"], label="field_up", color=colors.red_1, lw=2)
# ax1.plot(scan["delay"], scan["field_down"], label="field_down", color=colors.blue_1, lw=2)
# ax2.plot(scan["delay"], scan["sum"], label="sum", color=colors.orange_2, lw=2)
# ax3.plot(scan["delay"], scan["moke"], label="moke", color=colors.grey_1, lw=2)

# ax1.set_ylabel('single signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+/-}   = I_{+/-}^{1} - I_{+/-}^{0}}$ ')
# ax2.set_ylabel('sum signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+}\,\, +\,\, S_{-}}$ ')
# ax3.set_ylabel('MOKE signal ($\,\mathrm{V}$) \n $\mathrm{S_{+} \,\,-\,\, S_{-}}$')
# ax3.set_xlabel('time (ps)')

# if "t_min" in params:
#     t_min = params["t_min"]
# else:
#     t_min = min(scan["delay"])

# if "t_max" in params:
#     t_max = params["t_max"]
# else:
#     t_max = max(scan["delay"])


# for ax in [ax1, ax2, ax3]:
#     # ax.set_xticks(np.arange(0,1100,100))
#     ax.set_xlim(t_min, t_max)
#     ax.legend(loc=4)
#     ax.axhline(y=0, ls="--", color="grey")


# ax1.xaxis.set_ticks_position('top')
# ax2.set_xticklabels([])

# # title = datepath+ "    "+str(powers[line])+ " mW    " + str(voltages[line]) +" V"
# title_text = params["sample"] + "   " + params["date"]+" "+params["time"] + "  " + \
#     str(np.round(scan["fluence"], 1)) + r"$\mathrm{\,\frac{mJ}{\,cm^2}}$" + "  " + str(params["voltage"]) + r"$\,$V"

# scan["title_text"] = params["title_text"] = title_text
# ax1.set_title(scan["title_text"], pad=13)

# if params["bool_save_plot"]:
#     plt.savefig(params["plot_path"] + params["id"] + ".png", dpi=150)


# # def load_data_reflectivity(data_path, date, time, voltage, col_to_plot):
# #    file = data_path+str(date)+"_"+tools.timestring(time)+"/Fluence/-1.0/"+str(voltage)+"/overviewData.txt"
# #    data = np.genfromtxt(file, comments="#")
# #    return(data[:, 0], data[:, col_to_plot])
# #
