import numpy as np
import matplotlib.pyplot as plt
import pickle as pickle
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


def plot_overview(scan):

    # integral_result_text = str(int(round(scan["fwhm_x"]))) + r'$\,\mathrm{\mu{}}$m' + " x " + str(
    #   int(round(scan["fwhm_y"]))) + r'$\,\mathrm{\mu{}}$m'

    X, Y = np.meshgrid(scan["delay_raw"], scan["wavelength"])

    delta_x = np.max(np.abs(scan["delay_raw"]))-np.min(np.abs(scan["delay_raw"]))
    delta_y = np.max(scan["wavelength"])-np.min(scan["wavelength"])

    mean_id = data_id.mean(axis=1)  # wird nach Reihen gemittelt - also für einen Stage Delay über die alle Wellenlängen
    mean_sig = data_sig.mean(axis=1)

    min_id = min(mean_id)
    min_sig = min(mean_sig)

    norm_id = -(mean_id / min_id)
    norm_sig = -(mean_sig / min_sig)

    plt.figure(2, figsize=(5.2, 5.2*np.max(scan["wavelength"])/np.max(np.abs(scan["delay_raw"]))), linewidth=2)
    gs = gridspec.GridSpec(2, 2,
                           width_ratios=[3, 1],
                           height_ratios=[1, 3],
                           wspace=0.0, hspace=0.0)

    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    # (ax1) Horizontal Profile ##
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")

    ax1.plot(scan["delay_raw"], scan["x_integral"]/np.max(scan["x_integral"]),
             '-', color=colors.grey_1, lw=2, label='integral')
    ax1.plot(scan["delay_raw"], scan["slice_y"]/np.max(scan["slice_y"]),
             '-', color=colors.blue_1, lw=1, label='slice')

    ax1.set_ylabel('I (a.u.)')
    ax1.set_xlabel(r'x ($\mathrm{\mu{}}$m)')
    ax1.set_ylim([0, 1.05])
    ax1.set_yticks(np.arange(0.25, 1.25, .25))

    # (ax 3) Colormap of the Profile #############
    if scan["plot_logarithmic"]:
        pl = ax3.pcolormesh(X, Y, 100*np.transpose(scan["data_averaged"]), cmap=colors.fireice(),
                            norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.max(100*np.transpose(scan["data_averaged"]))))
    else:
        pl = ax3.pcolormesh(X, Y, 100*np.transpose(scan["data_averaged"]),
                            vmin=0, vmax=np.max(100*np.transpose(scan["data_averaged"])))

    ax3.axis([0, scan["x_max"]-scan["x_min"], 0, scan["y_max"]-scan["y_min"]])
    ax3.set_xlabel(r'x ($\mathrm{\mu{}}$m)')
    ax3.set_ylabel(r'y ($\mathrm{\mu{}}$m)')

    ax3.axvline(x=scan["distance_x_cut"][scan["index_max_x"]], ls="--", color=colors.blue_1, lw=0.5)
    ax3.axhline(y=scan["distance_y_cut"][scan["index_max_y"]], ls="--", color=colors.blue_1, lw=0.5)

    # colorbar placement #############
    axins3 = inset_axes(ax3,
                        width="60%",  # width = 10% of parent_bbox width
                        height="3%",  # height : 50%                   )
                        loc=4)
    ax3.add_patch(Rectangle((0.35, 0.018), 0.65, 0.18, edgecolor="none",
                  facecolor="white", alpha=0.75, transform=ax3.transAxes))
    cbar = plt.colorbar(pl, cax=axins3, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)

    cbar.ax.set_title('$I$ (cts)', fontsize=10, pad=-2)

    axins3.xaxis.set_ticks_position("top")
    axins3.xaxis.set_label_position("top")
    cl = plt.getp(cbar.ax, 'xmajorticklabels')

    plt.setp(cl, color="black")

    ax2.axis('off')

    # Plot 4) Bottom Right Vertical Profile

    ax4.plot(scan["y_integral"]/np.max(scan["y_integral"]), scan["distance_y_cut"],
             '-', color=colors.grey_3, lw=2, label='integral')
    ax4.plot(scan["fit_result_y"].best_fit, scan["distance_y_cut"], '-',
             color=colors.grey_1, lw=1, label="fit " + integral_result_text)
    ax4.plot(scan["slice_x"]/np.max(scan["slice_x"]), scan["distance_y_cut"],
             '-', color=colors.blue_2, lw=1, label='slice')
    ax4.plot(scan["fit_result_x_slice"].best_fit, scan["distance_y_cut"],  '--',
             color=colors.blue_1, lw=1, label="fit " + slice_result_text)

    ax4.set_ylabel(r'y ($\mathrm{\mu{}}$m)')
    ax4.set_xlabel('I (a.u.)')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.legend(loc=(0.05, 1.05), frameon=False, fontsize=8, title=scan["name"])

    ax4.set_ylim(0, 1.05)
    ax4.set_ylim(0, delta_y)

    ax4.set_xticks([0, 1])
    ax1.set_yticks([0, 1])

    ax1.set_xlim([0, delta_x])
    ax4.set_ylim([0, delta_y])


plt.show()

# %% Clear and averaging Data


def clear_data(probe, pump, **bad_loops):

    probe = probe.drop(bad_loops)
    pump = pump.drop(bad_loops)

    probe = probe[~(probe == 0).any(axis=1)]
    pump = pump[~(pump == 0).any(axis=1)]
    # variable null
    clear_probe = probe[~probe.lt(0).all(axis=1)]
    clear_pump = pump[~pump.lt(0).all(axis=1)]

    return clear_probe, clear_pump


def averaging(clear_probe, clear_pump):

    u_delays = np.unique(clear_probe.iloc[:, 0])
    y = len(clear_probe.columns) - 1  # Spalte der Dealys noch rausrechnen, damit man nur die Steps der Wellenlänge hat

    avr_probe = np.zeros((len(u_delays), y))
    avr_pump = np.zeros((len(u_delays), y))

    clear_probe = clear_probe.to_numpy()
    clear_pump = clear_pump.to_numpy()

    for i in range(len(u_delays)):
        select_delays_probe = clear_probe[:, 0] == u_delays[i]
        select_delays_pump = clear_pump[:, 0] == u_delays[i]

        data_avr_probe = np.mean(clear_probe[select_delays_probe, 1:], axis=0)
        data_avr_pump = np.mean(clear_pump[select_delays_pump, 1:], axis=0)

        avr_probe[i, :] = data_avr_probe
        avr_pump[i, :] = data_avr_pump

        i += 1

    return avr_probe, avr_pump, u_delays
