# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import dill as pickle
import pandas as pd
import os as os


import udkm.tools.functions as tools
import udkm.tools.colors as colors

import lmfit as lm
import matplotlib.colors as matplotlib_colors
from matplotlib.patches import Rectangle

teststring = "Successfully loaded udkm.beamprofile.functions"


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
    params["angle"] = 0
    params["rep_rate"] = 1000
    params["suffix"] = "_image"

    params["initial_amplitude_fit_x"] = 200
    params["initial_slope_fit_x"] = 0
    params["initial_offset_fit_x"] = 0

    params["bool_fit_amplitude_x"] = True
    params["bool_fit_slope_x"] = False
    params["bool_fit_intercept_x"] = True

    params["initial_amplitude_fit_y"] = 200
    params["initial_slope_fit_y"] = 0
    params["initial_offset_fit_y"] = 0

    params["bool_fit_amplitude_y"] = True
    params["bool_fit_slope_y"] = False
    params["bool_fit_intercept_y"] = True

    params["id"] = params["date"] + "_" + params["time"]

    params["measurements"] = len(param_file[entry])

    params["scan_directory"] = "scan_export\\"
    params["data_directory"] = "data\\"

    params["bool_force_reload"] = False

    return params


def load_data(params):
    '''Loads the data for a given parameter set into a scan dictionary and returns overview plot.
       Makes a peak fit for the slice through max intensity and the integral of the picture.'''
    if (params["bool_force_reload"] or not (os.path.isfile(params["scan_directory"]+params["id"]+".pickle"))):
        print("load data from : " + params["data_directory"]+params["id"])
        scan = {}
        scan["date"] = params["date"]
        scan["time"] = params["time"]
        scan["id"] = params["id"]
        scan["filename"] = scan["date"] + "_" + scan["time"] + params["suffix"] + ".txt"

        scan["path"] = params["data_directory"] + "\\" + scan["date"] + "\\"

        print("load data from file " + scan["path"]+scan["filename"]+"\n")

        scan["rep_rate"] = params["rep_rate"]
        scan["angle"] = params["angle"]

        scan["data"] = np.genfromtxt(scan["path"]+scan["filename"])
        scan["pixel_x"] = np.arange(0, np.shape(scan["data"])[1]+1, 1)
        scan["pixel_y"] = np.arange(0, np.shape(scan["data"])[0]+1, 1)

        scan["distances_x"] = scan["pixel_x"]*params["pixelsize_x"]  # pixel in µm
        scan["distances_y"] = scan["pixel_y"]*params["pixelsize_y"]  # pixel in µm

        entry_list = ["x_min", "x_max", "y_min", "y_max", "name", "power", "rep_rate", "angle", "plot_logarithmic",
                      "pixelsize_x", "pixelsize_y", "scan_directory", "id"]
        for entry in entry_list:
            if entry in params:
                scan[entry] = params[entry]

        # %% #Overview plot to define the ROI

        if not ("x_min" in params):
            scan["x_min"] = 0
        if not ("x_max" in params):
            scan["x_max"] = len(scan["pixel_x"])*scan["pixelsize_x"]

        if not ("y_min" in params):
            scan["y_min"] = 0
        if not ("y_max" in params):
            scan["y_max"] = len(scan["pixel_y"])*scan["pixelsize_y"]

        X, Y = np.meshgrid(scan["distances_x"], scan["distances_y"])
        plt.figure(1, figsize=(5.2, 5.2*(scan["y_max"]-scan["y_min"])/(scan["x_max"]-scan["x_min"])))
        if params["plot_logarithmic"]:
            plt.pcolormesh(X, Y, scan["data"], cmap=colors.fireice(),
                           norm=matplotlib_colors.LogNorm(vmin=1, vmax=np.max(scan["data"])))
        else:
            plt.pcolormesh(X, Y, scan["data"], cmap=colors.fireice(), vmin=0, vmax=np.max(scan["data"]))
    
        plt.axis([0, X.max(), 0, Y.max()])
        plt.xlabel(r'x ($\mathrm{\mu{}}$m)')
        plt.ylabel(r'y ($\mathrm{\mu{}}$m)')
        plt.title(scan["date"] + "  " + scan["time"] + "  " + scan["name"])
        cb = plt.colorbar()
        cb.ax.set_title('$I$ (cts)')

        scan["x_roi"], scan["y_roi"], scan["data_roi"], scan["x_integral"], scan["y_integral"] = tools.set_roi_2d(
            scan["distances_x"][1:], scan["distances_y"][1:], scan["data"], scan["x_min"], scan["x_max"],
            scan["y_min"], scan["y_max"])

        scan["distance_x_cut"] = scan["x_roi"]-scan["x_min"]
        scan["distance_y_cut"] = scan["y_roi"]-scan["y_min"]

        scan["index_max_y"] = tools.find(scan["y_integral"], np.max(scan["y_integral"]))
        scan["slice_y"] = scan["data_roi"][scan["index_max_y"], :]

        scan["index_max_x"] = tools.find(scan["x_integral"], np.max(scan["x_integral"]))
        scan["slice_x"] = scan["data_roi"][:, scan["index_max_x"]]

        # %% Fitting the resulting params
        model = lm.models.GaussianModel() + lm.models.LinearModel()
        fit_params_x = lm.Parameters()
        fit_params_y = lm.Parameters()

        com_x, std_x, Ix = tools.calc_moments(scan["distance_x_cut"], scan["x_integral"])
        com_y, std_y, Iy = tools.calc_moments(scan["distance_y_cut"], scan["y_integral"])

        # Here you can set the initial values and possible boundaries on the fitting params
        # Name       Value                 Vary     Min                   Max

        fit_params_x.add_many(('center',    com_x,    True),
                              ('sigma',     std_x,     True),
                              ('amplitude', params["initial_amplitude_fit_x"], params["bool_fit_amplitude_x"]),
                              ('slope',     params["initial_slope_fit_x"],  params["bool_fit_slope_x"]),
                              ('intercept', params["initial_offset_fit_x"], params["bool_fit_intercept_x"]))

        # Name       Value                 Vary     Min                   Max
        fit_params_y.add_many(('center',    com_y,    True),
                              ('sigma',    std_y,     True),
                              ('amplitude', params["initial_amplitude_fit_y"], params["bool_fit_amplitude_y"]),
                              ('slope',     params["initial_slope_fit_y"],  params["bool_fit_slope_y"]),
                              ('intercept', params["initial_offset_fit_y"], params["bool_fit_intercept_y"]))

        # Fitting takes place here
        scan["fit_result_x"] = model.fit(scan["x_integral"]/np.max(scan["x_integral"]),
                                         fit_params_x, x=scan["distance_x_cut"])
        scan["fit_result_y"] = model.fit(scan["y_integral"]/np.max(scan["y_integral"]),
                                         fit_params_y, x=scan["distance_y_cut"])

        scan["fit_result_x_slice"] = model.fit(scan["slice_x"]/np.max(scan["slice_x"]),
                                               fit_params_x, x=scan["distance_y_cut"])
        scan["fit_result_y_slice"] = model.fit(scan["slice_y"]/np.max(scan["slice_y"]),
                                               fit_params_y, x=scan["x_roi"]-scan["x_min"])

        # Writing the results into the peaks dictionary takes place here
        scan["fwhm_x"] = 2.35482 * scan["fit_result_x"].values["sigma"]  # in micron
        scan["fwhm_y"] = 2.35482 * scan["fit_result_y"].values["sigma"]  # in micron

        scan["fwhm_x_slice"] = 2.35482 * scan["fit_result_x_slice"].values["sigma"]  # in micron
        scan["fwhm_y_slice"] = 2.35482 * scan["fit_result_y_slice"].values["sigma"]  # in micron

        scan["fluence"] = tools.calc_fluence(scan["power"], scan["fwhm_x"],
                                             scan["fwhm_y"], scan["angle"], scan["rep_rate"])
        scan["fluence_slice"] = tools.calc_fluence(
            scan["power"], scan["fwhm_x_slice"], scan["fwhm_y_slice"], scan["angle"], scan["rep_rate"])

        print(str(int(round(scan["fwhm_x"]))) + " x " +
              str(int(round(scan["fwhm_y"]))) + str(" integral FWHM in microns"))
        print("power_in  = " + str(np.round(scan["power"], 1)) + " mW (without chopper)\n->  F at " +
              str(int(scan["angle"]))+"° = " + str(np.round(scan["fluence"], 1)) + " mJ/cm^2\n")

        print(str(int(round(scan["fwhm_x_slice"]))) + " x " +
              str(int(round(scan["fwhm_y_slice"]))) + str(" slice FWHM in microns"))

        print("power_in  = " + str(np.round(scan["power"], 1)) + " mW (without chopper)\n->  F at " +
              str(int(scan["angle"]))+"° = " + str(np.round(scan["fluence_slice"], 1)) + " mJ/cm^2 for slice \n")

        scan["plot_logarithmic"] = params["plot_logarithmic"]
    else:
        print("reload scan from: " + params["scan_directory"]+params["id"]+".pickle")
        scan = load_scan(params["date"], params["time"], params["scan_directory"])

    return scan

    # %% Plotting the results in the ROI


def plot_overview(scan):
    ''' returns an overview plot of the beamprofile for a given ROI'''

    integral_result_text = str(
        int(round(scan["fwhm_x"]))) + r'$\,\mathrm{\mu{}}$m' + " x " + str(int(round(scan["fwhm_y"]))) + r'$\,\mathrm{\mu{}}$m'
    slice_result_text = str(int(round(scan["fwhm_x_slice"]))) + \
        r'$\,\mathrm{\mu{}}$m' + " x " + str(int(round(scan["fwhm_y_slice"]))) + r'$\,\mathrm{\mu{}}$m'

    x_grid, y_grid = np.meshgrid(scan["distance_x_cut"], scan["distance_y_cut"])

    delta_x = scan["x_max"]-scan["x_min"]
    delta_y = scan["y_max"]-scan["y_min"]

    plt.figure(2, figsize=(5.2, 5.2*delta_y/delta_x), linewidth=2)
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

    ax1.plot(scan["distance_x_cut"], scan["x_integral"]/np.max(scan["x_integral"]),
             '-', color=colors.grey_3, lw=2, label='integral')
    ax1.plot(scan["distance_x_cut"],  scan["fit_result_x"].best_fit, '-',
             color=colors.grey_1, lw=1, label="fit " + integral_result_text)
    ax1.plot(scan["distance_x_cut"], scan["slice_y"]/np.max(scan["slice_y"]),
             '-', color=colors.blue_2, lw=1, label='slice')
    ax1.plot(scan["distance_x_cut"],  scan["fit_result_y_slice"].best_fit, '--',
             color=colors.blue_1, lw=1, label="fit" + slice_result_text)

    ax1.set_ylabel('I (a.u.)')
    ax1.set_xlabel(r'x ($\mathrm{\mu{}}$m)')
    ax1.set_ylim([0, 1.05])
    ax1.set_yticks(np.arange(0.25, 1.25, .25))

    # (ax 3) Colormap of the Profile #############
    if scan["plot_logarithmic"]:
        pl = ax3.pcolormesh(x_grid, y_grid, scan["data_roi"], cmap=colors.fireice(),
                            norm=matplotlib_colors.LogNorm(vmin=1, vmax=np.max(scan["data_roi"])))
    else:
        pl = ax3.pcolormesh(x_grid, y_grid, scan["data_roi"], cmap=colors.fireice(), vmin=0, vmax=np.max(scan["data"]))

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
    ax4.legend(loc=(0.05, 1.05), frameon=False, fontsize=8, title=scan["name"]+u"\n"+scan["id"], title_fontsize=8)

    ax4.set_ylim(0, 1.05)
    ax4.set_ylim(0, delta_y)

    ax4.set_xticks([0, 1])
    ax1.set_yticks([0, 1])

    ax1.set_xlim([0, delta_x])
    ax4.set_ylim([0, delta_y])


plt.show()


def save_scan(scan):
    '''Saves a scan dictionary to the scan_directory, given in scan, as python pickle'''
    pickle.dump(scan, open(scan["scan_directory"] + scan["id"] + ".pickle", "wb"))


def load_scan(date, time, scan_directory):
    '''Loads a scan from a python pickle into a scan dictionary'''
    path_string = scan_directory + tools.timestring(date)+"_"+tools.timestring(time)+".pickle"
    return pickle.load(open(path_string, "rb"))
