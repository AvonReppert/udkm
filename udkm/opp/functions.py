
import pickle as pickle
import shutil
import zipfile as zipfile

import numpy as np

import lmfit as lm
import udkm.tools.functions as tools
import matplotlib

import pandas as pd
import udkm.tools.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import os

teststring = "Successfully loaded udkm.tools.functions"


def get_scan_parameter(parameter_file_name, line):
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

    params["prefix"] = "Spectro1_"

    params["id"] = params["date"] + "_" + params["time"]

    params["measurements"] = len(param_file[entry])

    params["data_directory"] = "data\\"

    params["filename_pump"] = params["prefix"] + "Pump.txt"
    params["filename_probe"] = params["prefix"] + "Probe.txt"
    params["filename_averaged"] = params["prefix"] + "PumpProbe_AveragedData.txt"

    return params


def load_data(params):
    scan = {}

    scan["date"] = params["date"]
    scan["time"] = params["time"]
    scan["id"] = params["id"]
    scan["filename_pump"] = params["filename_pump"]
    scan["filename_probe"] = params["filename_probe"]
    scan["filename_averaged"] = params["filename_averaged"]

    scan["path"] = params["data_directory"] + "\\" + scan["id"] + "\\"
    print("load pump data from file " + scan["path"]+scan["filename_pump"]+"\n")

    scan["rep_rate"] = params["rep_rate"]
    scan["pump_angle"] = params["pump_angle"]

    scan["data_averaged"] = np.genfromtxt(scan["path"]+scan["filename_averaged"], comments="#")
    scan["data_pump"] = np.genfromtxt(scan["path"]+scan["filename_pump"], comments="#")
    scan["data_probe"] = np.genfromtxt(scan["path"]+scan["filename_probe"], comments="#")

    scan["wavelength"] = scan["data_averaged"][0, 1:]
    scan["time"] = scan["data_averaged"][1:, 0]
    scan["data_averaged"] = scan["data_averaged"][1:, 1:]

    # entry_list = ["x_min", "x_max", "y_min", "y_max", "name", "power", "rep_rate", "angle", "plot_logarithmic",
    #               "pixelsize_x", "pixelsize_y"]
    # for entry in entry_list:
    #     if entry in params:
    #         scan[entry] = params[entry]

    # %% #Overview plot to define the ROI

    # if not("x_min" in params):
    #     scan["x_min"] = 0
    # if not("x_max" in params):
    #     scan["x_max"] = len(scan["pixel_x"])*scan["pixelsize_x"]

    # if not("y_min" in params):
    #     scan["y_min"] = 0
    # if not("y_max" in params):
    #     scan["y_max"] = len(scan["pixel_y"])*scan["pixelsize_y"]

    X, Y = np.meshgrid(scan["time"], scan["wavelength"])

    plt.figure(1)
    if params["plot_logarithmic"]:
        plt.pcolormesh(X, Y, scan["data_averaged"], cmap=colors.fireice(),
                       norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.max(scan["data_averaged"])))
    else:
        plt.pcolormesh(X, Y, np.transpose(scan["data_averaged"]), cmap=colors.fireice(), vmin=np.min(scan["data_averaged"]),
                       vmax=np.max(scan["data_averaged"]), shading='nearest')
    #plt.axis([0, X.max(), 0, Y.max()])
    plt.xlabel(r'delay (ps)')
    plt.ylabel(r'wavelength (nm)')
    #plt.title(scan["date"] + "  " + scan["time"] + "  " + scan["name"])
    cb = plt.colorbar()
    cb.ax.set_title('$rel. change$ (%)')

    # scan["x_roi"], scan["y_roi"], scan["data_roi"], scan["x_integral"], scan["y_integral"] = tools.set_roi_2d(
    #     scan["distances_x"][1:], scan["distances_y"][1:], scan["data"], scan["x_min"], scan["x_max"],
    #     scan["y_min"], scan["y_max"])

    # scan["distance_x_cut"] = scan["x_roi"]-scan["x_min"]
    # scan["distance_y_cut"] = scan["y_roi"]-scan["y_min"]

    # scan["index_max_y"] = tools.find(scan["y_integral"], np.max(scan["y_integral"]))
    # scan["slice_y"] = scan["data_roi"][scan["index_max_y"], :]

    # scan["index_max_x"] = tools.find(scan["x_integral"], np.max(scan["x_integral"]))
    # scan["slice_x"] = scan["data_roi"][:, scan["index_max_x"]]

    # # %% Fitting the resulting params
    # model = lm.models.GaussianModel() + lm.models.LinearModel()
    # fit_params_x = lm.Parameters()
    # fit_params_y = lm.Parameters()

    # com_x, std_x, Ix = tools.calc_moments(scan["distance_x_cut"], scan["x_integral"])
    # com_y, std_y, Iy = tools.calc_moments(scan["distance_y_cut"], scan["y_integral"])

    # # Here you can set the initial values and possible boundaries on the fitting params
    # # Name       Value                 Vary     Min                   Max

    # fit_params_x.add_many(('center',    com_x,    True),
    #                       ('sigma',     std_x,     True),
    #                       ('amplitude', params["initial_amplitude_fit_x"], params["bool_fit_amplitude_x"]),
    #                       ('slope',     params["initial_slope_fit_x"],  params["bool_fit_slope_x"]),
    #                       ('intercept', params["initial_offset_fit_x"], params["bool_fit_intercept_x"]))

    # # Name       Value                 Vary     Min                   Max
    # fit_params_y.add_many(('center',    com_y,    True),
    #                       ('sigma',    std_y,     True),
    #                       ('amplitude', params["initial_amplitude_fit_y"], params["bool_fit_amplitude_y"]),
    #                       ('slope',     params["initial_slope_fit_y"],  params["bool_fit_slope_y"]),
    #                       ('intercept', params["initial_offset_fit_y"], params["bool_fit_intercept_y"]))

    # # Fitting takes place here
    # scan["fit_result_x"] = model.fit(scan["x_integral"]/np.max(scan["x_integral"]),
    #                                  fit_params_x, x=scan["distance_x_cut"])
    # scan["fit_result_y"] = model.fit(scan["y_integral"]/np.max(scan["y_integral"]),
    #                                  fit_params_y, x=scan["distance_y_cut"])

    # scan["fit_result_x_slice"] = model.fit(scan["slice_x"]/np.max(scan["slice_x"]),
    #                                        fit_params_x, x=scan["distance_y_cut"])
    # scan["fit_result_y_slice"] = model.fit(scan["slice_y"]/np.max(scan["slice_y"]),
    #                                        fit_params_y, x=scan["x_roi"]-scan["x_min"])

    # # Writing the results into the peaks dictionary takes place here
    # scan["fwhm_x"] = 2.35482 * scan["fit_result_x"].values["sigma"]  # in micron
    # scan["fwhm_y"] = 2.35482 * scan["fit_result_y"].values["sigma"]  # in micron

    # scan["fwhm_x_slice"] = 2.35482 * scan["fit_result_x_slice"].values["sigma"]  # in micron
    # scan["fwhm_y_slice"] = 2.35482 * scan["fit_result_y_slice"].values["sigma"]  # in micron

    # scan["fluence"] = tools.calc_fluence(scan["power"], scan["fwhm_x"],
    #                                      scan["fwhm_y"], scan["angle"], scan["rep_rate"])
    # scan["fluence_slice"] = tools.calc_fluence(
    #     scan["power"], scan["fwhm_x_slice"], scan["fwhm_y_slice"], scan["angle"], scan["rep_rate"])

    # print(str(int(round(scan["fwhm_x"]))) + " x " +
    #       str(int(round(scan["fwhm_y"]))) + str(" integral FWHM in microns"))
    # print("power_in  = " + str(np.round(scan["power"], 1)) + " mW (without chopper)\n->  F at " +
    #       str(int(scan["angle"]))+"° = " + str(np.round(scan["fluence"], 1)) + " mJ/cm^2\n")

    # print(str(int(round(scan["fwhm_x_slice"]))) + " x " +
    #       str(int(round(scan["fwhm_y_slice"]))) + str(" slice FWHM in microns"))

    # print("power_in  = " + str(np.round(scan["power"], 1)) + " mW (without chopper)\n->  F at " +
    #       str(int(scan["angle"]))+"° = " + str(np.round(scan["fluence_slice"], 1)) + " mJ/cm^2 for slice \n")

    # scan["plot_logarithmic"] = params["plot_logarithmic"]

    return scan
