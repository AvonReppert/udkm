# -*- coding: utf-8 -*-

import lmfit as lm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import udkm.tools.colors as colors


teststring = "Successfully loaded udkm.moke.static"


def load_data(file_name, title):
    scan = {}
    scan["title"] = title
    scan["data"] = np.genfromtxt(file_name, skip_header=14, delimiter=", ", skip_footer=1)

    scan["field"] = scan["data"][:, 0]*1000
    scan["rotation"] = scan["data"][:, 16]
    scan["ellipticity"] = scan["data"][:, 17]

    scan["select_up"] = np.diff(scan["field"]) >= 0
    scan["select_down"] = np.diff(scan["field"]) <= 0
    scan["unique_field"] = np.unique(scan["field"])

    scan["n_points"] = len(scan["unique_field"])

    scan["rotation_up"] = np.zeros(scan["n_points"])
    scan["rotation_down"] = np.zeros(scan["n_points"])

    scan["ellipticity_up"] = np.zeros(scan["n_points"])
    scan["ellipticity_down"] = np.zeros(scan["n_points"])

    for i, field_val in enumerate(scan["unique_field"]):
        if i < scan["n_points"] - 1:
            scan["rotation_up"][i] = np.mean(scan["rotation"][1:][scan["select_up"]
                                                                  & (field_val == scan["field"])[1:]])
            scan["rotation_down"][i] = np.mean(scan["rotation"][1:][scan["select_down"]
                                                                    & (field_val == scan["field"])[1:]])

            scan["ellipticity_up"][i] = np.mean(scan["ellipticity"][1:][scan["select_up"]
                                                                        & (field_val == scan["field"])[1:]])
            scan["ellipticity_down"][i] = np.mean(scan["ellipticity"][1:][scan["select_down"]
                                                                          & (field_val == scan["field"])[1:]])
        else:
            scan["rotation_up"][i] = scan["rotation_up"][i-2]
            scan["rotation_down"][i] = scan["rotation_down"][i-2]
            scan["ellipticity_up"][i] = scan["ellipticity_up"][i-2]
            scan["ellipticity_down"][i] = scan["ellipticity_down"][i-2]

            scan["rotation_up"][i-1] = scan["rotation_up"][i-2]
            scan["rotation_down"][i-1] = scan["rotation_down"][i-2]
            scan["ellipticity_up"][i-1] = scan["ellipticity_up"][i-2]
            scan["ellipticity_down"][i-1] = scan["ellipticity_down"][i-2]

        scan["rotation_up"][0] = scan["rotation_up"][1]
        scan["rotation_down"][0] = scan["rotation_down"][1]
        scan["ellipticity_up"][0] = scan["ellipticity_up"][1]
        scan["ellipticity_down"][0] = scan["ellipticity_down"][1]

    return scan


def plot_data_raw(scan):
    plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = ax1.twinx()

    ax1.plot(scan["field"], scan["rotation"], color=colors.blue_1)
    ax2.plot(scan["field"], scan["ellipticity"], color=colors.red_1)
    ax1.set_xlabel(r"applied field $\mu_\mathrm{0}H_\mathrm{ext}$ (mT)")
    ax1.set_ylabel(r"polarization rotation $\theta_\mathrm{s}$ (min)", color=colors.blue_1)
    ax2.set_ylabel(r"ellipticity change $\varepsilon_\mathrm{s}$ (min)", color=colors.red_1)
    plt.title(scan["title"] + "     raw data")


def plot_data_averaged(scan):
    plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0])
    ax2 = ax1.twinx()

    ax1.plot(scan["unique_field"], scan["rotation_up"], color=colors.blue_1, label="rotation_up")
    ax1.plot(scan["unique_field"], scan["rotation_down"], color=colors.blue_2, label="rotation_down")

    ax2.plot(scan["unique_field"], scan["ellipticity_up"], color=colors.red_1, label="ellipticity_up")
    ax2.plot(scan["unique_field"], scan["ellipticity_down"], color=colors.red_2, label="ellipticity_down")

    ax1.set_xlabel(r"applied field $\mu_\mathrm{0}H_\mathrm{ext}$ (mT)")
    ax1.set_ylabel(r"polarization rotation $\theta_\mathrm{s}$ (min)", color=colors.blue_1)
    ax2.set_ylabel(r"ellipticity change $\varepsilon_\mathrm{s}$ (min)", color=colors.red_1)
    plt.title(scan["title"] + "     average data")
    # ax1.legend(loc = 5)
    # ax2.legend(loc = 6)
    return ax1, ax2


def subtract_linear_background(scan, threshold, direction="up"):
    if threshold >= 0:
        scan["select_fit"] = scan["unique_field"] > threshold
    else:
        scan["select_fit"] = scan["unique_field"] < threshold

    model = lm.models.LinearModel()
    parameters = lm.Parameters()
    parameters.add_many(('slope', 0, True),
                        ('intercept', 0, True))

    if direction == "up":
        fit_rotation = model.fit(scan["rotation_up"][scan["select_fit"]], parameters,
                                 x=scan["unique_field"][scan["select_fit"]])

        scan["slope_rotation"] = fit_rotation.best_values["slope"]
        scan["offset_rotation"] = fit_rotation.best_values["intercept"]

        fit_ellipticity = model.fit(scan["ellipticity_up"][scan["select_fit"]], parameters,
                                    x=scan["unique_field"][scan["select_fit"]])

        scan["slope_ellipticity"] = fit_ellipticity.best_values["slope"]
        scan["offset_ellipticity"] = fit_ellipticity.best_values["intercept"]

    else:
        fit_rotation = model.fit(scan["rotation_down"][scan["select_fit"]], parameters,
                                 x=scan["unique_field"][scan["select_fit"]])

        scan["slope_rotation"] = fit_rotation.best_values["slope"]
        scan["offset_rotation"] = fit_rotation.best_values["intercept"]

        fit_ellipticity = model.fit(scan["ellipticity_down"][scan["select_fit"]], parameters,
                                    x=scan["unique_field"][scan["select_fit"]])

        scan["slope_ellipticity"] = fit_ellipticity.best_values["slope"]
        scan["offset_ellipticity"] = fit_ellipticity.best_values["intercept"]

    ax1, ax2 = plot_data_averaged(scan)
    ax1.plot(scan["unique_field"], scan["slope_rotation"]*scan["unique_field"] + scan["offset_rotation"],
             color=colors.blue_1, label="fit", ls="--")
    ax2.plot(scan["unique_field"], scan["slope_ellipticity"]*scan["unique_field"] + scan["offset_ellipticity"],
             color=colors.red_1, label="fit", ls="--")

    plt.title(scan["title"] + "     background correction")

    scan["rotation_up_corrected"] = scan["rotation_up"] - (scan["slope_rotation"]*scan["unique_field"]
                                                           + scan["offset_rotation"])
    scan["rotation_down_corrected"] = scan["rotation_down"] - (scan["slope_rotation"]*scan["unique_field"]
                                                               + scan["offset_rotation"])
    offset_1 = (np.mean(scan["rotation_up_corrected"]) + np.mean(scan["rotation_down_corrected"]))/2
    scan["rotation_up_corrected"] = scan["rotation_up_corrected"] - offset_1
    scan["rotation_down_corrected"] = scan["rotation_down_corrected"] - offset_1

    scan["ellipticity_up_corrected"] = scan["ellipticity_up"] - (scan["slope_ellipticity"]*scan["unique_field"]
                                                                 + scan["offset_ellipticity"])

    scan["ellipticity_down_corrected"] = scan["ellipticity_down"] - (scan["slope_ellipticity"]*scan["unique_field"]
                                                                     + scan["offset_ellipticity"])

    offset_2 = (np.mean(scan["ellipticity_up_corrected"]) + np.mean(scan["ellipticity_down_corrected"]))/2

    scan["ellipticity_up_corrected"] = scan["ellipticity_up_corrected"] - offset_2

    scan["ellipticity_down_corrected"] = scan["ellipticity_down_corrected"] - offset_2

    return scan


def plot_data_corrected(scan, plot_ellipticity=True):
    plt.figure()
    gs = gridspec.GridSpec(1, 1)
    ax1 = plt.subplot(gs[0])

    ax1.plot(scan["unique_field"], scan["rotation_up_corrected"], color=colors.blue_1, label="rotation_up")
    ax1.plot(scan["unique_field"], scan["rotation_down_corrected"], color=colors.blue_2, label="rotation_down")
    ax1.set_xlabel(r"applied field $\mu_\mathrm{0}H_\mathrm{ext}$ (mT)")
    ax1.set_ylabel(r"polarization rotation $\theta_\mathrm{s}$ (min)", color=colors.blue_1)

    axis_list = []
    axis_list.append(ax1)

    if plot_ellipticity:
        ax2 = ax1.twinx()
        ax2.plot(scan["unique_field"], scan["ellipticity_up_corrected"], color=colors.red_1, label="ellipticity_up")
        ax2.plot(scan["unique_field"], scan["ellipticity_down_corrected"], color=colors.red_2, label="ellipticity_down")
        ax2.set_ylabel(r"ellipticity change $\varepsilon_\mathrm{s}$ (min)", color=colors.red_1)
        axis_list.append(ax2)
    plt.title(scan["title"] + "     corrected data")
    ax1.grid("on")
    # ax1.legend(loc = 5)
    # ax2.legend(loc = 6)
    return axis_list
