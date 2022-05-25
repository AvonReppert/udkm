# -*- coding: utf-8 -*-

import numpy as np
import os as os
import pandas as pd
import lmfit as lm
import udkm.tools.functions as tools
import matplotlib


import udkm.tools.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import pickle

teststring = "Successfully loaded udkm.moke.functions"

# initialize some useful functions
t0 = 0    # Estimated value of t0
data_path = "data/"
export_path = "results/"

''' Some legacy functions
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
'''


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
    params["t0_column_name"] = "moke"

    params["pump_angle"] = 0
    params["rep_rate"] = 1000
    params["id"] = params["date"] + "_" + params["time"]
    params["measurements"] = len(param_file[entry])

    if not('fluence') in header:
        params['fluence'] = -1.0

    return params


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

    ax1.set_ylabel(r'single signal  ($\,\mathrm{V}$)' + "\n" + r'$\mathrm{S_{+/-}   = I_{+/-}^{1} - I_{+/-}^{0}}$ ')
    ax2.set_ylabel(r'sum signal  ($\,\mathrm{V}$)' + "\n" + r'$\mathrm{S_{+}\,\, +\,\, S_{-}}$ ')
    ax3.set_ylabel(r'MOKE signal ($\,\mathrm{V}$)' + "\n" + r'$\mathrm{S_{+} \,\,-\,\, S_{-}}$')
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


def save_scan(scan):
    pickle.dump(scan, open(scan["scan_path"] + scan["id"] + ".pickle", "wb"))


def load_scan(date, time, path):
    path_string = path + tools.timestring(date)+"_"+tools.timestring(time)+".pickle"
    return pickle.load(open(path_string, "rb"))


def load_data(params):
    if not("raw_data_directory" in params):
        params["raw_data_directory"] = "data\\"
    if not("overview_data_directory" in params):
        params["overview_data_directory"] = "data_overview\\"

    overview_file = params["overview_data_directory"] + "overview_data_"+params["date"] + "_" +\
        params["time"] + "_" + str(params["fluence"]) + "_" + str(params["voltage"]) + ".txt"

    if os.path.isfile(overview_file):
        print("read in overview data")
        data = np.genfromtxt(overview_file, comments="#", skip_footer=1)
    else:
        print("exporting data to overview file")
        export_raw_data(params)
        data = np.genfromtxt(overview_file, comments="#", skip_footer=1)

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
            print("t0 = " + str(np.round(scan["t0"], 2)) + " ps")
            scan["delay"] = scan["raw_delay"]-t0

    scan["fluence"] = tools.calc_fluence(params["power"], params["fwhm_x"], params["fwhm_y"],
                                         params["pump_angle"], params["rep_rate"])

    title_text = scan["sample"] + "   " + scan["date"]+" "+scan["time"] + "  " + \
        str(np.round(scan["fluence"], 1)) + r"$\mathrm{\,\frac{mJ}{\,cm^2}}$" + "  " + \
        str(scan["voltage"]) + r"$\,$V"

    scan["title_text"] = params["title_text"] = title_text
    return scan


def sort_rotation_series(series):
    series["angles"][series["angles"] > 360] = series["angles"][series["angles"] > 360] - 360
    series["angles"][series["angles"] < 0] = series["angles"][series["angles"] < 0] + 360

    sorting_indices = list(np.argsort(series["angles"]))
    series["angles"] = series["angles"][np.argsort(series["angles"])]
    for raw_list in [series["delay_list"], series["signal_list"], series["label_list"]]:
        raw_list[:] = [raw_list[i] for i in sorting_indices]


def get_measurements(params):
    measurement_list = []
    current_directory = os.getcwd()
    os.chdir(params["raw_data_directory"])
    data_directory = params["date"] + "_" + params["time"]
    filename = "AllData_Reduced.txt"
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file == filename:
                path_combined = os.path.join(root, file)
                path = root.split("\\")
                print(path)
                if len(path) >= 4:
                    measurement_list.append([path[0].split("_")[0], path[0].split("_")[1], path[2],
                                             path[3], params["sample"]])
                    print(path_combined)
    os.chdir(current_directory)
    return measurement_list


def export_raw_data(params):
    if not ("export_directory" in params):
        params["export_directory"] = ""
    if not ("raw_data_directory" in params):
        params["raw_data_directory"] = "data\\"

    plot_overview_path = params["export_directory"] + "plot_overview\\"
    plot_loopwise_path = params["export_directory"] + "plot_loopwise\\"

    result_overview_path = params["export_directory"] + "data_overview\\"
    result_loopwise_path = params["export_directory"] + "data_loopwise\\"

    for path in [plot_overview_path, plot_loopwise_path,
                 result_overview_path, result_loopwise_path]:
        tools.make_folder(path)

    measurement_list = get_measurements(params)
    data_directory = params["date"] + "_" + params["time"]

    # col_index = 0
    col_diff_signal = 1
    col_diode_a = 2
    col_diode_b = 3
    # col_reference = 4
    col_chopper = 5
    col_delay = 6
    col_loop = 7
    col_voltage = 8
    print(measurement_list)
    for line, pars in enumerate(measurement_list):
        print("Exporting measurement " + str(line + 1) + " / " + str(len(measurement_list)))
        print(pars)
        datepath = pars[0] + "_" + pars[1]
        result_file_name = datepath + "_" + pars[2] + "_" + pars[3]
        path = data_directory + "\\Fluence\\" + pars[2] + "\\" + pars[3] + "\\"
        title = pars[4] + "  " + datepath + "    " + pars[2] + "   " + str(pars[3]) + " V"

        data_in = np.genfromtxt(params["raw_data_directory"] + path + "AllData_Reduced.txt", skip_header=1)
        data_raw = data_in[np.isfinite(data_in[:, 1]), :]

        #
        if len(data_raw) > 0:
            unique_delays = np.unique(data_raw[:, col_delay])
            loops = int(np.max(data_raw[:, col_loop]))
            n_delays = len(unique_delays)
            data_avg_field_up = np.zeros((n_delays, 6))
            data_avg_field_down = np.zeros((n_delays, 6))
            data_avg_result = np.zeros((n_delays, 5))
            data_loop_field_up = np.zeros((n_delays, loops + 1))
            data_loop_field_down = np.zeros((n_delays, loops + 1))

            s_pumped = data_raw[:, col_chopper] == 1
            s_not_pumped = data_raw[:, col_chopper] == 0
            s_field_up = data_raw[:, col_voltage] > 0
            s_field_down = data_raw[:, col_voltage] <= 0
            if np.sum(s_field_up) == 0 or np.sum(s_field_down) == 0:
                s_field_up = s_field_down = np.ones(len(data_raw[:, col_chopper])) > 0
                # s_field_up = data_raw[:, col_voltage] > np.mean(data_raw[:, col_voltage])
                # s_field_down = #data_raw[:, col_voltage] <= np.mean(data_raw[:, col_voltage])

            for array in [data_avg_field_up, data_avg_field_down, data_avg_result,
                          data_loop_field_down, data_loop_field_up]:
                array[:, 0] = unique_delays

            for i, t in enumerate(unique_delays):
                s_delay = data_raw[:, col_delay] == t

                s_field_up_pumped = s_delay & s_field_up & s_pumped
                s_field_up_not_pumped = s_delay & s_field_up & s_not_pumped

                s_field_down_pumped = s_delay & s_field_down & s_pumped
                s_field_down_not_pumped = s_delay & s_field_down & s_not_pumped

                for column in [col_diff_signal, col_diode_a, col_diode_b]:
                    data_avg_field_up[i, column] = np.mean(data_raw[s_field_up_pumped, column]) - \
                        np.mean(data_raw[s_field_up_not_pumped, column])

                    data_avg_field_down[i, column] = np.mean(data_raw[s_field_down_pumped, column]) - \
                        np.mean(data_raw[s_field_down_not_pumped, column])

            for array in [data_avg_field_up, data_avg_field_down]:
                array[:, 4] = array[:, col_diode_a] + array[:, col_diode_b]
                array[:, 5] = array[:, col_diode_a] - array[:, col_diode_b]

            data_avg_result[:, 1] = data_avg_field_up[:, col_diff_signal] + data_avg_field_down[:, col_diff_signal]
            data_avg_result[:, 2] = data_avg_field_up[:, col_diff_signal] - data_avg_field_down[:, col_diff_signal]
            data_avg_result[:, 3] = data_avg_field_up[:, col_diff_signal]
            data_avg_result[:, 4] = data_avg_field_down[:, col_diff_signal]

            tools.write_list_to_file(result_overview_path + "overview_data_" + result_file_name + ".txt",
                                     u"time (ps)\t sum signal (V)\t MOKE (V)\t field up signal (V)"
                                     + " \t field down signal (V)", data_avg_result)

            # Plot results
            t_min = np.min(unique_delays)
            t_max = np.max(unique_delays)

            plt.figure(figsize=(5.2, 5.2 / 0.68))
            gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])

            ax1.plot(data_avg_result[:, 0], data_avg_result[:, 3],
                     '-', color=colors.red_1, lw=2, label="field  up")
            ax1.plot(data_avg_result[:, 0], data_avg_result[:, 4], '-',
                     color=colors.blue_1, lw=2, label="field down")
            ax2.plot(data_avg_result[:, 0], data_avg_result[:, 1], '-',
                     color=colors.orange_2, lw=2, label="sum signal")
            ax3.plot(data_avg_result[:, 0], data_avg_result[:, 2], '-', color=colors.grey_1,
                     lw=2, label="diff signal")

            ax1.set_ylabel(
                r'single signal  ($\,\mathrm{V}$)' + "\n" + r'$\mathrm{S_{+/-}   = I_{+/-}^{1} - I_{+/-}^{0}}$ ')
            ax2.set_ylabel(r'sum signal  ($\,\mathrm{V}$) ' + "\n" + r'$\mathrm{S_{+}\,\, +\,\, S_{-}}$')
            ax3.set_ylabel(r'MOKE signal ($\,\mathrm{V}$) ' + "\n" + r'$\mathrm{S_{+} \,\,-\,\, S_{-}}$')
            ax3.set_xlabel('time (ps)')

            for ax in [ax1, ax2, ax3]:
                ax.legend(loc=4)
                ax.set_xlim((t_min, t_max))
            ax3.axhline(y=0, ls="--", color="grey")
            ax1.xaxis.set_ticks_position('top')
            ax2.set_xticklabels([])
            ax1.set_title(title, pad=13)

            plt.savefig(plot_overview_path + 'overview_' + result_file_name + ".png", dpi=150, bbox_inches="tight")
            plt.show()

            # Read in data loopwise

            for loop in range(loops):
                s_loop = data_raw[:, col_loop] == loop + 1
                for i, t in enumerate(unique_delays):
                    s_delay = data_raw[:, col_delay] == t
                    s_field_up_pumped = ((s_loop & s_field_up) & s_pumped) & s_delay
                    s_field_up_not_pumped = ((s_loop & s_field_up) & s_not_pumped) & s_delay
                    s_field_down_pumped = ((s_loop & s_field_down) & s_pumped) & s_delay
                    s_field_down_not_pumped = ((s_loop & s_field_down) & s_not_pumped) & s_delay
                    column = col_diff_signal
                    data_loop_field_up[i, loop + 1] = np.mean(data_raw[s_field_up_pumped, column]) - \
                        np.mean(data_raw[s_field_up_not_pumped, column])
                    data_loop_field_down[i, loop + 1] = np.mean(data_raw[s_field_down_pumped, column]) - \
                        np.mean(data_raw[s_field_down_not_pumped, column])
            tools.write_list_to_file(result_loopwise_path + "field_up_" + result_file_name + ".txt",
                                     "time (ps)    data plus loopwise", data_loop_field_up)
            tools.write_list_to_file(result_loopwise_path + "field_down_" + result_file_name + ".txt",
                                     "time (ps)    data minus loopwise", data_loop_field_down)

            # Plot data loopwise
            c_map = colors.cmap_1

            plt.figure(figsize=(5.2, 5.2 / 0.68))
            gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])

            for loop in range(loops):
                ax1.plot(data_loop_field_up[:, 0], data_loop_field_up[:, loop + 1], '-',
                         color=c_map(loop / loops), lw=1, label=str(loop + 1))
                ax2.plot(data_loop_field_down[:, 0], data_loop_field_down[:, loop + 1],
                         '-', color=c_map(loop / loops), lw=1, label=str(loop + 1))
                ax3.plot(data_loop_field_down[:, 0],
                         data_loop_field_up[:, loop + 1] - data_loop_field_down[:, loop + 1],
                         '-', color=c_map(loop / loops), lw=1, label=str(loop + 1))

            ax1.plot(data_avg_result[:, 0], data_avg_result[:, 3], '-',
                     color=colors.grey_1, lw=2, label="avg")
            ax2.plot(data_avg_result[:, 0], data_avg_result[:, 4], '-',
                     color=colors.grey_1, lw=2, label="avg.")
            ax3.plot(data_avg_result[:, 0], data_avg_result[:, 2], '-',
                     color=colors.grey_1, lw=2, label="avg.")

            ax1.set_ylabel('field up  signal \n' + r'$\mathrm{S_{+}} (\mathrm{V})$')
            ax2.set_ylabel('field down signal \n' + r' $\mathrm{S_{-}} (\mathrm{V})$')
            ax3.set_ylabel('MOKE signal \n' r'$\mathrm{S_{+}\,\,-\,\, S_{-}}$ ($\,\mathrm{V}$)')
            ax3.set_xlabel('time (ps)')

            for ax in [ax1, ax2, ax3]:
                ax.legend(loc=1, fontsize=9, ncol=7, handlelength=1, columnspacing=1.5)
                ax.set_xlim((t_min, t_max))
            ax1.xaxis.set_ticks_position('top')
            ax2.set_xticklabels([])
            ax1.set_title(title, pad=13)

            plt.savefig(plot_loopwise_path + 'loopwise_' + result_file_name + ".png", dpi=150)
            plt.show()


def load_rotation_series(params):
    measurement_list = get_measurements(params)
    params["fluence"] = measurement_list[0][2]
    params["voltage"] = measurement_list[0][3]

    scan = load_data(params)
    n_measurements = len(measurement_list)

    series = scan
    series["angles"] = np.zeros(n_measurements)
    series["signal_list"] = []
    series["delay_list"] = []
    series["label_list"] = []

    for i, parameters in enumerate(measurement_list):
        angle = float(measurement_list[i][2])-params["angle_offset"]
        while angle < 0:
            angle += 360
        while angle > 360:
            angle -= 360

        series["angles"][i] = angle
        params["fluence"] = measurement_list[i][2]
        params["voltage"] = measurement_list[i][3]
        scan = load_data(params)

        if params["regrid_scan"]:
            delay_regridded, signal_regridded, _, _ = tools.regrid_measurement(scan["delay"], scan["sum"]
                                                                               - np.mean(scan["sum"]
                                                                                         [scan["delay"] < -1]),
                                                                               params["t_grid_new"])
            series["delay_list"].append(delay_regridded)
            series["signal_list"].append(signal_regridded)
        else:
            series["delay_list"].append(scan["delay"])
            series["signal_list"].append(scan[params["col_to_plot"]] -
                                         np.mean(scan[params["col_to_plot"]][scan["delay"] < -1]))
        series["label_list"].append(str(int(angle)) + r"$^\circ$")
    return series


def fft_series(series, fft_params):
    time_grid_fft = np.arange(fft_params["t_start"], fft_params["t_end"], fft_params["t_step"])
    delay = series["delay_list"][0]

    extension = 5000
    series["frequency_list"] = []
    series["fft_list"] = []

    for i, signal in enumerate(series["signal_list"]):
        signal_extended = np.append(signal, np.ones(extension)*signal[-1])

        extension = len(signal_extended)-len(delay)
        delay_extended = np.append(delay, np.linspace(delay[-1], 10000, extension+1)[1:])
        interpolated_signal = np.interp(time_grid_fft, delay_extended, signal_extended)

        frequency, fft_result = tools.fft(time_grid_fft, interpolated_signal)

        series["frequency_list"].append(frequency*1000)
        series["fft_list"].append(fft_result*fft_params["fft_factor"])
    return series


def plot_rotation_series(series, plot_params):
    w_space = 0.068   # the amount of width reserved for blank space between subplots
    h_space = 0.1   # the amount of height reserved for white space between subplots
    axlabel_fontsize = 10
    ticklabel_fontSize = 8

    c_map = colors.cmap_blue_red_3

    f = plt.figure(figsize=[5.2, 5.2])
    gs = gridspec.GridSpec(1, 2, wspace=w_space, hspace=h_space)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    X, Y = np.meshgrid(series["delay_list"][0], series["angles"])
    plotted_1 = ax1.pcolormesh(X, Y, np.array(series["signal_list"])*plot_params["factor"], cmap=c_map, shading='auto',
                               vmin=plot_params["signal_min"], vmax=plot_params["signal_max"])
    X, Y = np.meshgrid(series["frequency_list"][0], series["angles"])
    plotted_2 = ax2.pcolormesh(X, Y, np.array(np.abs(series["fft_list"])), cmap=c_map, shading='auto',
                               norm=matplotlib.colors.LogNorm(vmin=plot_params["fft_min"], vmax=plot_params["fft_max"]))

    for ax in [ax1]:
        ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        if "t_min" in plot_params:
            ax.set_xlim([plot_params["t_min"], plot_params["t_max"]])
        if "angle_min" in plot_params:
            ax.set_ylim(plot_params["angle_min"], plot_params["angle_max"])
        ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontSize)

    for ax in [ax2]:
        ax.set_yticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
        ax.set_yticklabels(["", "", "", "", "", "", "", "", ""])
        if "f_min" in plot_params:
            ax.set_xlim([plot_params["f_min"], plot_params["f_max"]])
        if "angle_min" in plot_params:
            ax.set_ylim(plot_params["angle_min"], plot_params["angle_max"])
        ax.yaxis.tick_right()
        ax.tick_params(axis='both', which='major', labelsize=ticklabel_fontSize)

    ax1.set_ylabel(r'$\phi$ ($\,^\circ$)', fontsize=axlabel_fontsize)
    ax1.set_xlabel('delay (ps)', fontsize=axlabel_fontsize)
    ax2.set_xlabel('frequency (GHz)', fontsize=axlabel_fontsize)

    cb_ax1 = f.add_axes([0.125, 0.89, 0.37, 0.02])
    cbar_1 = plt.colorbar(plotted_1, cax=cb_ax1, orientation='horizontal')
    cbar_1.set_label(label='MOKE (arb. units)', rotation=0, fontsize=axlabel_fontsize)

    cbar_1.ax.tick_params('both', length=5, width=0.5, which='major', direction='in')
    plt.getp(cbar_1.ax, 'xmajorticklabels')
    cbar_1.ax.tick_params(labelsize=8, direction='in')
    cb_ax1.xaxis.set_ticks_position("top")
    cb_ax1.xaxis.set_label_position("top")

    cb_ax2 = f.add_axes([0.53, 0.89, 0.37, 0.02])
    cbar_2 = plt.colorbar(plotted_2, cax=cb_ax2, orientation='horizontal')
    cbar_2.set_label(label='fft amplitude (arb. units)', fontsize=axlabel_fontsize)
    cbar_2.ax.tick_params('both', length=5, width=0.5, which='major', direction='in')
    plt.getp(cbar_2.ax, 'xmajorticklabels')
    cbar_2.ax.tick_params(labelsize=8, direction='in')
    cb_ax2.xaxis.set_ticks_position("top")
    cb_ax2.xaxis.set_label_position("top")
    return ax1, ax2


def analyze_beamprofile(parameters):
    scan = {}
    scan["date"] = parameters["date"]
    scan["time"] = parameters["time"]
    scan["filename"] = scan["date"] + "_" + scan["time"] + "_image.txt"
    scan["path"] = parameters["data_directory"] + "\\" + scan["date"] + "\\"

    scan["rep_rate"] = 1000
    scan["angle"] = 0

    scan["data"] = np.genfromtxt(scan["path"]+scan["filename"])
    scan["pixel_x"] = np.arange(0, np.shape(scan["data"])[1]+1, 1)
    scan["pixel_y"] = np.arange(0, np.shape(scan["data"])[0]+1, 1)

    pixelsize_x = 5.2  # µm per pixel
    pixelsize_y = 5.2  # µm per pixel
    scan["distances_x"] = scan["pixel_x"]*pixelsize_x  # pixel in µm
    scan["distances_y"] = scan["pixel_y"]*pixelsize_y  # pixel in µm

    # %% #Overview Plot to define the ROI
    scan["x_min"] = 0
    scan["x_max"] = len(scan["pixel_x"])*pixelsize_x

    scan["y_min"] = 0
    scan["y_max"] = len(scan["pixel_y"])*pixelsize_y

    X, Y = np.meshgrid(scan["distances_x"], scan["distances_y"])

    plt.figure(1, figsize=(5.2, 5.2*scan["x_max"]/scan["y_max"]))
    if parameters["plot_logarithmic"]:
        pl = plt.pcolormesh(X, Y, scan["data"], cmap=colors.fireice(),
                            norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.max(scan["data"])))
    else:
        pl = plt.pcolormesh(X, Y, scan["data"], cmap=colors.fireice(), vmin=0, vmax=np.max(scan["data"]))
    plt.axis([0, X.max(), 0, Y.max()])
    plt.xlabel(r'x ($\mathrm{\mu{}}$m)')
    plt.ylabel(r'y ($\mathrm{\mu{}}$m)')
    plt.title(scan["date"] + "  " + scan["time"])
    cb = plt.colorbar()
    cb.ax.set_title('$I$ (cts)')

    entry_list = ["x_min", "x_max", "y_min", "y_max", "power", "rep_rate"]
    for entry in entry_list:
        if entry in parameters:
            scan[entry] = parameters[entry]

    scan["x_roi"], scan["y_roi"], scan["data_roi"], scan["x_integral"], scan["y_integral"] = tools.set_roi_2d(
        scan["distances_x"][1:], scan["distances_y"][1:], scan["data"], scan["x_min"], scan["x_max"],
        scan["y_min"], scan["y_max"])

    scan["distance_x_cut"] = scan["x_roi"]-scan["x_min"]
    scan["distance_y_cut"] = scan["y_roi"]-scan["y_min"]

    X, Y = np.meshgrid(scan["distance_x_cut"], scan["distance_y_cut"])

    dX = scan["x_max"]-scan["x_min"]
    dY = scan["y_max"]-scan["y_min"]

    imax_y = tools.find(scan["y_integral"], np.max(scan["y_integral"]))
    slice_y = scan["data_roi"][imax_y, :]

    imax_x = tools.find(scan["x_integral"], np.max(scan["x_integral"]))
    slice_x = scan["data_roi"][:, imax_x]

    # %% Fitting the resulting Parameters
    model = lm.models.GaussianModel() + lm.models.LinearModel()
    parsX = lm.Parameters()
    parsY = lm.Parameters()

    com_x, std_x, Ix = tools.calc_moments(scan["distance_x_cut"], scan["x_integral"])
    com_y, std_y, Iy = tools.calc_moments(scan["distance_y_cut"], scan["y_integral"])
    # Here you can set the initial values and possible boundaries on the fitting parameters
    # Name       Value                 Vary     Min                   Max

    parsX.add_many(('center',    com_x,    True),
                   ('sigma',     std_x,     True),
                   ('amplitude', 200,     True),
                   ('slope',     0,    False),
                   ('intercept', 0, True))

    # Name       Value                 Vary     Min                   Max
    parsY.add_many(('center',    com_y,    True),
                   ('sigma',    std_y,     True),
                   ('amplitude', 200,     True),
                   ('slope',     0,   False),
                   ('intercept', 0, True))

    # Fitting takes place here
    result_x = model.fit(scan["x_integral"]/np.max(scan["x_integral"]), parsX, x=scan["distance_x_cut"])
    result_y = model.fit(scan["y_integral"]/np.max(scan["y_integral"]), parsY, x=scan["distance_y_cut"])

    result_x_slice = model.fit(slice_x/np.max(slice_x), parsX, x=scan["distance_y_cut"])
    result_y_slice = model.fit(slice_y/np.max(slice_y), parsY, x=scan["x_roi"]-scan["x_min"])

    # Writing the results into the peaks dictionary takes place here
    scan["fwhm_x"] = 2.35482*result_x.values["sigma"]  # in micron
    scan["fwhm_y"] = 2.35482*result_y.values["sigma"]  # in micron

    scan["fwhm_x_slice"] = 2.35482*result_x_slice.values["sigma"]  # in micron
    scan["fwhm_y_slice"] = 2.35482*result_y_slice.values["sigma"]  # in micron

    # x0 = scan["fwhm_x"]/np.sqrt(np.log(2))   # Definition according to excel-sheet
    # y0 = scan["fwhm_y"]/np.sqrt(np.log(2))   # in micron

    scan["fluence"] = tools.calc_fluence(scan["power"], scan["fwhm_x"], scan["fwhm_y"], 0, scan["rep_rate"])
    print("power_in  = " + str(np.round(scan["power"])) + "mW (without chopper)\n->  F at " +
          str(int(scan["angle"]))+"° = " + str(np.round(scan["fluence"])) + "mJ/cm^2")

    integral_result_text = str(
        int(round(scan["fwhm_x"]))) + r'$\,\mathrm{\mu{}}$m' + " x " + str(int(round(scan["fwhm_y"]))) + r'$\,\mathrm{\mu{}}$m'
    slice_result_text = str(int(round(scan["fwhm_x_slice"]))) + \
        r'$\,\mathrm{\mu{}}$m' + " x " + str(int(round(scan["fwhm_y_slice"]))) + r'$\,\mathrm{\mu{}}$m'

    # %% Plotting the results in the ROI

    plt.figure(2, figsize=(5.2, 5.2), linewidth=2)
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
    ax1.plot(scan["distance_x_cut"], result_x.best_fit, '-',
             color=colors.grey_1, lw=1, label="fit " + integral_result_text)
    ax1.plot(scan["distance_x_cut"], slice_y/np.max(slice_y), '-', color=colors.blue_2, lw=1, label='slice')
    ax1.plot(scan["distance_x_cut"], result_y_slice.best_fit, '--',
             color=colors.blue_1, lw=1, label="fit" + slice_result_text)

    ax1.set_ylabel('I (a.u.)')
    ax1.set_xlabel(r'x ($\mathrm{\mu{}}$m)')
    ax1.set_ylim([0, 1.05])
    ax1.set_yticks(np.arange(0.25, 1.25, .25))

    # (ax 3) Colormap of the Profile #############
    if parameters["plot_logarithmic"]:
        pl = ax3.pcolormesh(X, Y, scan["data_roi"], cmap=colors.fireice(),
                            norm=matplotlib.colors.LogNorm(vmin=1, vmax=np.max(scan["data_roi"])))
    else:
        pl = ax3.pcolormesh(X, Y, scan["data_roi"], cmap=colors.fireice(), vmin=0, vmax=np.max(scan["data"]))

    ax3.axis([0, scan["x_max"]-scan["x_min"], 0, scan["y_max"]-scan["y_min"]])
    ax3.set_xlabel(r'x ($\mathrm{\mu{}}$m)')
    ax3.set_ylabel(r'y ($\mathrm{\mu{}}$m)')

    ax3.axvline(x=scan["distance_x_cut"][imax_x], ls="--", color=colors.blue_1, lw=0.5)
    ax3.axhline(y=scan["distance_y_cut"][imax_y], ls="--", color=colors.blue_1, lw=0.5)

    # colorbar placement #############
    axins3 = inset_axes(ax3,
                        width="60%",  # width = 10% of parent_bbox width
                        height="3%",  # height : 50%                   )
                        loc=4)
    ax3.add_patch(Rectangle((0.35, 0.018), 0.65, 0.18, edgecolor="none",
                  facecolor="white", alpha=0.75, transform=ax3.transAxes))
    cbar = plt.colorbar(pl, cax=axins3, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8)

    cbar.ax.set_title('$I$ (cts)', fontsize=8)

    axins3.xaxis.set_ticks_position("top")
    axins3.xaxis.set_label_position("top")
    cl = plt.getp(cbar.ax, 'xmajorticklabels')

    plt.setp(cl, color="black")

    ax2.axis('off')

    # Plot 4) Bottom Right Vertical Profile

    ax4.plot(scan["y_integral"]/np.max(scan["y_integral"]), scan["distance_y_cut"],
             '-', color=colors.grey_3, lw=2, label='integral')
    ax4.plot(result_y.best_fit, scan["distance_y_cut"], '-',
             color=colors.grey_1, lw=1, label="fit " + integral_result_text)
    ax4.plot(slice_x/np.max(slice_x), scan["distance_y_cut"], '-', color=colors.blue_2, lw=1, label='slice')
    ax4.plot(result_x_slice.best_fit, scan["distance_y_cut"],  '--',
             color=colors.blue_1, lw=1, label="fit " + slice_result_text)

    ax4.set_ylabel(r'y ($\mathrm{\mu{}}$m)')
    ax4.set_xlabel('I (a.u.)')
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.legend(loc=(0.05, 1.05), frameon=False, fontsize=8)

    ax4.set_ylim(0, 1.05)
    ax4.set_ylim(0, dY)

    ax4.set_xticks([0, 1])
    ax1.set_yticks([0, 1])

    ax1.set_xlim([0, dX])
    ax4.set_ylim([0, dY])

    # %%
    print(str(int(round(scan["fwhm_x"]))) + " x " + str(int(round(scan["fwhm_y"]))) + str(" integral FWHM in microns"))
    print(str(int(round(scan["fwhm_x_slice"]))) + " x " +
          str(int(round(scan["fwhm_y_slice"]))) + str(" slice FWHM in microns"))

    return scan
