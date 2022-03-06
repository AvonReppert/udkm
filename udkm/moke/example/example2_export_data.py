# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:58:45 2022

@author: Udkm
"""

# import pickle
import udkm.moke.functions as moke
import udkm.tools.functions as tools
import numpy as np
import matplotlib.pyplot as plt
import udkm.tools.colors as colors
import matplotlib.gridspec as gridspec
import os


line = 0
parameter_file_name = "parameters/parameters_example_3.txt"

params = moke.get_scan_parameter(parameter_file_name, line)
params["export_directory"] = ""
params["data_directory"] = "data\\"


def load_raw_data(params):
    plot_overview_path = params["export_directory"] + "plot_overview\\"
    plot_loopwise_path = params["export_directory"] + "plot_loopwise\\"

    result_overview_path = params["export_directory"] + "data_overview\\"
    result_loopwise_path = params["export_directory"] + "data_loopwise\\"

    for path in [plot_overview_path, plot_loopwise_path,
                 result_overview_path, result_loopwise_path]:
        tools.make_folder(path)

    parameter_list = []
    data_directory = params["data_directory"] + params["date"] + "_" + params["time"]
    filename = "AllData_Reduced.txt"
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file == filename:
                path_combined = os.path.join(root, file)
                path = root.split("\\")
                if len(path) > 4:
                    parameter_list.append([path[1].split("_")[0], path[1].split("_")[1], path[3],
                                           path[4],  params["sample"]])
                    print(path_combined)

    col_index = 0
    col_diff_signal = 1
    col_diode_a = 2
    col_diode_b = 3
    col_reference = 4
    col_chopper = 5
    col_delay = 6
    col_loop = 7
    col_voltage = 8

    # %%
    for line, pars in enumerate(parameter_list):
        print("Exporting measurement " + str(line+1) + " / " + str(len(parameter_list)))
        print(pars)
        datepath = pars[0]+"_"+pars[1]
        result_file_name = datepath+"_"+pars[2]+"_"+pars[3]
        path = data_directory+"\\Fluence\\"+pars[2]+"\\"+pars[3]+"\\"
        title = pars[4] + "  " + datepath + "    " + pars[2] + "   " + str(pars[3]) + " V"

        data_in = np.genfromtxt(path+"AllData_Reduced.txt", skip_header=1)
        data_raw = data_in[np.isfinite(data_in[:, 1]), :]

    #
        if len(data_raw) > 0:
            unique_delays = np.unique(data_raw[:, col_delay])
            loops = int(np.max(data_raw[:, col_loop]))
            n_delays = len(unique_delays)
            data_avg_field_up = np.zeros((n_delays, 6))
            data_avg_field_down = np.zeros((n_delays, 6))
            data_avg_result = np.zeros((n_delays, 5))
            data_loop_field_up = np.zeros((n_delays, loops+1))
            data_loop_field_down = np.zeros((n_delays, loops+1))

            s_pumped = data_raw[:, col_chopper] == 1
            s_not_pumped = data_raw[:, col_chopper] == 0
            s_field_up = data_raw[:, col_voltage] > 0
            s_field_down = data_raw[:, col_voltage] <= 0
            if np.sum(s_field_up) == 0 or np.sum(s_field_down) == 0:
                s_field_up = data_raw[:, col_voltage] > np.mean(data_raw[:, col_voltage])
                s_field_down = data_raw[:, col_voltage] <= np.mean(data_raw[:, col_voltage])

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

            tools.write_list_to_file(result_overview_path+"overview_data_"+result_file_name + ".txt",
                                     u"time (ps)\t sum signal (V)\t MOKE (V)\t field up signal (V)"
                                     + " \t field down signal (V)", data_avg_result)
            # %%

            t_min = np.min(unique_delays)
            t_max = np.max(unique_delays)

            f = plt.figure(figsize=(5.2, 5.2/0.68))
            gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])

            ax1.plot(data_avg_result[:, 0], data_avg_result[:, 3],
                     '-', color=colors.red_1, lw=2,  label="field  up")
            ax1.plot(data_avg_result[:, 0], data_avg_result[:, 4], '-',
                     color=colors.blue_1, lw=2,  label="field down")
            ax2.plot(data_avg_result[:, 0], data_avg_result[:, 1], '-',
                     color=colors.orange_2,  lw=2,  label="sum signal")
            ax3.plot(data_avg_result[:, 0], data_avg_result[:, 2], '-', color=colors.grey_1,
                     lw=2,  label="diff signal")

            ax1.set_ylabel(
                'single signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+/-}   = I_{+/-}^{1} - I_{+/-}^{0}}$ ')
            ax2.set_ylabel(u'sum signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+}\,\, +\,\, S_{-}}$ ')
            ax3.set_ylabel(u'MOKE signal ($\,\mathrm{V}$) \n $\mathrm{S_{+} \,\,-\,\, S_{-}}$')
            ax3.set_xlabel('time (ps)')

            for ax in [ax1, ax2, ax3]:
                ax.legend(loc=4)
                ax.set_xlim((t_min, t_max))
            ax3.axhline(y=0, ls="--", color="grey")
            ax1.xaxis.set_ticks_position('top')
            ax2.set_xticklabels([])
            ax1.set_title(title,  pad=13)

            plt.savefig(plot_overview_path+'overview_'+result_file_name+".png", dpi=150)
            plt.show()

            # Read in data loopwise

            for loop in range(loops):
                s_loop = data_raw[:, col_loop] == loop+1
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
            tools.write_list_to_file(result_loopwise_path+"field_up_"+result_file_name + ".txt",
                                     u"time (ps)\t data plus loopwise", data_loop_field_up)
            tools.write_list_to_file(result_loopwise_path+"field_down_"+result_file_name + ".txt",
                                     u"time (ps)\t data minus loopwise", data_loop_field_down)

            # Plot data loopwise
            c_map = colors.cmap_1

            f = plt.figure(figsize=(5.2, 5.2/0.68))
            gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])

            for loop in range(loops):
                ax1.plot(data_loop_field_up[:, 0], data_loop_field_up[:, loop+1], '-',
                         color=c_map(loop/loops), lw=1,  label=str(loop+1))
                ax2.plot(data_loop_field_down[:, 0], data_loop_field_down[:, loop+1],
                         '-',  color=c_map(loop/loops), lw=1,  label=str(loop+1))
                ax3.plot(data_loop_field_down[:, 0], data_loop_field_up[:, loop+1]-data_loop_field_down[:, loop+1],
                         '-',  color=c_map(loop/loops), lw=1,  label=str(loop+1))

            ax1.plot(data_avg_result[:, 0], data_avg_result[:, 3], '-',
                     color=colors.grey_1, lw=2,  label="avg")
            ax2.plot(data_avg_result[:, 0], data_avg_result[:, 4], '-',
                     color=colors.grey_1, lw=2,  label="avg.")
            ax3.plot(data_avg_result[:, 0], data_avg_result[:, 2], '-',
                     color=colors.grey_1, lw=2,  label="avg.")

            ax1.set_ylabel('field up  signal \n'+r'$\mathrm{S_{+}} (\mathrm{V})$')
            ax2.set_ylabel('field down signal \n'+r' $\mathrm{S_{-}} (\mathrm{V})$')
            ax3.set_ylabel('MOKE signal \n' r'$\mathrm{S_{+}\,\,-\,\, S_{-}}$ ($\,\mathrm{V}$)')
            ax3.set_xlabel('time (ps)')

            for ax in [ax1, ax2, ax3]:
                ax.legend(loc=1, fontsize=9, ncol=7, handlelength=1, columnspacing=1.5)
                ax.set_xlim((t_min, t_max))
            ax1.xaxis.set_ticks_position('top')
            ax2.set_xticklabels([])
            ax1.set_title(title,  pad=13)

            plt.savefig(plot_loopwise_path+'loopwise_'+result_file_name+".png", dpi=150)
            plt.show()


load_raw_data(params)
