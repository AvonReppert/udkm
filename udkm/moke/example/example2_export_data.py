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
sample_name = params["sample"]
plot_path = "plot_overview\\"

directory = "data\\"+params["date"]+"_"+params["time"]
filename = "AllData_Reduced.txt"
result_path = "data_overview\\"

parameter_list = []
for root, _, files in os.walk(directory):
    for file in files:
        if file == filename:
            path_combined = os.path.join(root, file)
            path = root.split("\\")
            if len(path) > 4:
                parameter_list.append([path[1].split("_")[0], path[1].split("_")[1], path[3],
                                       path[4], sample_name])
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
    print(line)
    print(pars)
    datepath = pars[0]+"_"+pars[1]
    result_file_name = datepath+"_"+pars[2]+"_"+pars[3]
    path = directory+"\\Fluence\\"+pars[2]+"\\"+pars[3]+"\\"
    title = pars[4] + "  " + datepath + "    " + pars[2] + "   " + str(pars[3]) + " V"

    data_in = np.genfromtxt(path+"AllData_Reduced.txt", skip_header=1)
    data_raw = data_in[np.isfinite(data_in[:, 1]), :]
# %%
    if len(data_raw) > 0:
        if float(pars[3]) > 0:

            unique_delays = np.unique(data_raw[:, col_delay])
            loops = int(np.max(data_raw[:, col_loop]))
            n_delays = len(unique_delays)+-
            data_avg_field_up = np.zeros((n_delays, 6))
            data_avg_field_down = np.zeros((n_delays, 6))
            data_avg_result = np.zeros((n_delays, 5))
            data_loop_field_up = np.zeros((n_delays, loops+1))
            data_loop_field_down = np.zeros((n_delays, loops+1))

            s_pumped = data_raw[:, col_chopper] == 1
            s_not_pumped = data_raw[:, col_chopper] == 0
            s_field_up = data_raw[:, col_voltage] > 0
            s_field_down = data_raw[:, col_voltage] <= 0

            for array in [data_avg_field_up, data_avg_field_down, data_avg_result, data_loop_field_down, data_loop_field_up]:
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
            # %%
            plot_overview = True
            if plot_overview:
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
                ax2.set_ylabel('sum signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+}\,\, +\,\, S_{-}}$ ')
                ax3.set_ylabel('MOKE signal ($\,\mathrm{V}$) \n $\mathrm{S_{+} \,\,-\,\, S_{-}}$')
                ax3.set_xlabel('time (ps)', fontsize=14)

                for ax in [ax1, ax2, ax3]:
                    ax.legend(loc=4)
                ax3.axhline(y=0, ls="--", color="grey")
                ax1.xaxis.set_ticks_position('top')
                ax2.set_xticklabels([])
                ax1.set_title(title,  pad=13)

                plt.savefig(plot_path+'overview_'+result_file_name+".png")
                plt.show()

                tools.write_list_to_file(result_path+"overview_data_"+result_file_name + ".txt",
                                         u"time (ps)\t sum signal (V)\t MOKE (V)\t field up signal (V) \t field down signal (V)",
                                         data_avg_result)

            #    # %%
            #    for l in range(loops):
            #         sLoop = data_raw[:, col_loop] == l+1
            #         for i in range(len(unique_delays)):
            #             t = unique_delays[i]
            #             s_delay = data_raw[:, col_delay] == t

            #             s_field_up_pumped = ((sLoop & s_field_up) & s_pumped) & s_delay
            #             s_field_up_not_pumped = ((sLoop & s_field_up) & s_not_pumped) & s_delay
            #             s_field_down_pumped = ((sLoop & s_field_down) & s_pumped) & s_delay
            #             s_field_down_not_pumped = ((sLoop & s_field_down) & s_not_pumped) & s_delay
            #             column = col_diff_signal
            #             data_loop_field_up[i, l+1] = np.mean(data_raw[s_field_up_pumped, column]) - \
            #                 np.mean(data_raw[s_field_up_not_pumped, column])
            #             data_loop_field_down[i, l+1] = np.mean(data_raw[s_field_down_pumped, column]) - \
            #                 np.mean(data_raw[s_field_down_not_pumped, column])

            #     # %%
            #     cMap = plt.get_cmap('hsv')

            #     f = plt.figure(figsize=(8, 8))
            #     gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
            #     ax1 = plt.subplot(gs[0])
            #     ax2 = plt.subplot(gs[1])
            #     ax3 = plt.subplot(gs[2])

            #     for l in range(loops):
            #         ax1.plot(data_loop_field_up[:, 0], data_loop_field_up[:, l+1], '-',
            #                  color=cMap(l/loops), lw=1,  label=str(l+1))
            #         ax2.plot(data_loop_field_down[:, 0], data_loop_field_down[:, l+1],
            #                  '-',  color=cMap(l/loops), lw=1,  label=str(l+1))
            #         ax3.plot(data_loop_field_down[:, 0], data_loop_field_up[:, l+1]-data_loop_field_down[:, l+1],
            #                  '-',  color=cMap(l/loops), lw=1,  label=str(l+1))

            #     ax1.plot(data_avg_result[:, 0], data_avg_result[:, 3], '-',  color="blue", lw=2,  label="B plus avg")
            #     ax2.plot(data_avg_result[:, 0], data_avg_result[:, 4], '-',  color="green", lw=2,  label="B minus avg")
            #     ax3.plot(data_avg_result[:, 0], data_avg_result[:, 2], '-',  color="black", lw=2,  label="MOKE avg")

            #     ax1.set_ylabel('plus signal  ($\,\mathrm{V}$)', fontsize=14)
            #     ax2.set_ylabel('minus signal  ($\,\mathrm{V}$) ', fontsize=14)
            #     ax3.set_ylabel('MOKE signal ($\,\mathrm{V}$) ', fontsize=14)
            #     ax3.set_xlabel('time (ps)', fontsize=14)

            #     # delaystep = (np.max(delay)-np.min(delay))/10
            #     # xticks = np.arange(np.min(delay),np.max(delay)+delaystep,delaystep)
            #     tMin = 0
            #     tMax = 1000
            #     for ax in [ax1, ax2, ax3]:
            #         # ax.set_xticks(np.arange(0,1100,100))
            #         ax.set_xlim(tMin, tMax)
            #         ax.legend(loc=4)
            #         ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
            #         ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
            #     ax3.axhline(y=0, ls="--", color="grey")

            #     ax1.xaxis.set_ticks_position('top')
            #     ax2.set_xticklabels([])

            #     ax1.set_title(title, fontsize=16)

            #     plt.savefig(path+'resultsLoopwise.png', bbox_inches='tight', dpi=300)
            #     plt.savefig("MOKE_PumpProbe\\loopwise\\" +
            #                 str(int(identifier[line]))+'_overview.png', bbox_inches='tight', dpi=300)

            #     plt.show()

            #     h.writeListToFile(path+"dataBPlus.txt", u"time (ps)\t data plus loopwise", data_loop_field_up)
            #     h.writeListToFile(path+"dataBMinus.txt", u"time (ps)\t data plus loopwise", data_loop_field_down)
            # else:
            #     print("V=0")
            #     unique_delays = np.unique(data_raw[:, col_delay])
            #     loops = int(np.max(data_raw[:, col_loop]))
            #     n_delays = len(unique_delays)
            #     data_avg_result = np.zeros((n_delays, 2))
            #     dataLoopResult = np.zeros((n_delays, loops+1))

            #     s_pumped = data_raw[:, col_chopper] == 1.0
            #     s_not_pumped = data_raw[:, col_chopper] == 0.0

            #     for array in [data_avg_result, dataLoopResult]:
            #         array[:, 0] = unique_delays

            #     for i in range(len(unique_delays)):
            #         t = unique_delays[i]
            #         s_delay = data_raw[:, col_delay] == t
            #         s_pumped2 = s_delay & s_pumped
            #         s_not_pumped2 = s_delay & s_not_pumped
            #         data_avg_result[i, 1] = np.mean(data_raw[s_pumped2, col_diff_signal]) - \
            #             np.mean(data_raw[s_not_pumped2, col_diff_signal])

            #     f = plt.figure(figsize=(6, 0.68*6))
            #     gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1], wspace=0.0, hspace=0.0)
            #     ax1 = plt.subplot(gs[0])
            #     ax1.plot(data_avg_result[:, 0], data_avg_result[:, 1], '-',
            #              color="blue", lw=2,  label="reflectivity change")

            #     tMin = 0
            #     tMax = 1000
            #     for ax in [ax1]:
            #         ax.set_xticks(np.arange(0, 1100, 100))
            #         ax.set_xlim(tMin, tMax)
            #         ax.legend(loc=4)
            #         ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
            #         ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
            #         ax.axhline(y=0, ls="--", color="grey")

            #     ax1.set_title(title, fontsize=16)
            #     ax1.set_ylabel('balanced. signal  ($\,\mathrm{V}$) ', fontsize=14)
            #     ax1.set_xlabel('time (ps)', fontsize=14)

            #     plt.savefig(path+'overviewResults.png', bbox_inches='tight', dpi=300)
            #     plt.savefig("MOKE_PumpProbe\\overview\\" +
            #                 str(int(identifier[line]))+'_overview.png', bbox_inches='tight', dpi=300)

            #     plt.show()
            #     h.writeListToFile(path+"overviewData.txt", u"time (ps)\t refl. signal (V)", data_avg_result)

            #     for l in range(loops):
            #         sLoop = data_raw[:, col_loop] == l+1
            #         for i in range(len(unique_delays)):
            #             t = unique_delays[i]
            #             s_delay = data_raw[:, col_delay] == t
            #             dataLoopResult[i, l+1] = np.mean(data_raw[(s_pumped & sLoop) & s_delay, col_diff_signal]) - \
            #                 np.mean(data_raw[(s_not_pumped & sLoop) & s_delay, col_diff_signal])

            #     f = plt.figure(figsize=(6, 0.68*6))
            #     gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1], wspace=0.0, hspace=0.0)
            #     ax1 = plt.subplot(gs[0])

            #     cMap = plt.get_cmap('hsv')

            #     for l in range(loops):
            #         ax1.plot(dataLoopResult[:, 0], dataLoopResult[:, l+1],
            #                  '-',  color=cMap(l/loops), lw=1,  label=str(l+1))

            #     tMin = 0
            #     tMax = 1000
            #     for ax in [ax1]:
            #         ax.set_xticks(np.arange(0, 1100, 100))
            #         ax.set_xlim(tMin, tMax)
            #         ax.legend(loc=4)
            #         ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
            #         ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
            #         ax.axhline(y=0, ls="--", color="grey")

            #     ax1.set_title(title, fontsize=16)
            #     ax1.set_ylabel('balanced. signal  ($\,\mathrm{V}$) ', fontsize=14)
            #     ax1.set_xlabel('time (ps)', fontsize=14)

            #     #plt.savefig(path+'overviewResults.png',bbox_inches = 'tight', dpi = 300)
            #     #plt.savefig("data\\overview\\"+str(int(identifier[line]))+'_overview.png',bbox_inches = 'tight', dpi = 300)

            #     plt.savefig(path+'resultsLoopwise.png', bbox_inches='tight', dpi=300)
            #     plt.savefig("MOKE_PumpProbe\\loopwise\\" +
            #                 str(int(identifier[line]))+'_overview.png', bbox_inches='tight', dpi=300)
            #     plt.show()

            #     h.writeListToFile(path+"reflectivityLoopwise.txt",
            #                       u"time (ps)\t data reflectivity loopwise", dataLoopResult)
