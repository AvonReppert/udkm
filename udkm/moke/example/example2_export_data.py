# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:58:45 2022

@author: Udkm
"""

# import pickle
import udkm.moke.functions as moke
# import udkm.tools.functions as tools
import numpy as np
# import matplotlib.pyplot as plt
# import udkm.tools.colors as colors
# import matplotlib.gridspec as gridspec
import os


parameter_file_name = "parameters/parameters_example_1.txt"
line = 0


params = moke.get_scan_parameter(parameter_file_name, line)


directory = "data\\"+params["date"]+"_"+params["time"]
filename = "AllData_Reduced.txt"


parameter_list = []
for root, _, files in os.walk(directory):
    for file in files:
        if file == filename:
            path_combined = os.path.join(root, file)
            path = root.split("\\")
            if len(path) > 4:
                parameter_list.append([path[1].split("_")[0], path[1].split("_")[1], path[3],
                                       path[4]])
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
    path = directory+"\\Fluence\\"+pars[2]+"\\"+pars[3]+"\\"
    title = datepath + "    " + pars[2] + " mJ/cm^2 " + str(pars[3]) + " V"

    dataIn = np.genfromtxt(path+"AllData_Reduced.txt", skip_header=1)
    dataRaw = dataIn[np.isfinite(dataIn[:, 1]), :]

    if len(dataRaw) > 0:
        if voltages[line] > 0:
            #dataAveraged    = np.genfromtxt(path+"MOKE_Average.txt")

            uniqueDelays = np.unique(dataRaw[:, col_delay])
            loops = int(np.max(dataRaw[:, col_loop]))
            NDelays = len(uniqueDelays)
            dataNewBPlus = np.zeros((NDelays, 6))
            dataNewBMinus = np.zeros((NDelays, 6))
            dataNewResult = np.zeros((NDelays, 5))
            dataLoopBPlus = np.zeros((NDelays, loops+1))
            dataLoopBMinus = np.zeros((NDelays, loops+1))

            sPumped = dataRaw[:, col_chopper] == 1
            sNotPumped = dataRaw[:, col_chopper] == 0
            sBPlus = dataRaw[:, col_voltage] > 0
            sBMinus = dataRaw[:, col_voltage] <= 0

            for array in [dataNewBPlus, dataNewBMinus, dataNewResult, dataLoopBMinus, dataLoopBPlus]:
                array[:, 0] = uniqueDelays

            for i in range(len(uniqueDelays)):
                t = uniqueDelays[i]
                sTime = dataRaw[:, col_delay] == t
                sBPlusPumped = (sTime & sBPlus) & sPumped
                sBPlusNotPumped = (sTime & sBPlus) & sNotPumped

                sBMinusPumped = (sTime & sBMinus) & sPumped
                sBMinusNotPumped = (sTime & sBMinus) & sNotPumped
                for column in [col_diff_signal, col_diode_a, colDiodeB]:
                    dataNewBPlus[i, column] = np.mean(dataRaw[sBPlusPumped, column]) - \
                        np.mean(dataRaw[sBPlusNotPumped, column])
                    dataNewBMinus[i, column] = np.mean(dataRaw[sBMinusPumped, column]) - \
                        np.mean(dataRaw[sBMinusNotPumped, column])
            for array in [dataNewBPlus, dataNewBMinus]:
                array[:, 4] = array[:, col_diode_a] + array[:, colDiodeB]
                array[:, 5] = array[:, col_diode_a] - array[:, colDiodeB]

            dataNewResult[:, 1] = dataNewBPlus[:, col_diff_signal] + dataNewBMinus[:, col_diff_signal]
            dataNewResult[:, 2] = dataNewBPlus[:, col_diff_signal] - dataNewBMinus[:, col_diff_signal]
            dataNewResult[:, 3] = dataNewBPlus[:, col_diff_signal]
            dataNewResult[:, 4] = dataNewBMinus[:, col_diff_signal]
            # %%

            f = plt.figure(figsize=(8, 8))
            gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])

            ax1.plot(dataNewResult[:, 0], dataNewResult[:, 3], '-',   color="blue", lw=2,  label="B plus")
            ax1.plot(dataNewResult[:, 0], dataNewResult[:, 4], '-', color="green", lw=2,  label="B minus")
            ax2.plot(dataNewResult[:, 0], dataNewResult[:, 1], '-',  color="red",  lw=2,  label="Sum Signal")
            ax3.plot(dataNewResult[:, 0], dataNewResult[:, 2], '-',
                     color="green", lw=2,  label="Difference Signal")
            #ax3.plot(dataAveraged[:,0],dataAveraged[:,1],'-',  color = "black",lw=2,  label = "Average MOKE")

            ax1.set_ylabel(
                'single signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+/-}   = I_{+/-}^{1} - I_{+/-}^{0}}$ ', fontsize=14)
            ax2.set_ylabel('sum signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+}\,\, +\,\, S_{-}}$ ', fontsize=14)
            ax3.set_ylabel('MOKE signal ($\,\mathrm{V}$) \n $\mathrm{S_{+} \,\,-\,\, S_{-}}$', fontsize=14)
            ax3.set_xlabel('time (ps)', fontsize=14)

             # delaystep = (np.max(delay)-np.min(delay))/10
             # xticks = np.arange(np.min(delay),np.max(delay)+delaystep,delaystep)
             tMin = 0
              tMax = 1000
               for ax in [ax1, ax2, ax3]:
                    # ax.set_xticks(np.arange(0,1100,100))
                    ax.set_xlim(tMin, tMax)
                    ax.legend(loc=4)
                    ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
                    ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
                ax3.axhline(y=0, ls="--", color="grey")

                ax1.xaxis.set_ticks_position('top')
                ax2.set_xticklabels([])

                ax1.set_title(title, fontsize=16)

                plt.savefig(path+'overviewResults.png', bbox_inches='tight', dpi=300)
                plt.savefig("MOKE_PumpProbe\\overview\\" +
                            str(int(identifier[line]))+'_overview.png', bbox_inches='tight', dpi=300)

                plt.show()

                h.writeListToFile(
                    path+"overviewData.txt", u"time (ps)\t sum signal (V)\t MOKE (V)\t B plus signal (V) \t B minus signal (V)", dataNewResult)

                # %%
                for l in range(loops):
                    sLoop = dataRaw[:, col_loop] == l+1
                    for i in range(len(uniqueDelays)):
                        t = uniqueDelays[i]
                        sTime = dataRaw[:, col_delay] == t

                        sBPlusPumped = ((sLoop & sBPlus) & sPumped) & sTime
                        sBPlusNotPumped = ((sLoop & sBPlus) & sNotPumped) & sTime
                        sBMinusPumped = ((sLoop & sBMinus) & sPumped) & sTime
                        sBMinusNotPumped = ((sLoop & sBMinus) & sNotPumped) & sTime
                        column = col_diff_signal
                        dataLoopBPlus[i, l+1] = np.mean(dataRaw[sBPlusPumped, column]) - \
                            np.mean(dataRaw[sBPlusNotPumped, column])
                        dataLoopBMinus[i, l+1] = np.mean(dataRaw[sBMinusPumped, column]) - \
                            np.mean(dataRaw[sBMinusNotPumped, column])

                # %%
                cMap = plt.get_cmap('hsv')

                f = plt.figure(figsize=(8, 8))
                gs = gridspec.GridSpec(3, 1, width_ratios=[1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
                ax1 = plt.subplot(gs[0])
                ax2 = plt.subplot(gs[1])
                ax3 = plt.subplot(gs[2])

                for l in range(loops):
                    ax1.plot(dataLoopBPlus[:, 0], dataLoopBPlus[:, l+1], '-',
                             color=cMap(l/loops), lw=1,  label=str(l+1))
                    ax2.plot(dataLoopBMinus[:, 0], dataLoopBMinus[:, l+1],
                             '-',  color=cMap(l/loops), lw=1,  label=str(l+1))
                    ax3.plot(dataLoopBMinus[:, 0], dataLoopBPlus[:, l+1]-dataLoopBMinus[:, l+1],
                             '-',  color=cMap(l/loops), lw=1,  label=str(l+1))

                ax1.plot(dataNewResult[:, 0], dataNewResult[:, 3], '-',  color="blue", lw=2,  label="B plus avg")
                ax2.plot(dataNewResult[:, 0], dataNewResult[:, 4], '-',  color="green", lw=2,  label="B minus avg")
                ax3.plot(dataNewResult[:, 0], dataNewResult[:, 2], '-',  color="black", lw=2,  label="MOKE avg")

                ax1.set_ylabel('plus signal  ($\,\mathrm{V}$)', fontsize=14)
                ax2.set_ylabel('minus signal  ($\,\mathrm{V}$) ', fontsize=14)
                ax3.set_ylabel('MOKE signal ($\,\mathrm{V}$) ', fontsize=14)
                ax3.set_xlabel('time (ps)', fontsize=14)

                # delaystep = (np.max(delay)-np.min(delay))/10
                # xticks = np.arange(np.min(delay),np.max(delay)+delaystep,delaystep)
                tMin = 0
                tMax = 1000
                for ax in [ax1, ax2, ax3]:
                    # ax.set_xticks(np.arange(0,1100,100))
                    ax.set_xlim(tMin, tMax)
                    ax.legend(loc=4)
                    ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
                    ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
                ax3.axhline(y=0, ls="--", color="grey")

                ax1.xaxis.set_ticks_position('top')
                ax2.set_xticklabels([])

                ax1.set_title(title, fontsize=16)

                plt.savefig(path+'resultsLoopwise.png', bbox_inches='tight', dpi=300)
                plt.savefig("MOKE_PumpProbe\\loopwise\\" +
                            str(int(identifier[line]))+'_overview.png', bbox_inches='tight', dpi=300)

                plt.show()

                h.writeListToFile(path+"dataBPlus.txt", u"time (ps)\t data plus loopwise", dataLoopBPlus)
                h.writeListToFile(path+"dataBMinus.txt", u"time (ps)\t data plus loopwise", dataLoopBMinus)
            else:
                print("V=0")
                uniqueDelays = np.unique(dataRaw[:, col_delay])
                loops = int(np.max(dataRaw[:, col_loop]))
                NDelays = len(uniqueDelays)
                dataNewResult = np.zeros((NDelays, 2))
                dataLoopResult = np.zeros((NDelays, loops+1))

                sPumped = dataRaw[:, col_chopper] == 1.0
                sNotPumped = dataRaw[:, col_chopper] == 0.0

                for array in [dataNewResult, dataLoopResult]:
                    array[:, 0] = uniqueDelays

                for i in range(len(uniqueDelays)):
                    t = uniqueDelays[i]
                    sTime = dataRaw[:, col_delay] == t
                    sPumped2 = sTime & sPumped
                    sNotPumped2 = sTime & sNotPumped
                    dataNewResult[i, 1] = np.mean(dataRaw[sPumped2, col_diff_signal]) - \
                        np.mean(dataRaw[sNotPumped2, col_diff_signal])

                f = plt.figure(figsize=(6, 0.68*6))
                gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1], wspace=0.0, hspace=0.0)
                ax1 = plt.subplot(gs[0])
                ax1.plot(dataNewResult[:, 0], dataNewResult[:, 1], '-',
                         color="blue", lw=2,  label="reflectivity change")

                tMin = 0
                tMax = 1000
                for ax in [ax1]:
                    ax.set_xticks(np.arange(0, 1100, 100))
                    ax.set_xlim(tMin, tMax)
                    ax.legend(loc=4)
                    ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
                    ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
                    ax.axhline(y=0, ls="--", color="grey")

                ax1.set_title(title, fontsize=16)
                ax1.set_ylabel('balanced. signal  ($\,\mathrm{V}$) ', fontsize=14)
                ax1.set_xlabel('time (ps)', fontsize=14)

                plt.savefig(path+'overviewResults.png', bbox_inches='tight', dpi=300)
                plt.savefig("MOKE_PumpProbe\\overview\\" +
                            str(int(identifier[line]))+'_overview.png', bbox_inches='tight', dpi=300)

                plt.show()
                h.writeListToFile(path+"overviewData.txt", u"time (ps)\t refl. signal (V)", dataNewResult)

                for l in range(loops):
                    sLoop = dataRaw[:, col_loop] == l+1
                    for i in range(len(uniqueDelays)):
                        t = uniqueDelays[i]
                        sTime = dataRaw[:, col_delay] == t
                        dataLoopResult[i, l+1] = np.mean(dataRaw[(sPumped & sLoop) & sTime, col_diff_signal]) - \
                            np.mean(dataRaw[(sNotPumped & sLoop) & sTime, col_diff_signal])

                f = plt.figure(figsize=(6, 0.68*6))
                gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1], wspace=0.0, hspace=0.0)
                ax1 = plt.subplot(gs[0])

                cMap = plt.get_cmap('hsv')

                for l in range(loops):
                    ax1.plot(dataLoopResult[:, 0], dataLoopResult[:, l+1],
                             '-',  color=cMap(l/loops), lw=1,  label=str(l+1))

                tMin = 0
                tMax = 1000
                for ax in [ax1]:
                    ax.set_xticks(np.arange(0, 1100, 100))
                    ax.set_xlim(tMin, tMax)
                    ax.legend(loc=4)
                    ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
                    ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
                    ax.axhline(y=0, ls="--", color="grey")

                ax1.set_title(title, fontsize=16)
                ax1.set_ylabel('balanced. signal  ($\,\mathrm{V}$) ', fontsize=14)
                ax1.set_xlabel('time (ps)', fontsize=14)

                #plt.savefig(path+'overviewResults.png',bbox_inches = 'tight', dpi = 300)
                #plt.savefig("data\\overview\\"+str(int(identifier[line]))+'_overview.png',bbox_inches = 'tight', dpi = 300)

                plt.savefig(path+'resultsLoopwise.png', bbox_inches='tight', dpi=300)
                plt.savefig("MOKE_PumpProbe\\loopwise\\" +
                            str(int(identifier[line]))+'_overview.png', bbox_inches='tight', dpi=300)
                plt.show()

                h.writeListToFile(path+"reflectivityLoopwise.txt",
                                  u"time (ps)\t data reflectivity loopwise", dataLoopResult)
