# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 09:58:07 2022

@author: Aleks
"""
import udkm.moke.functions as moke 
import pandas as pd
import udkm.tools.functions as tools
import numpy as np
import matplotlib.pyplot as plt
import udkm.tools.colors as colors
import matplotlib.gridspec as gridspec

plt.style.use("udkm_base.mplstyle")
#print(moke.teststring)

parameter_file_name = "parameters/parameters_example_2.txt"


line = 0




def load_overview_data(params):
    data_path = params["date"]+"_"+params["time"]+"/Fluence/"+str(params["fluence"])+"/"+str(params["voltage"])+"/"
    prefix = "data/"
    file_name = "overviewData.txt"
    data = np.genfromtxt(prefix+data_path+file_name, comments="#")

    scan = {}
    scan["raw_delay"] = data[:,0]
    scan["delay"] = data[:,0]
    scan["sum"] = data[:,1]
    scan["moke"] = data[:,2]
    scan["field_up"] = data[:,3]
    scan["field_down"] = data[:,4]

    scan["date"] = params["date"]
    scan["time"] = params["time"]
    
    if params["bool_t0_shift"]:
        
        if "t0" in params:
            scan["delay"] = scan["raw_delay"]-params["t0"]
            scan["t0"] = params["t0"]
            print("t0 = " + str(scan["t0"]) + " ps")
        else:
            differences = np.abs(np.diff(scan[params["t0_column_name"]]))
            t0_index = tools.find(differences,np.max(differences))
            t0 = scan["raw_delay"][:-1][t0_index]
            scan["t0"] = t0
            print("t0 = " + str(scan["t0"]) + " ps")
            scan["delay"] = scan["raw_delay"]-t0
        
    return scan

params = moke.get_scan_parameter(parameter_file_name,line)
params["bool_t0_shift"] = False   
params["t0_column_name"] = "moke"
scan = load_overview_data(params)
#params["t0"] = 297

#%%    
plt.figure(figsize=(5.2, 5.2/0.68))
gs = gridspec.GridSpec(3, 1, wspace=0, hspace=0)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])
    
ax1.plot(scan["delay"],scan["field_up"], label ="field_up", color = colors.red_1,lw = 2)
ax1.plot(scan["delay"],scan["field_down"], label = "field_down", color = colors.blue_1,lw = 2)
ax2.plot(scan["delay"],scan["sum"], label = "sum", color = colors.orange_2,lw = 2)
ax3.plot(scan["delay"],scan["moke"], label = "moke", color = colors.grey_1,lw = 2)

ax1.set_ylabel('single signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+/-}   = I_{+/-}^{1} - I_{+/-}^{0}}$ ')
ax2.set_ylabel('sum signal  ($\,\mathrm{V}$) \n $\mathrm{S_{+}\,\, +\,\, S_{-}}$ ')
ax3.set_ylabel('MOKE signal ($\,\mathrm{V}$) \n $\mathrm{S_{+} \,\,-\,\, S_{-}}$')
ax3.set_xlabel('time (ps)')

tMin = 0
tMax = 1000
for ax in [ax1,ax2,ax3]:
     #ax.set_xticks(np.arange(0,1100,100))
     ax.set_xlim(tMin,tMax)
     ax.legend(loc = 4)
ax3.axhline(y = 0,ls = "--", color = "grey")   


   
ax1.xaxis.set_ticks_position('top')
ax2.set_xticklabels([])

#title = datepath+ "    "+str(powers[line])+ " mW    " + str(voltages[line]) +" V" 
#ax1.set_title(title) 



#def load_data_reflectivity(data_path, date, time, voltage, col_to_plot):
#    file = data_path+str(date)+"_"+tools.timestring(time)+"/Fluence/-1.0/"+str(voltage)+"/overviewData.txt"
#    data = np.genfromtxt(file, comments="#")
#    return(data[:, 0], data[:, col_to_plot])
#    