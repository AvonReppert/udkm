# -*- coding: utf-8 -*-

import numpy as np
import udkm.tools.helpers as h

import pandas as pd
import udkm.tools.functions as tools

teststring = "Successfully loaded udkm.moke.functions"

# initialize some useful functions
t0 = 0    # Estimated value of t0
data_path = "data/"
export_path = "results/"


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



def get_scan_parameter(parameter_file_name, line):
    params = {'line': line}
    param_file = pd.read_csv(parameter_file_name, delimiter="\t", header=0, comment="#")
    header = list(param_file.columns.values)
    
    if not('fluence') in header:
        params['fluence'] = -1.0
    #initialize default values
    params["pump_angle"] = 0
    params["bool_t0_shift"] = False
    params["t0_column_name"] = "moke"
        
    for i, entry in enumerate(header):
        if entry == 'date' or entry == 'time':
            params[entry] = tools.timestring(int(param_file[entry][line]))
        else:
            params[entry] = param_file[entry][line]
    return params

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