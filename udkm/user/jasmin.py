# -*- coding: utf-8 -*-

import numpy as np
import udkm.tools.functions as tools

teststring = "Successfully loaded udkm.user.jasmin"


def smooth_curve(time, trace, steps):
    '''smooth_curve returns a smoothed array of the y data (trace) and the corresponding x axis. '''
    traceNew = np.zeros(int(len(trace)/steps))
    timeNew = np.zeros(int(len(trace)/steps))
    for n in range(int(len(trace)/steps)-1):
        traceNew[n] = np.mean(trace[n*steps:(n+1)*steps])
        timeNew[n] = time[(n*steps)+round(1/2*steps)]
    timeNew[-1] = timeNew[-2]
    return(timeNew, traceNew)


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
