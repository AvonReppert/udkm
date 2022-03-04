# -*- coding: utf-8 -*-

import numpy as np

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
