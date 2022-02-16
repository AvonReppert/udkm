# -*- coding: utf-8 -*-
"""
This is an example skript for the data analysis in RSS mode at the pxs.

-- Further Description --

"""

import udkm.pxs.pxshelpers as pxs
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import lmfit as lm
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter



#%%
ref_file      = 'DyOverview'
number        = 2
bad_loops     = []
crystal_off   = 0.0211595
crystal_tresh = 0.01
t0            = 0
delay_first   = True
bad_pixels    = []

qzMinCOM = 2.15
qzMaxCOM = 2.27
qzMinFit = 2.05
qzMaxFit = 2.37
peak_order = 1

t_split = 100
plot_logy = False
plot_fit = False
save_plot_fit = False
t_lim  = [-10,100,2000]
width_ratio = [2,1]
ylim = [-0.5,7.2,-10,100,-40,10]
t_ticks1 = [0,20,40,60,80]
t_ticks2 = [500,1000,1500]

#%%
''' Read In the data and return the omega axis (radians), delays (including various loops)
and the crystal-normed intensity as function of omega and pixels for all delays '''
omega, delays, intensity = ReadDataOneAngle(ref_file,number,bad_loops,
                                            crystal_off,crystal_tresh,t0,delay_first)

''' Transformation from omega-pixel space in the reciprocal space '''
thetaAxis, qzAxis = QzAxisOneAngle(ref_file,number,omega,intensity)

''' Get the rocking curves for the unique delays including bad pixels with first column = delays and first row = qz axis'''
rockingCurve = GetRockingCurvesOneAngle(ref_file,number,omega,delays,intensity,bad_pixels)

''' Plot the time-resolved rocking curves as lines '''
PlotRockingCurve1DUXRD(ref_file,number,rockingCurve,plot_logy)

''' Plot the time-resolved rocking curves as map '''
PlotRockingCurve2DUXRD(ref_file,number,rockingCurve,t_split)

''' Calculate the moments of the rocking curves for each delay '''
''' This includes a background correction of COM by a refernece Gaussian Fit with background '''
COMt, STDt, Integralt, FitRef, uniqueDelays = GetMoments(rockingCurve,qzMinCOM,qzMaxCOM)

''' Get the Gaussian (with background) fit of the peak '''
centerFit,  widthFit, areaFit, uniqueDelays = getFitvalues(ref_file,number,rockingCurve,qzMinFit,qzMaxFit,FitRef,plot_fit,plot_logy,save_plot_fit)

''' Calculate the relative change of the moments of the peak as function of the delay '''
strainCOM, STDrelative, Integralrelative = getPeakDynamic(uniqueDelays,COMt,STDt,Integralt,peak_order)

''' Calculate the relative change of the fit values of the peak as function of the delay '''
strainFit, widthFitrelative, areaFitrelative = getPeakDynamic(uniqueDelays,centerFit,widthFit,areaFit,peak_order)

''' Summarize it to a array '''
exportedResults = np.zeros((len(uniqueDelays),7))
exportedResults[:,0] = uniqueDelays
exportedResults[:,1] = strainCOM
exportedResults[:,2] = STDrelative
exportedResults[:,3] = Integralrelative
exportedResults[:,4] = strainFit
exportedResults[:,5] = widthFitrelative
exportedResults[:,6] = areaFitrelative

''' Save the transient changes of teh peak properties '''
exportPeakDynamic(ref_file,number,uniqueDelays,strainFit,widthFitrelative,areaFitrelative,strainCOM,STDrelative,Integralrelative)
#%%
''' Plot the transient change of all quantities '''
PlotTransientResults(ref_file,number,exportedResults,t_lim,width_ratio,ylim,t_ticks1,t_ticks2)

