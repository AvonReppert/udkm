#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 17:22:19 2022

@author: felix
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter
# from scipy.optimize import curve_fit
# from lmfit import minimize, Parameters, report_fit


# %% Colormap for heatmaps

def make_cmap(colors, position=None, bit=False):
    '''
    make_cmap takes a list of tuples which contain RGB values. The RGB
    values may either be in 8-bit [0 to 255] (in which bit must be set to
    True when called) or arithmetic [0 to 1] (default). make_cmap returns
    a cmap with equally spaced colors.
    Arrange your tuples so that the first color is the lowest value for the
    colorbar and the last is the highest.
    position contains values from 0 to 1 to dictate the location of each color.
    '''
    import matplotlib as mpl
    import numpy as np
    bit_rgb = np.linspace(0, 1, 256)
    if position == None:
        position = np.linspace(0, 1, len(colors))
    else:
        if len(position) != len(colors):
            sys.exit("position length must be the same as colors")
        elif position[0] != 0 or position[-1] != 1:
            sys.exit("position must start with 0 and end with 1")
    if bit:
        for i in range(len(colors)):
            colors[i] = (bit_rgb[colors[i][0]],
                         bit_rgb[colors[i][1]],
                         bit_rgb[colors[i][2]])
    cdict = {'red': [], 'green': [], 'blue': []}
    for pos, color in zip(position, colors):
        cdict['red'].append((pos, color[0], color[0]))
        cdict['green'].append((pos, color[1], color[1]))
        cdict['blue'].append((pos, color[2], color[2]))

    cmap = mpl.colors.LinearSegmentedColormap('my_colormap', cdict, 256)
    return cmap


colors = [(10/256, 72/256, 117/256),
          (15/256, 118/256, 187/256),
          (1, 1, 1),
          (242/255, 168/255, 70/255),
          (217/255, 123/255, 0/255)
          ]

#    colors = [(57/256, 60/256, 151/256),(1, 1, 1), (256/256, 151/256, 57/256), (151/256, 57/256, 60/256)]
position = [0, 0.15, 0.5, 0.85, 1]
cm = make_cmap(colors, position=position)

FelixRed = '#D7181E'
FelixBlue = '#0F76BB'
FelixOrange = '#EF951E'


# %% Importing Data and changing data type to float

data = pd.read_csv(r'//141.89.115.120/boye1/Auswertung/NbO2/20220824_134331/Spectro1_PumpProbe_AveragedData.txt', #windows
                   sep='\t', index_col=0, skipinitialspace=True, skiprows=1)
        #mac pd.read_csv(r'/Volumes/boye1/Auswertung/NbO2/20220824_113906/Spectro1_PumpProbe_AveragedData.txt',
        
data = data.astype(float)
data.index = data.index.astype(float)
data.columns = data.columns.astype(float)



prozent_trans = 0.2 #in Prozent 0.2=20%


# Plotting raw data
plt.figure(1)
plt.gcf().clear()
plt.pcolormesh(data.columns.astype(float), data.index,
               data, vmin=-prozent_trans, vmax=+prozent_trans, cmap=cm)
plt.vlines(x=1118.896, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black')
plt.vlines(x=1203.946, ymin=-2010, ymax=-2005, color= 'black')
plt.vlines(x=1301.711, ymin=-2010, ymax=-2005, color= 'black')
plt.vlines(x=1373.166, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black')
plt.vlines(x=2065.869, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black') 
plt.vlines(x=2135.707, ymin=-2010, ymax=-2005, color= 'black')  # passt
plt.vlines(x=2388.901, ymin=-2010, ymax=-2005, color= 'black')
plt.vlines(x=2433.108, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black')
plt.vlines(x=2502.528, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black')
plt.colorbar(label='Relative change of Transmission [%]')

plt.xlim(1050, 2505)
plt.ylim(-2010, -2006)

plt.title('Raw data - 50mW, small delays')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Stage Delay [ps]')
plt.savefig('raw_data.png')

#%%
# Defining region in around time zero for local extrema
data_cut = data.loc[-2010:-2006, :]
length = len(data_cut)
local_min = argrelextrema(data_cut.values, np.less_equal, order=50)

# Smoothing the data for more accurate position of extrema
data_cut_smooth = data_cut.apply(savgol_filter,  window_length=length-1, polyorder=2) #60 auf 39 geändert
#%%
# Plotting smoothed data
plt.figure(2)
plt.gcf().clear()
plt.pcolormesh(data_cut_smooth.columns.astype(
    float), data_cut_smooth.index, data_cut_smooth, vmin=-0.01, vmax=0.01, cmap=cm)
plt.vlines(x=1118.896, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black')
plt.vlines(x=1203.946, ymin=-2010, ymax=-2005, color= 'black')
plt.vlines(x=1301.711, ymin=-2010, ymax=-2005, color= 'black')
plt.vlines(x=1373.166, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black')
plt.vlines(x=2065.869, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black') 
plt.vlines(x=2135.707, ymin=-2010, ymax=-2005, color= 'black')  # passt
plt.vlines(x=2388.901, ymin=-2010, ymax=-2005, color= 'black')
plt.vlines(x=2433.108, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black')
plt.vlines(x=2502.528, ymin=-2010, ymax=-2005, linestyles='dotted', color= 'black')
plt.colorbar(label='Relative change of Transmission [%]')

plt.xlim(1050, 2505)
plt.ylim(-2010, -2006)

plt.title('Smooth data - 50mW, small delays')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Stage Delay [ps]')
plt.savefig('smooth_data.png')


# Dividing data into regions with positive or negative transmission changes #???
data_area1 = data_cut_smooth.loc[:, 1053.262:1118.896] #black
data_area2 = data_cut_smooth.loc[:, 1203.946:1301.711] #blue
data_area3 = data_cut_smooth.loc[:, 1373.166:2148.394] #black
data_area4 = data_cut_smooth.loc[:, 2168.436:2388.901] #blue
data_area5 = data_cut_smooth.loc[:, 2401.534:2502.528] #black


# Finding extrema in the respective regions and combining to one array #???
ext_area1 = data_area1.idxmin()
ext_area2 = data_area2.idxmax()
ext_area3 = data_area3.idxmin()
ext_area4 = data_area4.idxmax()
ext_area5 = data_area4.idxmin()


ext_all = pd.concat([ext_area1, ext_area2, ext_area3,
                    ext_area4, ext_area5])  # ???


# Fitting the extrema curve with np.polyfit
poly = np.polyfit(ext_all.index, ext_all.values, 3)

# Defining function with obtained parameters


def fit(x, poly):
    return poly[0]*x**3+poly[1]*x**2+poly[2]*x**1+poly[3]*x**0#+poly[4]*x**1+poly[5]


wave = data.columns


# Plotting position of extrema together with fit
plt.figure(3)
plt.gcf().clear()
plt.plot(wave, fit(wave, poly))
plt.plot(ext_all, '.')
plt.xlim(1050, 2505)
plt.ylim(-2010, -2005)

plt.title('Position of extrema together with fit - 50mW, small delays')
plt.xlabel('Wavelength [nm]')
plt.ylabel('Stage Delay [ps]')
plt.savefig('extrema_fit.png')

# %% Chirp correcting the data
FROGmin = min(fit(wave, poly))  # tmin according to FROG curve
FROGmax = max(fit(wave, poly))  # tmin according to FROG curve
# mintime = min(time)            #Minimum of the time axis
# maxtime = max(time)            #Maximum of the time axis

dt = np.mean(np.diff(data.index))  # Average timesteps
n = len(data.index)  # Number of timesteps
dn = int(round((FROGmax-FROGmin)/dt))  # Number of discarded timesteps

# New time axis (had to add the factor since the chirp was measured with 800 nm and thus with a different time zero)
time_new = data.index[0:(n-dn)]-FROGmin

# %%
data_new = np.zeros([len(time_new), len(wave)])
for i in range(len(wave)):
    data_new[0:len(data.values[int(round((fit(wave[i], poly)-FROGmin)/dt)):n-dn + int(round((fit(wave[i], poly)-FROGmin)/dt)), i]),
             i] = data.values[int(round((fit(wave[i], poly)-FROGmin)/dt)):n-dn + int(round((fit(wave[i], poly)-FROGmin)/dt)), i]

# %%
plt.figure(4)
plt.gcf().clear()
plt.pcolormesh(wave, time_new, data_new, vmin=-0.01, vmax=0.01, cmap=cm)
plt.ylim(min(time_new), max(time_new))
#plt.hlines(y = 0, xmin = 1050, xmax = 2505, color='gray')

plt.xlim(1050, 2505)
plt.ylim(min(time_new), max(time_new)) #??

plt.title('Chirp corrected data - 50mW, small delays')
plt.xlabel('Wavelength [nm]')
plt.ylabel('New time ???')
plt.savefig('chirp_corrected.png')
# plt.colorbar()


# %% Writing corrected data in file
N = np.zeros([len(time_new)+1, 1+len(wave)])
N[1:, 0] = time_new
N[0, 1:] = wave
N[1:, 1:] = data_new
np.savetxt("Spectro1_PumpProbe_AveragedData_ChirpCorrected.txt",
           N, delimiter='\t')


# %% Transienten

ind = data.index

data_time = data
data_time_id = data
data_time_sig = data

time_zero = 2008
time_zero_id = 2008.535 #selbst wählen
time_zero_sig = 2008.687 #für Idler und Signal unterschiedlich

time_delay = ind + time_zero
time_delay_id = ind + time_zero_id
time_delay_sig = ind + time_zero_sig

data_time = data_time.set_index(time_delay)
data_time_id = data_time_id.set_index(time_delay_id)
data_time_sig = data_time_sig.set_index(time_delay_sig)

idler = 1200
signal = 2300
bereich = 200

idler_min = idler - bereich
idler_max = idler +  bereich

signal_min = signal - bereich
signal_max = signal + bereich

data_id = data_time_id.loc[:, idler_min:idler_max] #oder mit data_time für Werte von -3 bis 0 etc.
data_sig = data_time_sig.loc[:, signal_min:signal_max]

mean_id = data_id.mean(axis=1) #wird nach Reihen gemittelt - also für einen Stage Delay über die alle Wellenlängen
mean_sig = data_sig.mean(axis=1)

min_id = min(mean_id)
min_sig = min(mean_sig)

norm_id = -(mean_id / min_id)
norm_sig = -(mean_sig / min_sig)



plt.figure(5)
plt.gcf().clear()
plt.plot(mean_id, color= 'tab:red', label='Idler')
plt.plot(mean_sig, color= 'tab:blue', label='Signal')
plt.xlim(min(data_id.index), max(data_id.index))
plt.ylim(min(mean_sig)-0.01, max(mean_sig)+0.01) #von -2010 bis -2005

plt.title('Plot of Transient - 50mW, small delays')
plt.xlabel('Stage Delay [ps]')
plt.ylabel('Relative change of Transmission [%]')
plt.legend(loc='best')
plt.savefig('transient.png')



plt.figure(6)
plt.gcf().clear()
plt.plot(norm_id, color= 'tab:red', label='Idler')
plt.plot(norm_sig, color= 'tab:blue', label='Signal')
plt.xlim(min(data_id.index), max(data_id.index))
plt.ylim(min(norm_sig)-0.01, max(norm_sig)+0.01) #von -2010 bis -2005

plt.title('Plot of normalized Transient - 50mW, small delays')
plt.xlabel('Stage Delay [ps]')
plt.ylabel('Relative change of normalized Transmission [%]')
plt.legend(loc='best')
plt.savefig('transient_norm_min_rightT0.png')







