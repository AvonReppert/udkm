# -*- coding: utf-8 -*-
"""
This is an example skript for the data analysis in RSS mode at the pxs.

-- Further Description --

"""
import udkm.pxs.pxshelpers as pxs
import numpy as np
import matplotlib.pyplot as plt

#%%
'''Here, you can choose the measurement and get relevant parameters 
to choose suitable analysis method parameters in the following.
'''
ref_file  = 'Reference'
measure   = 1
peak      = pxs.read_param(ref_file, measure)[5]
refresh   = False
data_path = 'test'

'''Here, you can exclude bad loops and pixels from the further analysis.
Furthermore, the threshold may exclude scans with very low x-ray intensity.
'''
bad_loops   = []
bad_pixels  = []
crystal_off = 0.0211595
treshold    = 0.01
time_zero   = 0

'''Here, you should decide for a suiatble analysis method from the implemented
possible methods and give parameters for the data analysis like the relevant qz 
region of the Bragg peak and wether or not the shoulder of a substrate Bragg peak
should be taken into account during analysis.
'''
bool_substrate_back = False 
centerpixel = 244 
if peak == 0:
    qz_borders = [4.08, 4.24]
else:
    qz_borders = [3.7, 4.05]

'''Here, you should give plot parameters.'''
delay_steps   = [10,40,40]
bool_plot_fit = False

#%%
'''In this part the data analysis takes place.'''

omega, omega_deg, delays, intensity = pxs.read_data_rss(ref_file,measure,bad_loops,bad_pixels,crystal_off,treshold,time_zero,refresh,data_path)

rocking_curves = pxs.get_rocking_rss(ref_file,measure,omega,centerpixel,delays,intensity)

pxs.plot_rocking_overview(ref_file,measure,rocking_curves,qz_borders,delay_steps)

com, std, integral, unique_delays, ref_fit, intensity_peak, qz_roi = pxs.get_moments_back(rocking_curves,qz_borders,bool_substrate_back,[])

center_fit, width_fit, area_fit = pxs.get_fit_values(ref_file,measure,intensity_peak,qz_roi,unique_delays,ref_fit,bool_plot_fit)

transient_results = pxs.get_peak_dynamic(ref_file,measure,unique_delays,com,std,integral,center_fit,width_fit,area_fit)

pxs.plot_transient_results(ref_file,measure,transient_results,delay_steps)