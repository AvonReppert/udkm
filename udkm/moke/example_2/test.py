# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 10:51:31 2022

@author: matte
"""

import matplotlib.pyplot as plt
import numpy as np

import udkm.tools.colors as colors
import udkm.moke.functions_2 as moke
import udkm.tools.plot as plot

# %% Set parameters of the analysis

sample_name = 'P28b'
reference_file = 'fluence_series_2.txt'
analyze_line = 6
analyze_series = True

analysis_params = moke.set_analysis_params(sample_name)

# %% Analyze the measurements

if analyze_series:
    series = moke.load_series(reference_file, analysis_params)

    plt.figure()
    plot.list_plot(series['delay'], series['moke'], series['fluence'])

    plt.xlabel("delay (ps)")
    plt.ylabel("moke (V)")

    plt.axhline(y=0, ls="--", color=colors.grey_3)
    plt.tick_params(axis='both', which='both', direction="in")
    plt.tick_params(axis='y', left=True, right=True)
    plt.tick_params(axis='x', top=True, bottom=True)
    plt.legend(fontsize=7, loc=4,
               title=r"Fluence F $\left(\,\mathrm{\frac{mJ}{\,cm^2}}\right)$", handlelength=1, ncol=3)
    plt.savefig("exported_figures/"+str(sample_name)+"_1T_fluence.png")
    plt.show()
else:
    scan = moke.load_scan(reference_file, analysis_params, analyze_line)
    moke.plot_overview(reference_file, scan, analysis_params, analyze_line)