# -*- coding: utf-8 -*-
"""
This is an example skript for the data analysis in RSS mode at the pxs.
-- Further Description --
"""
import udkm.pxs.functions as fuc

# %%
'''Here, you can choose the measurement and get relevant parameters
to choose suitable analysis method parameters in the following.
'''
ref_file = 'Reference'
measure = 1

scan_dict = fuc.get_scan_parameter(ref_file, measure)
eval_dict = fuc.get_analysis_parameter(scan_dict)

eval_dict['bool_refresh_data'] = False
eval_dict['raw_data_path'] = 'test/test2/test3/'
eval_dict['bad_loops'] = []
eval_dict['bad_pixels'] = []
eval_dict['crystal_offset'] = 0.0211595
eval_dict['crystal_threshold'] = 0.01
eval_dict['bool_substrate_back'] = False
eval_dict['bool_plot_fit'] = False
eval_dict['delay_steps'] = [10, 40, 40]
eval_dict['centerpixel'] = 244

if scan_dict['peak_number'] == 0:
    eval_dict['qz_border'] = [4.08, 4.24]
else:
    eval_dict['qz_border'] = [3.7, 4.05]

# %%
'''In this part the data analysis takes place.'''

scan = fuc.read_data_rss(scan_dict, eval_dict)
# omega, omega_deg, delays, intensity = pxs.read_data_rss(
#    ref_file, measure, bad_loops, bad_pixels, crystal_off, treshold, time_zero, refresh, data_path)

scan_rocking = fuc.get_rocking_rss(scan, eval_dict)
# rocking_curves = pxs.get_rocking_rss(ref_file, measure, omega, centerpixel, delays, intensity)

fuc.plot_rocking_overview(scan_rocking, eval_dict)
# pxs.plot_rocking_overview(ref_file, measure, rocking_curves, qz_borders, delay_steps)
