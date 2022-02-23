# -*- coding: utf-8 -*-
"""
This is an example skript for the data analysis in RSS mode at the pxs.
-- Further Description --
"""
import udkm.pxs.pxshelpers as pxs
import udkm.pxs.functions as fuc

# %%
'''Here, you can choose the measurement and get relevant parameters
to choose suitable analysis method parameters in the following.
'''
ref_file = 'Reference'
measure = 1

scan_dict = fuc.get_scan_parameter(ref_file, measure)
eval_dict = fuc.get_analysis_parameter(scan_dict)

eval_dict['refresh_data'] = False
eval_dict['raw_data_path'] = 'test/test2/test3'
eval_dict['bad_loops'] = []
eval_dict['bad_pixels'] = []
eval_dict['crystal_offset'] = 0.0211595
eval_dict['crystal_threshold'] = 0.01
eval_dict['bool_substrate_back'] = False
eval_dict['bool_plot_fit'] = False
eval_dict['delay_steps'] = [10, 40, 40]

if scan_dict['peak_number'] == 0:
    eval_dict['qz_borders'] = [4.08, 4.24]
else:
    eval_dict['qz_borders'] = [3.7, 4.05]

# %%
'''In this part the data analysis takes place.'''

omega, omega_deg, delays, intensity = pxs.read_data_rss(
    ref_file, measure, bad_loops, bad_pixels, crystal_off, treshold, time_zero, refresh, data_path)

rocking_curves = pxs.get_rocking_rss(ref_file, measure, omega, centerpixel, delays, intensity)

pxs.plot_rocking_overview(ref_file, measure, rocking_curves, qz_borders, delay_steps)

# %%

com, std, integral, unique_delays, ref_fit, intensity_peak, qz_roi = pxs.get_moments_back(
    rocking_curves, qz_borders, bool_substrate_back, [])

center_fit, width_fit, area_fit = pxs.get_fit_values(
    ref_file, measure, intensity_peak, qz_roi, unique_delays, ref_fit, bool_plot_fit)

transient_results = pxs.get_peak_dynamic(ref_file, measure, unique_delays, com,
                                         std, integral, center_fit, width_fit, area_fit)

pxs.plot_transient_results(ref_file, measure, transient_results, delay_steps)
