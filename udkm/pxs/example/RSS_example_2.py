# -*- coding: utf-8 -*-
'''
This script povides an example for the data evaluation using the rss scheme
at the pxs. We read in the raw data given by scattered intensity per detector
pixel and calculate the corresponding intensity in reciprocal space. The
intensity distribution along q_z corresponding to the out-of-plane direction
of the sample is analysed by a fit and center-of-mass after subtracting a
scattering background. The transient relative change of the peak parameters
are exported as final results of the measurement.

The usage of the script requires a certain data format:
1. column: delay_loop, 2. colum: delay, 3. column: theta_loop, 4. coulumn:
theta angle, 5. column: reference diode voltage, 6. column: reference diode
photons per second, 7. column: detector phontons per second, 8. column:
repeats of the scan due to low x-ray flux, 9. column: imgae number and
10. to last column: intensity per detector pixel.

Before running this script you should create a folder 'ReferenceData' with
a reference file. If the raw measurement data are imported from the measurement
folder to the data evaluation folder nothing more is necassary. If not you should
copy the measurement file 'scan_time' to its folder 'RawData/date/time'.
'''

import udkm.pxs.functions as pxs
# %%
'''In this part the measurement parameters of the 'measure'. measurement
of the reference file 'ref_file' that contains all measurements are written
into teh dictionarys 'measurement' and 'export'.
'''
ref_file = 'Reference'
measure = 1

measurement = pxs.get_scan_parameter(ref_file, measure)
export = pxs.get_export_parameter(measurement)
# %%
'''In this section you should write all analyse parameters into 'measurement'
and the plotting and export parameters into 'export'.
Mandotory keys of 'measurement' are:
'crystal_offset': (float) offset of the reference diode in [V] for normalization,
'crystal_threshold': (float) threshold in [V] to exclude scans with too low intensity,
'qz_border': (list) lower and upper boundary of relevant qz_range in [1/AA].

Optinal keys are:
'bad_loops': (list) of loops to be excluded (default: []), 'bad_pixels': (list) of pixels to be
excluded (default: []), 'time_zero': (float) delay of pump-probe overlap in [ps] (default: 0),
'centerpixel': (integer) pixel where theta-2theta-condition is fullfilled influences qz_value (default: 244),
'com_border': (list) lower and upper boundary of qz_range for com calculation in [1/AA] (default: 'qz_border').

Mandotory keys of 'export' are:


Optinal keys are:
'bool_refresh_data': (boolean) should the raw data are refreshed from the measurement folder
(default: False) and 'raw_data_path': (str) path of the measurement folder (default: test/test1/test2/),
'bool_plot2D_rocking_semilogy': (boolean) should the transient 2D rocking cruve plot be with
logarithmic norm (default: True).
'''
measurement['crystal_offset'] = 0.0211595  # could be also default value
measurement['crystal_threshold'] = 0.01  # could be also default value
if measurement['peak_number'] == 0:
    measurement['qz_border'] = [4.08, 4.24]
    measurement['com_border'] = [4.1, 4.22]
else:
    measurement['qz_border'] = [3.7, 4.05]
    measurement['com_border'] = [3.75, 4.0]


measurement['bool_substrate_back'] = False
export['delay_steps'] = [10, 40, 40]
export['bool_plot_fit'] = False
measurement['bool_double_peak'] = False


# %%
'''In this part the data analysis takes place.'''

scan = pxs.read_data_rss(measurement, export)

scan = pxs.get_rocking_rss(scan, export)

pxs.plot_rocking_overview(scan, export)

scan = pxs.get_background(scan, export)

scan = pxs.get_peak_dynamics(scan, export)

pxs.plot_transient_results(scan, export)
