# -*- coding: utf-8 -*-
"""
This skript contains all basic functions used to evaluate pxs x-ray diffraction data.

-- Further information --

"""
import udkm.tools.functions as tools
import udkm.tools.colors as color
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import lmfit as lm
import matplotlib.gridspec as gridspec
import shutil
import pandas as pd

PXS_WL = 1.5418             # Cu K-alpha wavelength in [Ang]
PXS_K = 2*np.pi/PXS_WL      # absolute value of the incident k-vector in [1/Ang]
PIXELSIZE = 0.172e-3        # size of pilaus pixel in [m]


def get_scan_parameter(ref_file, number):
    '''This function writes the measurement parameters of the 'number'. measurement
    from the reference file 'ref_file' into a dictionary and returns it. Necessary
    entrys of the reference file are: 'identifier': combination of date and time,
    'date': date of measurement, 'time': time of measurement, 'temperature': start
    temperature in [K], 'fluence' - incident excitation fluence in [mJ/cm2],
    'time_zero': delay of pump-probe overlap in [ps], 'distance': distance between
    sample and detector in [m].


    Parameters
    ----------
    ref_file : str
        File name of the reference file storing information about measurement.
    number : int
        The number of the measurement equivalent to row in the reference file.

    Returns
    -------
    scan_parameters : dictionary
        Contains the parameters of the measurement. Always included keys are: 'identifier':
        combination of date and time,'date': date of measurement, 'time': time of
        measurement, 'temperature': start temperature in [K], 'fluence' - incident
        excitation fluence in [mJ/cm2], 'time_zero': delay of pump-probe overlap in [ps],
        'distance': distance between sample and detector in [m].
        Optinal keys are: 'power': laser power in [mW], 'spot_size_x': FWHM of laser spot
        along x-direction in [mum], 'spot_size_y': FWHM of laser spot along
        y-direction in [mum], 'peak': name of material of measured Bragg peak,
        'magnetic_field': applied magnetic field in [T], 'wavelength': excitation
        wavelength in [nm], 'double_pulse': was it 'first', 'second' or 'both' excitation pulses,
        'z_pos': z-position on the sample in [mm], 'y_pos': y-position on the
        sample in [mm], 'pulse_length': length of the pump-pulse in [ps], 'colinear_exc':
        wether or not pump and probe were collinear.

    Example
    -------
    >>> get_scan_parameter('reference.txt', 1) = {'number': 1,
                                                  'identifier': '20211123174703',
                                                  'date': '20211123',
                                                  'time': '183037',
                                                  'temperature': 300,
                                                  'fluence': 8.0,
                                                  'distance': 0.7,
                                                  'time_zero': 0.0,
                                                  'power': 110,
                                                  'spot_size_x': 960,
                                                  'spot_size_y': 960,
                                                  'double_pulse': first,
                                                  'pulse_length': 0.1,
                                                  'colinear_exc': no,
                                                  'wavelength': 800,
                                                  'peak': 'W',
                                                  'magnetic_field': 0.0,
                                                  'z_pos': -4.52,
                                                  'y_pos': 2.25}

    '''
    param_file = pd.read_csv(r'ReferenceData/'+str(ref_file)+'.txt', delimiter="\t", header=0, comment="#")
    header = list(param_file.columns.values)
    params = {'number': number}
    for i, entry in enumerate(header):
        if entry == 'identifier' or entry == 'date' or entry == 'time':
            params[entry] = tools.timestring(int(param_file[entry][number]))
        else:
            params[entry] = param_file[entry][number]
    return params


def get_export_parameter(params):
    '''This function creates a dictionary that contains parameters for exporting
    the analysis results. This dictionary contains the key words: 'plot_title':
    string that contains the measurement number, its date, time, temperature,
    fluence and material name to link plots to the measurement, 'save_path': string for a folder
    structure of date and time to save figures and results and 'file_name': string
    for suitable file names of exported results. Additionally, adjustable plot parameters
    are initialised by their default values.

    Parameters
    ----------
    params : dictionary
        This dictionary contains all parameters of the measurement from teh reference file.
        This function uses the keys: 'number', 'identifier', 'date', 'time', 'temperature',
        'fluence' and 'peak'.

    Returns
    -------
    export : dictionary
        The entrees of this dictionary are used to export and plot the analysed data.
        The adjustable keys and their initialised default values are: 'bool_refresh_data'
        and 'raw_data_path': should the raw data from the measurement folder are loaded
        (default: False), 'bool_plot_fit': should the fit of the rocking curve are plotted
        (default: False), 'delay_steps': xtick step for first delay range, delay of axis break and
        xtick step for long delays (default: [10, 100, 200]), 'bool_plot_2D_rocking_semilogy':
        logarithmic or linear norm of transient rocking curves (default: True), 'bool_plot_1D_rocking_semilogy':
        logarithmic or linear norm of fitted rocking curve (default: False), 'intensity_range_2D':
        lower and upper limit of plotted normed transient intensity (default: [0.01, 1]),
        'intensity_range_1D': lower and upper limit of plotted normed rocking curve (default: [-0.01, 1]).

    '''
    export = {}
    export['plot_title'] = str(params['number']) + ' | ' + params['date'] + \
        ' | ' + params['time'] + ' | ' + str(params['temperature']) + r'$\,$K' + ' | ' + \
        str(params['fluence']) + r'$\,$mJ/cm$^2$' + ' | ' + params['peak']
    export['save_path'] = params['date'] + '/' + params['time']
    export['file_name'] = params['identifier']

    export['bool_refresh_data'] = False
    export['raw_data_path'] = 'test/test2/test3/'
    export['bool_plot_fit'] = False
    export['delay_steps'] = [10, 100, 200]
    export['bool_plot_2D_rocking_semilogy'] = True
    export['bool_plot_1D_rocking_semilogy'] = False
    export['intensity_range_2D'] = [0.01, 1]
    export['intensity_range_1D'] = [-0.01, 1]
    return export


def calc_theta_value(omega_rad, distance, pixel, centerpixel):
    '''This function calculates the corresponding diffraction angle 'theta' of a
    'pixel' of the detector in respect to its 'centerpixel'. The 'distance' between
    sample and detector determines the angular space fraction covered by the pixel.

    Parameters
    ----------
    omega_rad : 0D array
        the incident angle of x-ray beam in [rad]
    distance : float
        distance between sample and detector in [m]
    pixel : 1D array
        pixel or pixels for which the theta angle should be calculated
    centerpixel : integer
        pixel where Bragg condition 'omega' = 'theta' is fulfilled

    Returns
    -------
    theta_rad : 1D array
        theta angle of the pixel or pixels in [rad]

    Example
    --------
    >>> calc_theta_value(0.492, 0.700, 244, 244)[0] = 0.492

    '''
    delta_theta = np.arctan(PIXELSIZE / distance)
    theta_rad = omega_rad + delta_theta * (pixel - centerpixel)
    return theta_rad


def calc_qx_qz(omega_rad, theta_rad):
    '''This function calculates the qx-qz coordinate in reciprocal space for a
    given incident and diffraction angle 'omega_rad' and 'theta_rad',respectively.

    Parameters
    ----------
    omega_rad : 0D array
        the incident angle of x-ray beam in [rad]
    theta_rad : 1D array
        the diffraction angle or angles of x-ray beam in [rad]

    Returns
    -------
    qx_val : 1D array
        in-plane reciprocal space coordinate in [1/Ang]
    qz_val : 1D array
        out-off-plane reciprocal space coordinate in [1/Ang]

    Example
    --------
    >>> calc_qx_qz(0.492,0.492)[0] = 0.0
    >>> calc_qx_qz(0.492,0.492)[1] = 3.85

    '''
    qx_val = PXS_K * (np.cos(theta_rad) - np.cos(omega_rad))
    qz_val = PXS_K * (np.sin(theta_rad) + np.sin(omega_rad))
    return qx_val, qz_val


def get_background(scan_rocking):
    '''This function subtracts the scattering background from the averaged rocking curve
    before t0 and thereby extracts the intensity distribution associated with the Bragg peak.
    To avoid negative intensity after subtracting the background fit the minimum is set to zero.

    Parameters
    ----------
    scan_rocking : dictionary
        This dictionary contains the raw and analysed measurement data. This function uses
        the keys: 'unique_delays': measured different delays, 'qz_axis': relevant range in reciprocal space,
        'intensity_rock': intensity for each delay and 'qz_axis' loop-wise averaged.

    Returns
    -------
    scan_rocking : dictionary
        The function added the keys: 'intensity_peak': background-free time-dependent intensity
        of the Bragg peak, 'ref_fit': lm model object that contains the peak properties before
        excitation as reference for time-dependent fit.

    '''
    print('Subtracting linear scattering background.')
    intensity_t0 = np.mean(scan_rocking['intensity_rock'][scan_rocking['unique_delays'] < 0, :], 0)
    com_t0, std_t0, integral_t0 = tools.calc_moments(scan_rocking['qz_axis'], intensity_t0)
    intensity_peak = scan_rocking['intensity_rock']
    ref_fit = gaussian_back_fit(com_t0, std_t0, np.max(intensity_t0),
                                scan_rocking['qz_axis'], intensity_t0)
    for i in range(np.size(scan_rocking['unique_delays'])):
        intensity_peak[i, :] = scan_rocking['intensity_rock'][i, :] - \
            (scan_rocking['qz_axis']*ref_fit.values["slope"]+ref_fit.values["intercept"])
        intensity_peak[i, :] = intensity_peak[i, :] - np.min(intensity_peak[i, :])
    scan_rocking['intensity_peak'] = intensity_peak
    scan_rocking['ref_fit'] = ref_fit
    return scan_rocking


def get_peak_dynamics(scan_back, export):
    '''This function uses the extracted intensity distribution along qz associated with the Bragg
    peak to determine its time-dependent position, width and amplitude using a fit and a center-of-mass
    analysis. Finally, the transient relative change in respect to the averaged value before excitation
    is determined and exportedto a folder 'exportedResults'. The key 'com_border' of teh dictionary 'scan_back'
    provides the option to narrow the evaluted qz-range for the center-of-mass analysis. The key 'bool_plot_fit'
    of the dictionary 'export' enables plotting the comparison of teh Bragg peak fit and the actual rocking curve.

    Parameters
    ----------
    scan_back : dictionary
        This dictionary contains the time-dependent intensity distribution along qz associated
        with the Bragg peak. This function uses the keys: 'unique_delays': measured different delays,
        'qz_axis': relevant range along qz containing the Bragg peak, 'com_border': lower and upper border
        along qz for center-of-mass evaluation (default: full 'qz_axis' range), 'intensity_peak': background-
        free time-dependent intensity of the Bragg peak, 'ref_fit': lm model object that contains the peak
        properties before excitation as reference for time-dependent fit, 'temperature': temperature of the
        measurement in [K] and 'fluence': excitation fluence in [mJ/cm2].
    export : dictionary
        This dictionary contains the parameters to plot and export the calculated results. This
        function uses the keys: 'bool_plot_fit': whether or not the time-dependent fit of the
        rocking curve is plotted (default: False).

    Returns
    -------
    scan_back : dictionary
        The function added the key: 'com': time-dependent position, 'std': time-dependent width,
        'integral': time-dependent amplitude, 'center_fit': time-dependent position, 'width_fit':
        time-dependent width, 'area_fit': time-dependent amplitude, 'transient_results': two-dimensional
        array that contains the relative change of the peak properties for both evaluations for each
        different measured delay.

    '''
    com = np.zeros(len(scan_back['unique_delays']))
    std = np.zeros(len(scan_back['unique_delays']))
    integral = np.zeros(len(scan_back['unique_delays']))
    center_fit = np.zeros(len(scan_back['unique_delays']))
    width_fit = np.zeros(len(scan_back['unique_delays']))
    area_fit = np.zeros(len(scan_back['unique_delays']))

    for i in range(len(scan_back['unique_delays'])):
        if 'com_border' in scan_back:
            s_com_qz = (scan_back['qz_axis'] >= scan_back['com_border'][0]) & (
                scan_back['qz_axis'] <= scan_back['com_border'][1])
            com[i], std[i], integral[i] = tools.calc_moments(
                scan_back['qz_axis'][s_com_qz], scan_back['intensity_peak'][i, s_com_qz])
        else:
            com[i], std[i], integral[i] = tools.calc_moments(scan_back['qz_axis'], scan_back['intensity_peak'][i, :])
        fit = gaussian_fit(scan_back['ref_fit'].values["g_center"], scan_back['ref_fit'].values["g_sigma"],
                           scan_back['ref_fit'].values["g_amplitude"],
                           scan_back['qz_axis'], scan_back['intensity_peak'][i, :])
        center_fit[i] = fit.values["g_center"]
        width_fit[i] = fit.values["g_sigma"]
        area_fit[i] = fit.values["g_amplitude"]
        if export['bool_plot_fit']:
            plot_peak_fit(export, scan_back['qz_axis'], fit, scan_back['ref_fit'], scan_back['unique_delays'][i])

    s_before_t0 = scan_back['unique_delays'] < 0
    com_relative = -1 * tools.rel_change(com, np.mean(com[s_before_t0]))
    std_relative = tools.rel_change(std, np.mean(std[s_before_t0]))
    integral_relative = tools.rel_change(integral, np.mean(integral[s_before_t0]))
    center_relative = -1 * tools.rel_change(center_fit, np.mean(center_fit[s_before_t0]))
    width_relative = tools.rel_change(width_fit, np.mean(width_fit[s_before_t0]))
    area_relative = tools.rel_change(area_fit, np.mean(area_fit[s_before_t0]))

    transient_results = np.zeros((len(scan_back['unique_delays']), 9))
    transient_results[:, 0] = scan_back['temperature']*np.ones(len(scan_back['unique_delays']))
    transient_results[:, 1] = scan_back['fluence']*np.ones(len(scan_back['unique_delays']))
    transient_results[:, 2] = scan_back['unique_delays']
    transient_results[:, 3] = 1e3*com_relative
    transient_results[:, 4] = 1e2*std_relative
    transient_results[:, 5] = 1e2*integral_relative
    transient_results[:, 6] = 1e3*center_relative
    transient_results[:, 7] = 1e2*width_relative
    transient_results[:, 8] = 1e2*area_relative

    scan_back['com'] = com
    scan_back['std'] = std
    scan_back['integral'] = integral
    scan_back['center_fit'] = center_fit
    scan_back['width_fit'] = width_fit
    scan_back['area_fit'] = area_fit
    scan_back['transient_results'] = transient_results

    tools.make_folder('exportedResults')
    tools.write_list_to_file('exportedResults/' + scan_back['identifier'] + '.dat', '0T(K)  1F(mJ/cm²)  '
                             + '2delay(ps)  3strainCOM(permille)  4STDrelative(%)  5Integralrelative(%)  '
                             + '6strainFit(permille)  7widthFit(%)  8areaFit(%)', transient_results)
    return scan_back


# %%
'''------------------------------------------------
   Functions specialised on rss experimental scheme
   ------------------------------------------------'''

# %%


def read_data_rss(scan, export):
    '''This function reads in the raw data (intensity per pixel and omega angle) and
    adds it to the dictionary 'scan'. This dictionary contains the corresponding
    measurement parameters added by 'get_scan_parameter' and the evaluation parameters
    added in the main script. From the raw data the function exclude bad loops and pixels
    given by the keys 'bad_loops' and 'bad_pixels' of 'scan' and normalize the intensity
    on the crystal voltage using the key 'crystal_offset'. Scan with an intensity below
    'crystal_threshold' are excluded. By the keysof the dictionary 'export 'raw_data_path'
    and 'bool_refresh_data' you can refresh the data from the data path of the raw data
    during measurement.

    Parameters
    ----------
    scan : dictionary
        This dictionary contains the parameters from the reference file. This function uses
        the keys: 'number': row of refernece file, 'date': date of measurement, 'time':
        time of measurement, 'temperature': temperature in [K], 'fluence': excitation fluence
        in [mJ/cm2], 'crystal_offset': offset of teh reference diode in [V], 'crystal_threshold':
        minimum intensity not excluded in [V], 'bad_loops': loops to excluded (default: []),
        'bad_pixels': pixels to exclude (default: []), 'time_zero': delay of pump-probe overlap
        in [ps] (default: 0.0).
    export : dictionary
        This dictionary contains the parameters to plot and export the calculated results. This
        function uses the keys: 'bool_refresh_data': whether or not the raw data should be updated
        during measurement (default: False), 'raw_data_path': the corresponding path to the measurement
        folder (default: 'test/test2/test3/'), 'save_path': path to the folder to finally save the data.

    Returns
    -------
    scan : dictionary
        The function added the keys: 'pixel_axis': array of the pixel numbers of the detector, 'delays':
        all delays including several loops, 'unique_delays': different measured delays, 'intensity':
        on reference diode voltage normalized intensity per non-excluded pixel and all delays and
        non-excluded loops, 'omega_deg': incident x-ray angle in [deg], 'omega_rad': incident x-ray
        angle in [rad].

    '''
    if 'bool_refresh_data' in export:
        if export['bool_refresh_data']:
            tools.make_folder('RawData/' + export['save_path'])
            shutil.copyfile(export['raw_data_path'] + export['save_path'] + '/parameters' + scan['time'] +
                            '.txt', 'RawData/' + export['save_path'] + '/parameters' + scan['time'] + '.txt')
            shutil.copyfile(export['raw_data_path'] + export['save_path'] + '/scans' + scan['time'] +
                            '.txt', 'RawData/' + export['save_path'] + '/scans' + scan['time'] + '.txt')
            print('refreshing data from' + export['raw_data_path'] + export['save_path'] + ' ...')

    data_raw = np.genfromtxt('RawData/' + export['save_path'] + '/scans' + scan['time'] + '.dat', comments='%')
    print('─' * 65)
    print('Evaluating measurement: ' + str(scan['number']) + ' | ' + scan['date'] +
          ' | ' + scan['time'] + ' | ' + str(scan['temperature']) + r'K' + ' | ' +
          str(scan['fluence']) + r'mJ/cm2')
    print('Based on crystal threshold ' + str(sum(data_raw[:, 4] - scan['crystal_offset'] < scan['crystal_threshold']))
          + '/' + str(len(data_raw[:, 4])) + ' scans are excluded.')
    print('─' * 65)
    data_raw = data_raw[data_raw[:, 4] - scan['crystal_offset'] > scan['crystal_threshold'], :]

    if 'bad_loops' in scan:
        print('Excluded loops from the measurement:' + str(scan['bad_loops']))
        for ii in range(len(scan['bad_loops'])):
            data_raw = data_raw[data_raw[:, 0] != scan['bad_loops'][ii], :]
    if 'bad_pixels' in scan:
        print('Excluded pixels from the measurement:' + str(scan['bad_pixels']))
        mask_bad_pixels = np.zeros(len(data_raw[0, :]))
        for bp in scan['bad_pixels']:
            mask_bad_pixels[bp+9] = 1
        data_raw = data_raw[:, mask_bad_pixels != 1]
    if 'time_zero' in scan:
        scan['delays'] = data_raw[:, 1] - scan['time_zero']
    else:
        scan['delays'] = data_raw[:, 1] - 0

    crystal_v = data_raw[:, 4] - scan['crystal_offset']
    intensity = data_raw[:, 9:]
    for i in range(len(scan['delays'])):
        intensity[i, :] = data_raw[i, 9:]/crystal_v[i]
    scan['pixel_axis'] = np.arange(0, np.size(intensity[0, :]), 1)
    scan['unique_delays'] = np.unique(scan['delays'])
    scan['intensity'] = intensity
    scan['omega_deg'] = np.unique(data_raw[:, 3])
    scan['omega_rad'] = np.radians(scan['omega_deg'])
    return scan


def get_rocking_rss(scan_raw, export):
    '''This function calculates the intensity distribution along qz from an intensity
    as function of the detector pixel for a certain incident x-ray angle. The resulting
    loop-wise averaged rocking curve as function of the pump-probe delay is appended to
    the dictionary and saved in the respective folder structure. By the key 'qz_border'
    of 'scan_raw' you can choose a certain range in reciprocal space that contains the
    relevant Bragg peak (default: full covered qz range).

    Parameters
    ----------
    scan_raw : dictionary
        This dictionary contains the parameters of the measurement as well as the x-ray
        intensity as function of pump-probe delay and detector pixel. This function uses
        the keys: 'omge_rad': incident x-ray angle in [rad], 'distance': distance between
        sample and detector in [m], 'pixel_axis': array of the pixel numbers of the detector,
        'centerpixel': pixel where incident equal to diffraction angle is fullfilled (default:
        244), 'qz_border': lower and upper border of teh relevant qz range (default: full
        covered qz range), 'delays': all delays including several loops, 'unique_delays':
        different measured delays and 'intensity': on reference diode voltage normalized intensity
        per non-excluded pixel and all delays and non-excluded loops.
    export : dictionary
        This dictionary contains the parameters to plot and export the calculated results. This
        function uses the keys: 'save_path': path to the folder to finally save the data and
        'file_name': name of teh saved data given by a combination of date and time of the
        measurement.

    Returns
    -------
    scan_raw : dictionary
        The function added the keys: 'theta_axis': corresponding diffraction angles of the detector
        pixels in [rad] given by 'omega_rad' and 'distance', 'qx_axis': values of the detector pixels
        along qx given by 'theta_axis', 'qz_axis': values of the detector pixels along qz given by
        'theta_axis' and limited by 'qz_border', 'intensity_rock': loop-wise averaged intensity as
        function of 'unique_delays' and the choosen range in qz 'qz_axis'.

    '''
    print('Calculating rocking curve.')
    if 'centerpixel' in scan_raw:
        theta_axis = calc_theta_value(scan_raw['omega_rad'], scan_raw['distance'],
                                      scan_raw['pixel_axis'], scan_raw['centerpixel'])
    else:
        theta_axis = calc_theta_value(scan_raw['omega_rad'], scan_raw['distance'],
                                      scan_raw['pixel_axis'], 244)

    scan_raw['qx_axis'], qz_axis = calc_qx_qz(scan_raw['omega_rad'], theta_axis)
    if 'qz_border' in scan_raw:
        s_qz_range = (qz_axis >= scan_raw['qz_border'][0]) & (qz_axis <= scan_raw['qz_border'][1])
        scan_raw['qz_axis'] = qz_axis[s_qz_range]
        scan_raw['theta_axis'] = theta_axis[s_qz_range]
    else:
        scan_raw['qz_axis'] = qz_axis
        scan_raw['theta_axis'] = theta_axis

    rocking = np.zeros((len(scan_raw['unique_delays'])+2, len(scan_raw['qz_axis'])+1))
    rocking[0, :] = np.append([-8000], scan_raw['qz_axis'])
    rocking[1, :] = np.append([-8000], scan_raw['theta_axis'])
    rocking[:, 0] = np.append([-8000, -8000], scan_raw['unique_delays'])
    counter = 0
    for d in scan_raw['unique_delays']:
        rocking[counter+2, 1:] = np.array(np.mean(scan_raw['intensity'][scan_raw['delays'] == d, :][:, s_qz_range], 0))
        counter += 1

    scan_raw['intensity_rock'] = rocking[2:, 1:]

    print('Exporting calculated rocking curves.')
    tools.make_folder('exportedRockingCurves')
    tools.write_list_to_file('RawData/' + export['save_path'] + '/rockingCurves'
                             + export['file_name'] + '.dat',
                             'column 1: delays, row 1: qz, row 2: theta', rocking)
    tools.write_list_to_file('exportedRockingCurves/' + export['file_name'] + '.dat',
                             'column 1: delays, row 1: qz, row 2: theta', rocking)
    return scan_raw


# %%
'''------------------------------------------
   Fit functions for evaluating rocking curve
   ------------------------------------------'''

# %%


def gaussian_back_fit(center_start, sigma_start, amplitude_start, qz_axis, intensity):
    """This function returns the fit parameters for a gaussian with linear background model
    as a lm model object for the 'intensity' as function of 'qz_axis'. The start values of
    'slope' and 'intercept' are set to zero by default.

    Parameters:
    -----------
    center_start: float
            start value for the center of gaussian
    sigma_start: float
            start value for the width of gaussian
    amplitude_start: float
            start value for the amplitude of gaussian
    qz_axis: 1D array
            part of interest of reciprocal space
    intensity: 1D array
            part of interest of the detector intensities

    Returns:
    --------
    fit: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit)
            and the best fit (.best_fit) for the further analysis

    Example:
    --------
    >>> 3.855 = gaussian_back_fit(3.85, 0.05, 100, qz, intensity)['g_center']"""

    model = lm.models.GaussianModel(prefix="g_") + lm.models.LinearModel()
    pars = lm.Parameters()
    pars.add_many(('g_center', center_start, True),
                  ('g_sigma', sigma_start, True),
                  ('g_amplitude', amplitude_start, True),
                  ('slope', 0, True),
                  ('intercept', 0, True))
    fit = model.fit(intensity, pars, x=qz_axis)
    return fit


def gaussian_fit(center_start, sigma_start, amplitude_start, qz_axis, intensity):
    """This function returns the fit parameters for a gaussian peak model as a lm
    model object for the 'intensity' as function of 'qz_axis'.

    Parameters:
    -----------
    center_start: float
            start value for the center of gaussian
    sigma_start: float
            start value for the width of gaussian
    amplitude_start: float
            start value for the amplitude of gaussian
    qz_axis: 1D array
            part of interest of reciprocal space
    intensity: 1D array
            part of interest of the detector intensities

    Returns:
    --------
    fit: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit)
            and the best fit (.best_fit) for the further analysis

    Example:
    --------
    >>> 3.855 = gaussian_back_fit(3.85, 0.05, 100, qz, intensity)['g_center']"""

    model = lm.models.GaussianModel(prefix="g_")
    pars = lm.Parameters()
    pars.add_many(('g_center', center_start, True),
                  ('g_sigma', sigma_start, True),
                  ('g_amplitude', amplitude_start, True))
    fit = model.fit(intensity, pars, x=qz_axis)
    return fit


# %%
'''------------------------------------
   Figures evaluating the rocking curve
   ------------------------------------'''

# %%


def plot_rocking_overview(scan_rock, export):
    '''This function generates an illustration of the transient loopwise-averaged
    intensity distribution along qz. The three-dimensional figure consists of two
    parts separated by an axes break to visualize both strain waves at small delays
    and heat transport for long delays.

    Parameters
    ----------
    scan_rock : dictionary
        This dictionary contains the measurmeent parameters as well as the raw x-ray
        data and the calculated intensity distribution along qz. This function uses the
        keys: 'unique_delays': measured delays, 'qz_axis': part of reciprocal space given
        by key 'qz_range', 'intensity_rock': loop-wise averaged intensity of transient
        rocking curves.
    export : dictionary
        This dictionary contains the parameters to plot and export the calculated results
        in 'scan_rock'. This function uses the keys: 'bool_plot_2D_rocking_semilogy': whether
        the norm is logarithmic or linear, 'intensity_range_2D': list of intensity min and max
        for plotting range, 'delay_steps': list of xtick step first part, delay of axis break and
        xtick step second part, 'plot_title': main measurmeent parameter as title of the figure,
        'save_path': location in the folder structure to save the figure.

    Returns
    -------
    None.

    '''
    print('Plot time-resolved loopwise-averaged rocking curves.')

    f = plt.figure()
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], height_ratios=[1], wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    X, Y = np.meshgrid(scan_rock['unique_delays'], scan_rock['qz_axis'])

    for ax in [ax1, ax2]:
        if export['bool_plot_2D_rocking_semilogy']:
            p = ax.pcolormesh(X, Y, np.transpose(scan_rock['intensity_rock']/np.max(scan_rock['intensity_rock'])),
                              norm=matplotlib.colors.LogNorm(
                vmin=export['intensity_range_2D'][0], vmax=export['intensity_range_2D'][1]),
                shading='auto', cmap=color.cmap_blue_red_5)
            ax.axvline(x=export['delay_steps'][1], ls='--', color="black", lw=1)
        else:
            p = ax.pcolormesh(X, Y, np.transpose(scan_rock['intensity_rock']/np.max(scan_rock['intensity_rock'])),
                              vmin=export['intensity_range_2D'][0], vmax=export['intensity_range_2D'][1],
                              shading='auto', cmap=color.cmap_blue_red_5)
            ax.axvline(x=export['delay_steps'][1], ls='--', color="black", lw=1)

    ax1.axis([scan_rock['unique_delays'].min(), export['delay_steps'][1],
             scan_rock['qz_axis'].min(), scan_rock['qz_axis'].max()])
    ax1.set_xticks(np.arange(scan_rock['unique_delays'].min(), export['delay_steps'][1],
                             export['delay_steps'][0]))
    ax2.axis([export['delay_steps'][1], scan_rock['unique_delays'].max(),
             scan_rock['qz_axis'].min(), scan_rock['qz_axis'].max()])
    ax2.set_xticks(np.arange(export['delay_steps'][1],
                   scan_rock['unique_delays'].max(), export['delay_steps'][2]))

    ax1.axvline(x=0, ls='--', color="black", lw=1)
    ax1.set_xlabel('delay (ps)')
    ax1.xaxis.set_label_coords(0.75, -0.08)
    ax1.set_ylabel(r'$q_z$ ($\mathrm{\AA^{-1}}$)')
    ax1.yaxis.set_ticks_position('left')
    ax1.spines['right'].set_visible(False)
    ax2.yaxis.set_ticks_position('right')
    ax2.spines['left'].set_visible(False)
    ax2.set_yticklabels([])

    cbaxes = f.add_axes([0.925, 0.125, 0.023, 0.76])
    cbar = plt.colorbar(p, cax=cbaxes, orientation='vertical')
    cbar.ax.tick_params(labelsize=9, pad=3)
    cbar.set_label(r'normed x-ray intensity', rotation=90)
    cbar.outline.set_linewidth(0.75)
    cbar.ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
    cbar.ax.tick_params('both', length=0, width=0.5, which='minor', direction='in')

    plt.text(0.15, 1.07, export['plot_title'], fontsize=11, ha='left', va='center', transform=ax1.transAxes)
    tools.make_folder('RawData/' + export['save_path'] + '/exportedFigures')
    plt.savefig('RawData/' + export['save_path'] + '/exportedFigures/rocking_curves_2D.png',
                dpi=400, bbox_inches='tight')
    plt.show()


def plot_peak_fit(export, qz_axis, fit, ref_fit, delay):
    '''This function generates a comparison of the rocking curve (intensity(qz)) with
    the fit of the Bragg peak. The rocking curve and its fit at a certan delay is
    compared with the 'ref_fit' of the averaged rocking curve before t0.

    Parameters:
    -----------
    export: dictionary
            This dictionary contains the parameters to plot and export. This function uses the keys:
            'bool_plot_1D_rocking_semilogy': whether the norm is logarithmic or linear,
            'intensity_range_1D': list of intensity min and max for plotting range, 'plot_title':
            main measurmeent parameter as title of the figure, 'save_path': location in the folder
            structure to save the figure.
    qz_axis: 1D array
            the relevant region in reciprocal space
    fit: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit)
            and the best fit (.best_fit).
    ref_fit: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit)
            and the best fit (.best_fit) for delays before exciation
    delay: float
            delay of the fitted rocking curve, used in the plot title

   Returns
   -------
   None.

   '''
    plt.figure()
    gs = gridspec.GridSpec(1, 1, width_ratios=[1], height_ratios=[1], wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[0])

    if export['bool_plot_1D_rocking_semilogy']:
        ax1.semilogy(qz_axis, fit.data/np.max(fit.data), 's', color='black', label="data")
        ax1.semilogy(qz_axis, ref_fit.best_fit/np.max(fit.data), '-', color="grey", label="fit t<0")
        ax1.semilogy(qz_axis, fit.best_fit/np.max(fit.data), '-', color=color.cmap_blue_red_5(0.99), label="best fit")
    else:
        ax1.plot(qz_axis, fit.data/np.max(fit.data), 's', color='black', label="data")
        ax1.plot(qz_axis, ref_fit.best_fit/np.max(fit.data), '-', color="grey", label="fit t<0")
        ax1.plot(qz_axis, fit.best_fit/np.max(fit.data), '-', color=color.cmap_blue_red_5(0.99), label="best fit")

    ax1.set_xlabel(r"$q_{\mathrm{z}}$" + r" ($\AA^{-1}$)")
    ax1.set_ylabel(" norm x-ray intensity")
    ax1.set_xlim(np.min(qz_axis), np.max(qz_axis))
    ax1.set_ylim(export['intensity_range_1D'][0], export['intensity_range_1D'][1])

    ax1.legend(loc=0)
    plt.text(-0.05, 1.1, str(np.round(delay, 2)) + 'ps | ' + export['plot_title'],
             fontsize=11, ha='left', va='center', transform=ax1.transAxes)
    tools.make_folder('RawData/' + export['save_path'] + '/Fitgraphics')
    plt.savefig('RawData/' + export['save_path'] + '/Fitgraphics' + '/' +
                str(int(10*delay))+'ps.png', dpi=200, bbox_inches='tight')
    plt.show()


def plot_transient_results(scan_results, export):
    '''This function generates an illustration of the transient change of the Bragg
    peak properties by the fit and a center-of-mass analysis. The figure consists of
    three panels for teh transient relative change of the peak position along qz, its
    width and its amplitude. The delay axis consists of two parts separated by an axis
    break to visualize both strain waves at small delays and heat transport for long delays.

    Parameters
    ----------
    scan_results : dictionary
        This dictionary contains the measurmeent parameters as well as the transient properties
        of the measured Bragg peak and its relative change. This function uses the
        keys: 'unique_delays': measured delays and 'transient_results': two-dimensional array that
        contains the relative change of teh peak properties determined from a center-of-mass analysis
        and a fit.
    export : dictionary
        This dictionary contains the parameters to plot and export the calculated results
        in 'scan_results'. This function uses the keys: 'delay_steps': list of xtick step first part,
        delay of axis break and xtick step second part, 'plot_title': main measurmeent parameter
        as title of the figure and 'save_path': location in the folder structure to save the figure.

    Returns
    -------
    None.

    '''
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1], height_ratios=[1, 1, 1], wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    ax5 = plt.subplot(gs[4])
    ax6 = plt.subplot(gs[5])

    ax1.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 3], 's-', label="com")
    ax1.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 6], 's-', label="fit")
    ax2.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 3], 's-', label="com")
    ax2.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 6], 's-', label="fit")
    ax1.set_ylim(scan_results['transient_results'][:, 6].min() - 0.1*(scan_results['transient_results'][:, 6].max()
                                                                      - scan_results['transient_results'][:, 6].min()),
                 scan_results['transient_results'][:, 6].max() + 0.1*(scan_results['transient_results'][:, 6].max()
                                                                      - scan_results['transient_results'][:, 6].min()))
    ax2.set_ylim(scan_results['transient_results'][:, 6].min() - 0.1*(scan_results['transient_results'][:, 6].max()
                                                                      - scan_results['transient_results'][:, 6].min()),
                 scan_results['transient_results'][:, 6].max() + 0.1*(scan_results['transient_results'][:, 6].max()
                                                                      - scan_results['transient_results'][:, 6].min()))
    ax2.legend(loc=0, fontsize=13)
    ax1.set_ylabel('strain ($10^{-3}$)', fontsize=16)
    ax1.yaxis.set_label_coords(-0.07, 0.5)

    ax3.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 4], 's-', label="com")
    ax3.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 7], 's-', label="fit")
    ax4.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 4], 's-', label="com")
    ax4.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 7], 's-', label="fit")
    ax3.set_ylim(scan_results['transient_results'][:, 7].min() - 0.1*(scan_results['transient_results'][:, 7].max()
                                                                      - scan_results['transient_results'][:, 7].min()),
                 scan_results['transient_results'][:, 7].max() + 0.1*(scan_results['transient_results'][:, 7].max()
                                                                      - scan_results['transient_results'][:, 7].min()))
    ax4.set_ylim(scan_results['transient_results'][:, 7].min() - 0.1*(scan_results['transient_results'][:, 7].max()
                                                                      - scan_results['transient_results'][:, 7].min()),
                 scan_results['transient_results'][:, 7].max() + 0.1*(scan_results['transient_results'][:, 7].max()
                                                                      - scan_results['transient_results'][:, 7].min()))
    ax4.legend(loc=0, fontsize=13)
    ax3.set_ylabel('width change ($10^{-2}$)', fontsize=16)
    ax3.yaxis.set_label_coords(-0.07, 0.5)

    ax5.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 5], 's-', label="com")
    ax5.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 8], 's-', label="fit")
    ax6.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 5], 's-', label="com")
    ax6.plot(scan_results['unique_delays'], scan_results['transient_results'][:, 8], 's-', label="fit")
    ax5.set_ylim(scan_results['transient_results'][:, 8].min() - 0.1*(scan_results['transient_results'][:, 8].max()
                                                                      - scan_results['transient_results'][:, 8].min()),
                 scan_results['transient_results'][:, 8].max() + 0.1*(scan_results['transient_results'][:, 8].max()
                                                                      - scan_results['transient_results'][:, 8].min()))
    ax6.set_ylim(scan_results['transient_results'][:, 8].min() - 0.1*(scan_results['transient_results'][:, 8].max()
                                                                      - scan_results['transient_results'][:, 8].min()),
                 scan_results['transient_results'][:, 8].max() + 0.1*(scan_results['transient_results'][:, 8].max()
                                                                      - scan_results['transient_results'][:, 8].min()))
    ax6.legend(loc=0, fontsize=13)
    ax5.set_ylabel('amplitude change ($10^{-2}$)', fontsize=16)
    ax5.yaxis.set_label_coords(-0.07, 0.5)

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.axhline(y=0, ls='--', color="gray", lw=1)
        ax.tick_params('both', length=4, width=1, which='major', direction='in', labelsize=12)
        ax.tick_params(axis='x', which='both', bottom='on', top='on', labelbottom='off')
    for ax in [ax1, ax2]:
        ax.xaxis.set_ticks_position('top')
    for ax in [ax1, ax3, ax5]:
        ax.set_xticks(np.arange(scan_results['unique_delays'].min(),
                      export['delay_steps'][1], export['delay_steps'][0]))
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.axvline(x=export['delay_steps'][1], ls='--', color="gray", lw=1)
        ax.axvline(x=0, ls='--', color="gray", lw=1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim([scan_results['unique_delays'].min(), export['delay_steps'][1]])

    for ax in [ax2, ax4, ax6]:
        ax.yaxis.tick_right()
        ax.axvline(x=export['delay_steps'][1], ls='--', color="gray", lw=1)
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(export['delay_steps'][1],
                      scan_results['unique_delays'].max(), export['delay_steps'][2]))
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(export['delay_steps'][1], scan_results['unique_delays'].max())

    ax5.set_xlabel(r"pump-probe delay t (ps)", fontsize=16)
    ax5.xaxis.set_label_coords(0.75, -0.1)

    plt.text(0.25, 1.16, export['plot_title'], fontsize=16, ha='left', va='center', transform=ax1.transAxes)
    plt.savefig('RawData/' + export['save_path'] + '/OverviewResults.png', dpi=400, bbox_inches='tight')
    plt.show()
