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
    scan_dict : dictionary
        Contains the parameters of the measurement. Mandatory keys are: 'identifier':
        combination of date and time,'date': date of measurement, 'time': time of
        measurement, 'temperature': start temperature in [K], 'fluence' - incident
        excitation fluence in [mJ/cm2], 'time_zero': delay of pump-probe overlap in [ps],
        'distance': distance between sample and detector in [m].
        Optinal keys are: 'power': laser power in [mW], 'spot_size_x': FWHM of laser spot
        along x-direction in [mum], 'spot_size_y': FWHM of laser spot along
        y-direction in [mum], 'peak': measured Bragg peak of sample coded by
        an integer, 'magnetic_field': applied magnetic field in [T], 'wavelength': excitation
        wavelength in [nm], 'double_pulse': first, second or both excitation pulses coded
        by an integer, 'z_pos': z-position on the sample in [mm], 'y_pos': y-position on the
        sample in [mm], 'pulse_length': length of the pump-pulse in [ps], 'colinear_exc':
        coded wether or not pump and probe are collinear with an integer.

    Example
    -------
    >>> get_scan_parameter('reference.txt', 1) = {'number': 1,
                                                  'identifier': 20211123174703,
                                                  'date': '20211123',
                                                  'time': '183037',
                                                  'temperature': 300,
                                                  'fluence': 8.0,
                                                  'time_zero': 0.7,
                                                  'distance': 0,
                                                  'power': 110,
                                                  'spot_size_x': 960,
                                                  'spot_size_y': 960,
                                                  'double_pulse': 0,
                                                  'pulse_length': 0.1,
                                                  'wavelength': 0,
                                                  'colinear_exc': 800,
                                                  'peak_number': 1,
                                                  'magnetic_field': 0.0,
                                                  'z_pos': -4.52,
                                                  'y_pos': 2.25,
                                                 }

    '''
    parameters = pd.read_csv(r'ReferenceData/'+str(ref_file)+'.txt', delimiter="\t", header=0, comment="#")
    header = list(parameters.columns.values)
    scan_dict = {'number': number}
    for i, entry in enumerate(header):
        if entry == 'date' or entry == 'time':
            scan_dict[entry] = tools.timestring(int(parameters[entry][number]))
        else:
            scan_dict[entry] = parameters[entry][number]
    return scan_dict


def get_analysis_parameter(scan_dict):
    eval_dict = {}
    eval_dict['plot_title'] = str(scan_dict['number']) + ' | ' + scan_dict['date'] + ' | ' + scan_dict['time'] + \
        ' | ' + str(scan_dict['temperature']) + r'$\,$K' + ' | ' + str(int(scan_dict['fluence'])) + r'$\,$mJ/cm$^2$'
    eval_dict['save_path'] = scan_dict['date'] + '/' + scan_dict['time']
    eval_dict['file_name'] = scan_dict['date'] + scan_dict['time']
    eval_dict['time_zero'] = 0
    eval_dict['centerpixel'] = 244
    return eval_dict


def read_data_rss(scan_dict, eval_dict):

    if eval_dict['bool_refresh_data']:
        tools.make_folder('RawData/' + eval_dict['save_path'])
        shutil.copyfile(eval_dict['raw_data_path'] + eval_dict['save_path'] + '/parameters' + scan_dict['time'] +
                        '.txt', 'RawData/' + eval_dict['save_path'] + '/parameters' + scan_dict['time'] + '.txt')
        shutil.copyfile(eval_dict['raw_data_path'] + eval_dict['save_path'] + '/scans' + scan_dict['time'] +
                        '.txt', 'RawData/' + eval_dict['save_path'] + '/scans' + scan_dict['time'] + '.txt')

    data_raw = np.genfromtxt('RawData/' + eval_dict['save_path'] + '/scans' + scan_dict['time'] + '.dat', comments='%')
    data_raw = data_raw[data_raw[:, 4] - eval_dict['crystal_offset'] > eval_dict['crystal_threshold'], :]
    # print('Based on crystal threshold ' + str(sum(data_raw[:, 4] - crystal_off < threshold)) + ' scans are excluded.')

    for ii in range(len(eval_dict['bad_loops'])):
        data_raw = data_raw[data_raw[:, 0] != eval_dict['bad_loops'][ii], :]

    mask_bad_pixels = np.zeros(len(data_raw[0, :]))
    for bp in eval_dict['bad_pixels']:
        mask_bad_pixels[bp+9] = 1
    data_raw = data_raw[:, mask_bad_pixels != 1]

    delays = data_raw[:, 1]-eval_dict['time_zero']

    crystal_v = data_raw[:, 4] - eval_dict['crystal_offset']
    intensity = data_raw[:, 9:]
    for i in range(len(delays)):
        intensity[i, :] = data_raw[i, 9:]/crystal_v[i]
    scan_dict['delays'] = delays
    scan_dict['pixel_axis'] = np.arange(0, np.size(intensity[0, :]), 1)
    scan_dict['unique_delays'] = np.unique(delays)
    scan_dict['intensity'] = intensity
    scan_dict['omega_deg'] = np.unique(data_raw[:, 3])
    scan_dict['omega_rad'] = np.radians(scan_dict['omega_deg'])
    return scan_dict


def calc_theta_value(omega_rad, distance, pixel, centerpixel):
    """This function calculates the corresponding diffraction angle 'theta' of a
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
    >>> calc_theta_value(0.405,0.666,244,244)[0] = 0.405

    """
    pixel_size = 0.172e-3  # delete
    delta_theta = np.arctan(pixel_size / distance)
    theta_rad = omega_rad + delta_theta * (pixel - centerpixel)
    return theta_rad


def calc_qx_qz(omega_rad, theta_rad):
    """This function calculates the qx-qz coordinate in reciprocal space for a
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
    >>> calc_qx_qz(0.405,0.405)[1] = 0
    >>> calc_qx_qz(0.405,0.405)[1] = 3.21

    """
    wave_vector = 4.07523  # delete
    qx_val = wave_vector * (np.cos(theta_rad) - np.cos(omega_rad))
    qz_val = wave_vector * (np.sin(theta_rad) + np.sin(omega_rad))
    return qx_val, qz_val


def get_rocking_rss(scan_dict, eval_dict):
    scan_dict['theta_axis'] = calc_theta_value(scan_dict['omega_rad'], scan_dict['distance'],
                                               scan_dict['pixel_axis'], eval_dict['centerpixel'])
    scan_dict['qx_axis'], scan_dict['qz_axis'] = calc_qx_qz(scan_dict['omega_rad'], scan_dict['theta_axis'])

    rocking = np.zeros((len(scan_dict['unique_delays'])+2, len(scan_dict['pixel_axis'])+1))
    rocking[0, :] = np.append([-8000], scan_dict['qz_axis'])
    rocking[1, :] = np.append([-8000], scan_dict['theta_axis'])
    rocking[:, 0] = np.append([-8000, -8000], scan_dict['unique_delays'])
    counter = 0
    for d in scan_dict['unique_delays']:
        rocking[counter+2, 1:] = np.array(np.mean(scan_dict['intensity'][scan_dict['delays'] == d, :], 0))
        counter += 1
    scan_dict['rocking'] = rocking

    tools.make_folder('exportedRockingCurves')
    tools.write_list_to_file('RawData/' + eval_dict['save_path'] + '/rockingCurves'
                             + eval_dict['file_name'] + '.dat',
                             'column 1: delays, row 1: qz, row 2: theta', rocking)
    tools.write_list_to_file('exportedRockingCurves/' + eval_dict['file_name'] + '.dat',
                             'column 1: delays, row 1: qz, row 2: theta', rocking)
    return scan_dict


def plot_rocking_overview(scan_dict, eval_dict):
    s_qz_range = (scan_dict['qz_axis'] >= eval_dict['qz_border'][0]) & (
        scan_dict['qz_axis'] <= eval_dict['qz_border'][1])
    qz_axis = scan_dict['qz_axis'][s_qz_range]
    rocking_intensity = scan_dict['rocking'][2:, 1:][:, s_qz_range]

    f = plt.figure(figsize=(5.3, 3.6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1], height_ratios=[1], wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    X, Y = np.meshgrid(scan_dict['unique_delays'], qz_axis)

    plotted = ax1.pcolormesh(X, Y, np.transpose(rocking_intensity/np.max(rocking_intensity)),
                             norm=matplotlib.colors.LogNorm(vmin=0.006, vmax=1),
                             shading='auto', cmap=color.cmap_blue_red_5)
    ax1.axis([scan_dict['unique_delays'].min(), eval_dict['delay_steps'][1], qz_axis.min(), qz_axis.max()])
    ax1.set_xticks(np.arange(scan_dict['unique_delays'].min(), eval_dict['delay_steps'][1],
                             eval_dict['delay_steps'][0]))
    ax1.set_xlabel('delay (ps)', fontsize=11)
    ax1.xaxis.set_label_coords(0.75, -0.08)
    ax1.set_ylabel(r'$q_z$ ($\mathrm{\AA^{-1}}$)', fontsize=11)
    ax1.spines['right'].set_visible(False)
    ax1.axvline(x=0, ls='--', color="black", lw=1)
    ax1.axvline(x=eval_dict['delay_steps'][1], ls='--', color="black", lw=1)
    ax2.axvline(x=eval_dict['delay_steps'][1], ls='--', color="black", lw=1)

    plotted = ax2.pcolormesh(X, Y, np.transpose(rocking_intensity/np.max(rocking_intensity)),
                             norm=matplotlib.colors.LogNorm(vmin=0.006, vmax=1),
                             shading='auto', cmap=color.cmap_blue_red_5)
    ax2.axis([eval_dict['delay_steps'][1], scan_dict['unique_delays'].max(), qz_axis.min(), qz_axis.max()])
    ax2.set_xticks(np.arange(eval_dict['delay_steps'][1],
                   scan_dict['unique_delays'].max(), eval_dict['delay_steps'][2]))
    ax2.yaxis.set_ticks_position('right')
    ax2.spines['left'].set_visible(False)
    ax2.set_yticklabels([])

    for ax in [ax1, ax2]:
        ax.tick_params('both', length=4, width=1, which='major', direction='in')
        ax.tick_params('both', length=1.5, width=0.5, which='minor', direction='in')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='y', which='minor', left=False)
        ax.tick_params(axis='both', which='major', labelsize=9, pad=3)
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1)

    cbaxes = f.add_axes([0.925, 0.125, 0.023, 0.76])
    cbar = plt.colorbar(plotted, cax=cbaxes, orientation='vertical')
    cbar.ax.tick_params(labelsize=9, pad=3)
    cbar.set_label(r'x-ray intensity', rotation=90, fontsize=11)
    cbar.ax.tick_params('both', length=3, width=0.5, which='major', direction='in')
    cbar.ax.tick_params('both', length=0, width=0.5, which='minor', direction='in')
    cbar.outline.set_linewidth(0.75)

    plt.text(0.15, 1.07, eval_dict['plot_title'], fontsize=11, ha='left', va='center', transform=ax1.transAxes)
    tools.make_folder('RawData/' + eval_dict['save_path'] + '/exportedFigures')
    plt.savefig('RawData/' + eval_dict['save_path'] + '/exportedFigures/rockingCurves2Dplot.png',
                dpi=400, bbox_inches='tight')
    plt.show()
