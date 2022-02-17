# -*- coding: utf-8 -*-
"""
This skript contains all basic functions used to evaluate pxs x-ray diffraction data.

-- Further information --

"""
import numpy as np
import numpy.ma as ma
import udkm.tools.helpers as hel
import udkm.tools.colors as color
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import lmfit as lm
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import shutil
import os

PXS_WL      = 1.5418           # Cu K-alpha wavelength in [Ang]
PXS_K       = 2*np.pi/PXS_WL   # absolute value of the incident k-vector in [1/Ang]
CRYSTAL_OFF = 0.0211595        # offset of crystal diode in [V]
PIXELSIZE   = 0.172e-3         # size of pilaus pixel in [m]

def read_param(ref_file,number):
    """This function reads the reference data file 'ref_file' and returns all parameters
    of the 'number'. measurement including 'date' and 'time' that are used for unique
    reference on the measurement.
    
    Parameters
    -----------
    ref_file : string 
            file name of the corresponding reference file in the folder 'ReferenceData'
    number : integer
            line in the reference datafile wich corresponds to the measurement
    
    Returns
    --------
    date : string
            date of the measurement
    time : string
            time of the measurement
    temperature : float
            temperature during the choosen measurement in [K]
    fluence : float
            fluence during the choosen measurement in [mJ/cm^2]
    distance : float
            distance between sample and detector for choosen measurement in [m]
    
    Example
    --------
    >>>  20191123 = read_param('Overview',0)[0]
    >>>  235959 = read_param('Overview',0)[1]
    >>>  300 = read_param('Overview',0)[2]
    >>>  5.2 = read_param('Overview',0)[3]
    >>>  0.666 = read_param('Overview',0)[4]
    """   
    param_list  = np.genfromtxt(r'ReferenceData/'+str(ref_file)+'.txt', comments='#')
    date = str(int(param_list[number,1]))
    date = (6-len(date))*'0'+date
    time = str(int(param_list[number,2]))
    time = (6-len(time))*'0'+time
    temperature  = param_list[number,3]
    fluence      = param_list[number,4]
    distance     = param_list[number,5] 
    peak_number  = param_list[number,6] 
    double_pulse = param_list[number,7]
    wavelength   = param_list[number,8]
    return date, time, temperature, fluence, distance, peak_number, double_pulse, wavelength 

def get_label(ref_file,number):
    """This function create a plot title, a saving path and a saving name for the 'number'.
    measurement of the 'ref_file' based on the corresponding parameters of the measurement.
    Therefore this function uses the function 'read_param'.
    
    Parameters
    -----------
    ref_file : string 
            file name of the corresponding reference file in the folder 'ReferenceData'
    number : integer
            line in the reference datafile wich corresponds to the measurement
    
    Returns
    --------
    plot_title : string
            contains the 'number' of measurement, 'date', 'time', 'temperature' and 'fluence'
            of the choosen measurement
    save_path : string
            contains the 'date' and 'time' of the choosen measurement as an order structure to define a 
            saving path like the order structure insaved the folder 'RawData'
    file_name : string
            contains the 'date' and 'time' of the choosen measurement to give saved plots and data
            a usefull name
    Example
    --------
    >>> '0 / 20191123 / 235959 / 300 K / 5.2 mJ/cm2' = get_label('Overview',0)[0]
    >>> '20191123/235959' = get_label('Overview',0)[1]
    >>> '20191123235959' = get_label('Overview',0)[2]
    """
    plot_title = str(number) +' | ' + read_param(ref_file,number)[0] + ' | ' + read_param(ref_file,number)[1] + ' | '+ str(read_param(ref_file,number)[2]) + 'K' +' | '+ str(read_param(ref_file,number)[3]) + 'mJ/cm2'
    save_path  = read_param(ref_file,number)[0] + '/' + read_param(ref_file,number)[1]
    file_name  = read_param(ref_file,number)[0] + read_param(ref_file,number)[1]
    return plot_title, save_path, file_name

def read_data_rss(ref_file,number,bad_loops,bad_pixels,crystal_off,threshold,t_0,bool_refresh_data,data_path):
    """This function imports the x-ray intensity as function of the detector pixel for the 'number'.
    single angle measurement listed in 'ref_file'. The function returns the measurement angle, the 
    delays in respect to temporal overlapp given by 't_0' and the normed intensity on the crystal
    voltage, where low intensity scans below 'threshold', bad pixels and bad loops are excluded 
    given by 'bad_loops' and 'bad_pixels', respectively.
    This function uses the functions 'get_label' and 'read_param'
    
    Parameters
    -----------
    ref_file : string 
            file name of the corresponding reference file in the folder 'ReferenceData'
    number : integer
            line in the reference datafile wich corresponds to the measurement
    bad_loops: list
            contain all bad loops that are excluded in further analysis
    bad_pixels : list
            contain bad pixels (0-487) that are excluded in further analysis
    crystal_off : float
            voltage at crystal without photons as reference for normalization
    treshold : float
            minimum of crystalvoltage corresponds to a minimum of photons per puls
    t_0 : float
            time when pump and probe arrives at the same time [ps]
            
    Returns
    --------
    omega_deg : 0D array
            incident angle of measurement in [deg]
    omega_rad : 0D array
            incident angle of measurement in [rad]
    delays : 1D array
            all measured delays with repetion for different loops in respect to 't_0'
    intensity : 2D array
            detected intensities for detectorpixel(x) and delays(y)
    
    Example
    --------
    >>> 23.2 = read_data_rss('Overview',0,[],[],0.021,0.1,0)[0]
    >>> 0.405 = read_data_rss('Overview',0,[],[],0.021,0.1,0)[1]
    >>> np.array([-1,0,0.5]) = read_data_rss('Overview',0,[],[],0.021,0.1,0)[2]
    >>> np.array(([0,0.5,1,0.5,0],[0,0.5,1,0.5,0],[0,0.6,0.8,0.3,0])) = read_data_rss('Overview',0,[],[],0.021,0.1,0)[3]
    
    """
    print(get_label(ref_file,number)[0] + str(read_param(ref_file,number)[4])+ 'm' )
    
    if bool_refresh_data:
        hel.makeFolder('RawData/' + get_label(ref_file,number)[1])
        shutil.copyfile(data_path + get_label(ref_file,number)[1] + '/parameters' + str(read_param(ref_file, number)[1]) + '.txt',
                        'RawData/' + get_label(ref_file,number)[1] + '/parameters' + str(read_param(ref_file, number)[1]) + '.txt')   
        shutil.copyfile(data_path + get_label(ref_file,number)[1] + '/scans' + str(read_param(ref_file, number)[1]) + '.txt',
                        'RawData/' + get_label(ref_file,number)[1] + '/scans' + str(read_param(ref_file, number)[1]) + '.txt')
    
    data_raw  = np.genfromtxt('RawData/' + str(get_label(ref_file,number)[1]) + '/scans' + str(read_param(ref_file,number)[1]) + '.dat', comments = '%')
    data_raw = data_raw[data_raw[:,4] - crystal_off > threshold,:]
    print('Based on crystal threshold ' + str(sum(data_raw[:,4] - crystal_off < threshold)) + ' scans are excluded.')
        
    for ii in range(len(bad_loops)):
        data_raw = data_raw[data_raw[:,0]!=bad_loops[ii],:]
        
    mask_bad_pixels = np.zeros(len(data_raw[0,:]))
    for bp in bad_pixels:
        mask_bad_pixels[bp+9] = 1
    data_raw = data_raw[:,mask_bad_pixels !=1]
    
    delays    = data_raw[:,1]-t_0
    crystal_v = data_raw[:,4] - crystal_off
    intensity = data_raw[:,9:]
    for i in range(len(delays)):
        intensity[i,:] = data_raw[i,9:]/crystal_v[i]
    angles    = data_raw[:,3]
    omega_deg = np.unique(angles)
    omega_rad = np.radians(omega_deg)
    return omega_rad, omega_deg, delays, intensity

def calc_theta_value(omega_rad,distance,pixel,centerpixel):
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
    pixel_size  = 0.172e-3  # delete
    delta_theta = np.arctan(pixel_size/distance)
    theta_rad   = omega_rad + delta_theta*(pixel-centerpixel)
    return theta_rad

def calc_qx_qz(omega_rad,theta_rad):
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
    qx_val = wave_vector * (np.cos(theta_rad)-np.cos(omega_rad))
    qz_val = wave_vector * (np.sin(theta_rad)+np.sin(omega_rad))
    return qx_val, qz_val

def get_rocking_rss(ref_file, number, omega_rad, centerpixel, delays, intensity): 
    """ This function gets the loopwise averaged intensity per unique delay as
    function of qz coordinates in reciprocal space calculated from 'omega_rad',
    'distance' and 'centerpixel' using the functions 'calc_theta_value' and
    'calc_qx_qz'. The resulting rocking curves are saved in the folder 'RawData'
    and 'exportedRockingCurves' using the function 'get_label'.
    
    Parameters
    -----------
    ref_file : string 
            file name of the corresponding reference file in the folder 'ReferenceData'
    number : integer
            line in the reference datafile wich corresponds to the measurement
    omega_rad : 0D array
            incident angle of measurement in [rad]
    centerpixel : integer
            pixel where Bragg condition 'omega' = 'theta' is fulfilled
    delays: 1D array
            all measured delays with repition for different loops
    intensity : 2D array
            detected intensities for detectorpixel(x) and delays(y)
            
    Returns:
    --------
    rocking: 2D array
            contains the averaged intensities for the unique measured delays(y1)
            in dependency of qz(x1) and theta(x2)
    
    Example:
    --------
    >>> get_rocking_rss('Overview',0,0.405,0.666,1,np.array([-1,0,0.5]),intensity)=
        np.array(([-8000,3.11,3.21,3.31],[-8000,0.40,0.405,0.41],[-1,0.1,0.8,0.1],[0,0.1,0.8,0.1],[0.5,0.2,0.7,0]))
    
    """
    unique_delays    = np.unique(delays) 
    pixel_axis       = np.arange(0, np.size(intensity[0,:]), 1) 
    distance         = read_param(ref_file,number)[4]
    theta_axis       = calc_theta_value(omega_rad, distance, pixel_axis, centerpixel)
    qx_axis, qz_axis = calc_qx_qz(omega_rad, theta_axis) 
    
    rocking      = np.zeros((len(unique_delays)+2, np.size(intensity[0,:])+1))
    rocking[0,:] = np.append([-8000], qz_axis)
    rocking[1,:] = np.append([-8000], theta_axis)
    rocking[:,0] = np.append([-8000,-8000], unique_delays)
    counter = 0
    for d in unique_delays:
        rocking[counter+2,1:] = np.array(np.mean(intensity[delays==d,:],0))
        counter +=1
        
    hel.makeFolder('exportedRockingCurves')
    hel.writeListToFile('RawData/' + get_label(ref_file,number)[1] + '/rockingCurves'
                        + get_label(ref_file,number)[2] + '.dat','column 1: delays, row 1: qz, row 2: theta',rocking)
    hel.writeListToFile('exportedRockingCurves/'+ get_label(ref_file,number)[2] + '.dat',
                        'column 1: delays, row 1: qz, row 2: theta',rocking) 
    
    return rocking

def plot_rocking_overview(ref_file,number,rocking,qz_range,delay_steps):
    """This function provides an overview plot of the transient 'rocking' curves
    of the 'number'. measurement in 'ref_file' for a given 'qz_range' as two-
    dimensional map. The parameter 'delay_steps' enables the adjustment of the 
    focus of the plot with the steps for small delays, delay of the axis split
    and the delay steps in the second narrower part of teh plot for large delays.
    Finally, the plot is saved in teh corresponding 'RawData' folder in a folder
    named 'exportedFigures' using the function 'get_label'.

    Parameters
    ----------
    ref_file : string 
            file name of the corresponding reference file in the folder 'ReferenceData'
    number : integer
            line in the reference datafile wich corresponds to the measurement
    rocking: 2D array
            contains the averaged intensities for the unique measured delays(y1)
            in dependency of qz(x1) and theta(x2)
    qz_range : list
        lower and higher border of the region of intrest in reciprocal space in [1/Ang]
    delay_steps : list
        first: steps first part of plot, second: delay of axis break, third: steps second part of plot

    Returns
    -------
    None.

    """
    
    qz_values  = rocking[0,1:]
    s_qz_range = (qz_values >= qz_range[0]) & (qz_values <= qz_range[1])
    qz_axis    = qz_values[s_qz_range]
    delays     = rocking[2:,0]
    intensity  = rocking[2:,1:]
    intensity  = intensity[:,s_qz_range]
     
    f       = plt.figure(figsize = (5.3, 3.6))
    gs      = gridspec.GridSpec(1, 2,width_ratios=[2,1],height_ratios=[1],wspace=0.0, hspace=0.0)
    ax1     = plt.subplot(gs[0])
    ax2     = plt.subplot(gs[1])        
    X,Y     = np.meshgrid(delays,qz_axis)
    
    plotted = ax1.pcolormesh(X,Y,np.transpose(intensity/np.max(intensity)),
                             norm=matplotlib.colors.LogNorm(vmin = 0.006, vmax = 1),
                             shading='auto',cmap = color.cmap_blue_red_5)
    ax1.axis([delays.min(), delay_steps[1], qz_axis.min(), qz_axis.max()])
    ax1.set_xticks(np.arange(delays.min(),delay_steps[1],delay_steps[0]))
    ax1.set_xlabel('delay (ps)', fontsize = 11)
    ax1.xaxis.set_label_coords(0.75, -0.08)
    ax1.set_ylabel(r'$q_z$ ($\mathrm{\AA^{-1}}$)',fontsize = 11)
    ax1.spines['right'].set_visible(False)
    ax1.axvline(x = 0 ,ls = '--',color = "black",lw = 1)
    ax1.axvline(x = delay_steps[1] ,ls = '--',color = "black",lw = 1)
    ax2.axvline(x = delay_steps[1] ,ls = '--',color = "black",lw = 1)
    
    plotted = ax2.pcolormesh(X,Y,np.transpose(intensity/np.max(intensity)),
                             norm=matplotlib.colors.LogNorm(vmin = 0.006, vmax = 1),
                             shading='auto',cmap = color.cmap_blue_red_5)
    ax2.axis([delay_steps[1], delays.max(), qz_axis.min(), qz_axis.max()])
    ax2.set_xticks(np.arange(delay_steps[1],delays.max(),delay_steps[2]))
    ax2.yaxis.set_ticks_position('right')
    ax2.spines['left'].set_visible(False)
    ax2.set_yticklabels([])
    
    for ax in [ax1,ax2]:
        ax.tick_params('both', length=4, width=1, which='major',direction = 'in')
        ax.tick_params('both', length=1.5, width=0.5, which='minor',direction = 'in')
        ax.xaxis.set_ticks_position('both')
        ax.tick_params(axis='y', which='minor', left=False)
        ax.tick_params(axis='both', which='major', labelsize=9,pad=3)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(1)
          
    cbaxes = f.add_axes([0.925, 0.125, 0.023, 0.76])  
    cbar = plt.colorbar(plotted,cax = cbaxes, orientation='vertical')
    cbar.ax.tick_params(labelsize=9,pad=3)
    cbar.set_label(r'x-ray intensity', rotation=90,fontsize = 11)
    cbar.ax.tick_params('both', length=3, width=0.5, which='major',direction = 'in')
    cbar.ax.tick_params('both', length=0, width=0.5, which='minor',direction = 'in')
    cbar.outline.set_linewidth(0.75)
    
    plt.text(0.15, 1.07, get_label(ref_file,number)[0], fontsize = 11, ha='left', va='center', transform=ax1.transAxes)
    hel.makeFolder('RawData/' + get_label(ref_file,number)[1] +'/exportedFigures')     
    plt.savefig('RawData/' + get_label(ref_file,number)[1] +'/exportedFigures/rockingCurves2Dplot.png',dpi =400,bbox_inches = 'tight')
    plt.show()

def gaussian_back_fit(center_start,sigma_start,amplitude_start,qz_axis,intensity):
    """This function returns the parameters for a gaussian with linear background modell for the data and axis 
    in a region of intrest.
    
    Parameters:
    -----------
    centerStart: float 
            start value for the center of gaussian
    sigmaStart: float
            start value for the width of gaussian
    amplitudeStart: float
            start value for the amplitude of gaussian
    slopeStart: float
            start value for the slope of the linear background
    interceptStart: float
            start value for the shift of the linear background
    qzROI: 1D array
            part of interest of qzaxis
    IntensityROI: 1D array
            part of interest of the detector intensities
            
    Returns:
    --------
    resultFit: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit) 
            and the best fit (.best_fit) for the further analysis
    
    Example:
    --------
    >>> c=GaussianBackFit(0,0,0,0,0,qz,I)['g_center']"""
    
    model = lm.models.GaussianModel(prefix = "g_") + lm.models.LinearModel() 
    pars  = lm.Parameters()                                                      
    pars.add_many(('g_center', center_start, True),
                ('g_sigma', sigma_start, True),
                ('g_amplitude', amplitude_start , True),
                ('slope', 0, True),
                ('intercept', 0, True))
    resultFit = model.fit(intensity, pars, x = qz_axis)
    return resultFit
    
def get_moments_back(rocking,qz_range,bool_substrate,list_param):  
    """This function returns the moments of the Bragg Peak (center of mass, standart derivation, integral) for all
    unique delays with the reference fit (gaussian + linear) for all delays before t0.
    
    Parameters:
    -----------
    RockingCurves: 2D array 
            contains the averaged intensities for the unique measured delays(x) in dependency of qz(y1) and the
            angle theta(y2)
    qzMin: float
            the lower border of region of interest
    qzmax: float
            the upper border of region of interest
            
    Returns:
    --------
    COM: 1D array
        contains the center of mass for every unique delay in consideration of the region of intrest
    STD: 1D array
        contains the standart derivation for every unique delay in consideration of the region of intrest
    Integral: 1D array
        contains the integral for every unique delay in consideration of the region of intrest
    resultFitRef: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit) 
            and the best fit (.best_fit) for the further analysis for meaan intensity before t0.
    uniqueDelays: 1D array
        contains the unique delays of measurement, part of delays without repetition
    
    Example:
    --------
    >>> [2.2,2.3,2.4,2.2,2.3,2.2] = 
    getMoments(RockingCurves,1.6,2.8)[0]"""
    
    unique_delays = rocking[2:,0]
    s_before_t0   = unique_delays < 0
    delays_t0     = unique_delays[s_before_t0]
    qz_axis       = rocking[0,1:]
    intensity     = rocking[2:,1:]
    s_qz_axis     = (qz_axis>qz_range[0]) & (qz_axis<qz_range[1])
    qz_roi        = qz_axis[s_qz_axis]
    intensity_roi = intensity[:,s_qz_axis]
    
    com      = np.zeros(np.size(unique_delays))
    std      = np.zeros(np.size(unique_delays))
    integral = np.zeros(np.size(unique_delays)) 
    for i in range(np.size(delays_t0)):
        com[i], std[i], integral[i] = hel.calcMoments(qz_roi,intensity_roi[s_before_t0,:][i,:])
    
    if bool_substrate:
        print('Sorry, this part is in progress')
    
    else:
        intensity_peak = np.zeros((len(intensity_roi[:,0]),len(intensity_roi[0,:])))
        ref_fit = gaussian_back_fit(np.mean(com[s_before_t0]),np.mean(std[s_before_t0]),
                                    np.max(intensity_roi),qz_roi,np.mean(intensity_roi[s_before_t0,:],0))
        for i in range(np.size(unique_delays)):
            intensity_peak[i,:] = intensity_roi[i,:] - (qz_roi*ref_fit.values["slope"]+ref_fit.values["intercept"])
            intensity_peak[i,:] = intensity_peak[i,:] - np.min(intensity_peak[i,:])
            com[i], std[i], integral[i] = hel.calcMoments(qz_roi,intensity_peak[i,:])    
        
    return com, std, integral, unique_delays, ref_fit, intensity_peak, qz_roi

def gaussian_fit(center_start,sigma_start,amplitude_start,qz_roi,intensity_roi):
    """This function returns the parameters for a gaussian for the data and axis in a region of intrest.
    
    Parameters:
    -----------
    centerStart: float 
            start value for the center of gaussian
    sigmaStart: float
            start value for the width of gaussian
    amplitudeStart: float
            start value for the amplitude of gaussian
    qzROI: 1D array
            part of interest of qzaxis
    IntensityROI: 1D array
            part of interest of the detector intensities
    variationBool: 1D array
            contains True or False for variation of the parameters
        
    Returns:
    --------
    resultFit: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit) 
            and the best fit (.best_fit) for the further analysis
    
    Example:
    --------
    >>> c=GaussianFit(0,0,0,0,0,qz,I)['g_center']"""
    
    model     = lm.models.GaussianModel(prefix = "g_")
    pars      = lm.Parameters()                                                      
    pars.add_many(('g_center', center_start, True),
                  ('g_sigma', sigma_start, True),
                  ('g_amplitude', amplitude_start, True))
    result_fit = model.fit(intensity_roi, pars, x = qz_roi)
    return result_fit

def get_fit_values(ref_file,number,intensity_peak,qz_roi,unique_delays,ref_fit,bool_plot):
    """This function returns the values of the fit parameters (centerFit,widthFit,areaFit) for every unique delay
    as result of a fit of the rocking curves in the region of intrest. There is the possibility to plot the results 
    of the fits for every unique delay.
    
    Parameters:
    -----------
    NameFile: string 
            name of the reference datafile
    Number: integer
            line in the reference datafile wich corresponds to the measurement
    RockingCurves: 2D array 
            contains the averaged intensities for the unique measured delays(x) in dependency of qz(y1) and the
            angle theta(y2)
    qzMin: float
            the lower border of region of interest of the qzaxis
    qzmax: float
            the upper border of region of interest of the qzaxis
    resultFitRef: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit) 
            and the best fit (.best_fit) for the further analysis is the fit before t0
    plotFitResults: boolean
        if True the results of every fit for the unique delays are plotted with the function plotPeakFit
            
    Returns:
    --------
    centerFit: 1D array
        contains the center of the gasussian fit of the Bragg peak for every unique delay in consideration of the region of intrest
    widthFit: 1D array
        contains the width sigma of the gasussian fit of the Bragg peak for every unique delay in consideration of the region of intrest
    areaFit: 1D array
        contains the amplitude of the gasussian fit of the Bragg peak for every unique delay in consideration of the region of intrest
    uniqueDelays: 1D array
        contains the unique delays of measurement, part of delays without repetition
    
    Example:
    --------
    >>> [2.2,2.3,2.4,2.2,2.3,2.2] = 
    getMoments(RockingCurves,1.6,2.8)[0]""" 
    
    center_fit = np.zeros(len(unique_delays))
    width_fit  = np.zeros(len(unique_delays))
    area_fit   = np.zeros(len(unique_delays))
    for i in range(np.size(unique_delays)):
        fit = gaussian_fit(ref_fit.values["g_center"],ref_fit.values["g_sigma"],
                           ref_fit.values["g_amplitude"],qz_roi,intensity_peak[i,:])
        center_fit[i] = fit.values["g_center"] 
        width_fit[i]  = fit.values["g_sigma"]
        area_fit[i]   = fit.values["g_amplitude"]
        delay_eval    = unique_delays[i]
        if bool_plot == True:
            plot_peak_fit(ref_file,number,qz_roi,fit,ref_fit,delay_eval)
    return center_fit, width_fit, area_fit

def plot_peak_fit(ref_file,number,qz_roi,fit,ref_fit,delay): 
    """This function plot the intensity depending on qzaxis together with the corresponding fit and the referenz fit 
    for delays before t0 together to check the fitting results.
    
    Parameters:
    -----------
    NameFile: string 
            name of the reference datafile
    Number: integer
            line in the reference datafile wich corresponds to the measurement
    index: integer
            the index to distinguish the different iterations in the plot title
    qzROI: 1D array
            the region of intrest of the qzaxis to fit the bragg peak
    resultFit: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit) 
            and the best fit (.best_fit) for the further analysis is the fit for every delay
    resultFitRef: lm model objekt
            allows to get fit values (.values), the used data (.data), the initial fit (.init_fit) 
            and the best fit (.best_fit) for the further analysis is the fit before t0
    t: float
            the time correspond to the index-parameter number via uniqueDelays
    PlotSemilogy: boolean
            if true the yaxis of the plot will be logarithmic
    fontSize: integer
            the fontsize for the plots
    
    Example:
    --------
    >>> plotPeakFit('Dataset',qz,Fit,Fitt0,-5,False,18)"""

    plt.figure(figsize = (6,4)) 
    gs  = gridspec.GridSpec(1, 1,width_ratios=[1],height_ratios=[1],wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[0])       
   
    ax1.plot(qz_roi,fit.data,'s',markersize=4,color='black',label = "data")
    ax1.plot(qz_roi,ref_fit.best_fit,'-',color = "grey",lw = 2,label = "Fit t<0")
    ax1.plot(qz_roi,fit.best_fit,'-',lw = 2,color=color.cmap_blue_red_5(0.99),label = "Best Fit")
        
    ax1.set_xlabel(r"$q_{\mathrm{z}}$" + r" ($\AA^{-1}$)",fontsize = 11)        
    ax1.set_ylabel("X-Ray Intensity",fontsize = 11) 
    
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params('both', length=5, width=1, which='major',direction = 'in',labelsize=11)
    ax1.set_xlim(np.min(qz_roi),np.max(qz_roi)) 
    ax1.legend(loc = 0)
    
    plt.title(str(np.round(delay,2)) + 'ps ' + get_label(ref_file,number)[0], fontsize = 12)
    
    hel.makeFolder('RawData/' + get_label(ref_file,number)[1] +'/Fitgraphics')
    plt.savefig('RawData/' + get_label(ref_file,number)[1] +'/Fitgraphics' +'/'+str(int(10*delay))+'ps.png',dpi =200,bbox_inches = 'tight')
    
    plt.show()
    
def get_peak_dynamic(ref_file,number,unique_delays,com,std,integral,center_fit,width_fit,area_fit): 
    """This function calculate the relative change of the center, width and area of the Bragg peak relative to 
    delays before t0.
    
    Parameters:
    -----------
    uniqueDelays: 1D array
        contains the unique delays of measurement, part of delays without repetition
    center: 1D array
            contains the center of the Bragg peak for every unique delay
    width: 1D array
            contains the width of the Bragg peak for every unique delay
    integral: 1D array
            contains the area of the Bragg peak for every unique delay
    order: integer
            the order of the measured Bragg peak for lattice constant calculation
            
    Returns:
    --------
    strain: array
            relative change of lattice constant to t<0 for different time delays from gaussian fit
    widthrelative: array
            relative change of width of the peak to t<0 for different time delays from gaussian fit
    arearelative: array
            relative change of amplitude of the peak to t<0 for different time delays from gaussian fit
    
    Example:
    --------
    >>> [0.001,0.01,0.09,0.08]
    =getPeakDynamic(delay,center,width,integral,2) """
    
    select_before_t0  = unique_delays < 0
    com_relative      = -1 * hel.relChange(com,np.mean(com[select_before_t0]))
    std_relative      = hel.relChange(std,np.mean(std[select_before_t0]))
    integral_relative = hel.relChange(integral,np.mean(integral[select_before_t0]))
    center_relative   = -1 * hel.relChange(center_fit,np.mean(center_fit[select_before_t0]))
    width_relative    = hel.relChange(width_fit,np.mean(width_fit[select_before_t0]))
    area_relative     = hel.relChange(area_fit,np.mean(area_fit[select_before_t0]))
    
    transient_results = np.zeros((len(unique_delays),9))
    transient_results[:,0] = read_param(ref_file,number)[2]*np.ones(len(unique_delays))
    transient_results[:,1] = read_param(ref_file,number)[2]*np.ones(len(unique_delays))
    transient_results[:,2] = unique_delays
    transient_results[:,3] = 1e3*com_relative
    transient_results[:,4] = 1e2*std_relative
    transient_results[:,5] = 1e2*integral_relative
    transient_results[:,6] = 1e3*center_relative
    transient_results[:,7] = 1e2*width_relative
    transient_results[:,8] = 1e2*area_relative
    
    hel.makeFolder('exportedResults')
    np.savetxt('exportedResults/' + get_label(ref_file,number)[2]+'.dat',transient_results,
               header='0T(K)  1F(mJ/cm²)  2delay(ps)  3strainCOM(permille)  4STDrelative(%)  5Integralrelative(%)  ' 
               + '6strainFit(permille)  7widthFit(%)  8areaFit(%)')
    
    return  transient_results

def plot_transient_results(ref_file,number,transient_results,delay_steps):
    unique_delays     = transient_results[:,2]
    strain_com        = transient_results[:,3]
    strain_fit        = transient_results[:,6]
    std_relative      = transient_results[:,4]
    width_relative    = transient_results[:,7]
    integral_relative = transient_results[:,5]
    area_relative     = transient_results[:,8]
    
    plt.figure(figsize = (14,14))
    gs = gridspec.GridSpec(3, 2,width_ratios=[2,1],height_ratios=[1,1,1],wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    ax5 = plt.subplot(gs[4])
    ax6 = plt.subplot(gs[5])
    
    ax1.plot(unique_delays,strain_com,'s-',color = 'gray',label= "COM (Dy)",linewidth = 2)
    ax1.plot(unique_delays,strain_fit,'o-',color = 'black',label= "Fit (Dy)",linewidth = 2)
    ax2.plot(unique_delays,strain_com,'s-',color = 'gray',label= "COM (Dy)",linewidth = 2)
    ax2.plot(unique_delays,strain_fit,'o-',color = 'black',label= "Fit (Dy)",linewidth = 2)
    ax1.set_ylim(strain_fit.min() - 0.1*(strain_fit.max()-strain_fit.min()),
                 strain_fit.max() + 0.1*(strain_fit.max()-strain_fit.min()))
    ax2.set_ylim(strain_fit.min() - 0.1*(strain_fit.max()-strain_fit.min()),
                 strain_fit.max() + 0.1*(strain_fit.max()-strain_fit.min()))
    ax2.legend(loc = 0,fontsize = 13)
    ax1.set_ylabel('strain ($10^{-3}$)',fontsize=16)
    ax1.yaxis.set_label_coords(-0.07, 0.5)
    
    ax3.plot(unique_delays,std_relative,'-o',color = 'gray',label ="width COM",linewidth = 2)
    ax3.plot(unique_delays,width_relative,'-o',color = 'black',label ="width Fit",linewidth = 2)
    ax4.plot(unique_delays,std_relative,'-o',color = 'gray',label ="width COM",linewidth = 2)
    ax4.plot(unique_delays,width_relative,'-o',color = 'black',label ="width Fit",linewidth = 2)
    ax3.set_ylim(width_relative.min() - 0.1*(width_relative.max()-width_relative.min()),
                 width_relative.max() + 0.1*(width_relative.max()-width_relative.min()))
    ax4.set_ylim(width_relative.min() - 0.1*(width_relative.max()-width_relative.min()),
                 width_relative.max() + 0.1*(width_relative.max()-width_relative.min()))
    ax4.legend(loc = 0,fontsize = 13)
    ax3.set_ylabel('width change ($10^{-2}$)',fontsize=16)
    ax3.yaxis.set_label_coords(-0.07, 0.5)
    
    ax5.plot(unique_delays,integral_relative,'-s',color = 'gray',label ="Area COM",linewidth = 2) 
    ax5.plot(unique_delays,area_relative,'-s',color = 'black',label ="Area Fit",linewidth = 2)    
    ax6.plot(unique_delays,integral_relative,'-s',color = 'gray',label ="Area COM",linewidth = 2)
    ax6.plot(unique_delays,area_relative,'-s',color = 'black',label ="Area Fit",linewidth = 2)
    ax5.set_ylim(area_relative.min() - 0.1*(area_relative.max()-area_relative.min()),
                 area_relative.max() + 0.1*(area_relative.max()-area_relative.min()))
    ax6.set_ylim(area_relative.min() - 0.1*(area_relative.max()-area_relative.min()),
                 area_relative.max() + 0.1*(area_relative.max()-area_relative.min()))
    ax6.legend(loc = 0,fontsize = 13)
    ax5.set_ylabel('amplitude change ($10^{-2}$)',fontsize=16)
    ax5.yaxis.set_label_coords(-0.07, 0.5)
    
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.axhline(y= 0 ,ls = '--',color = "gray",lw = 1)
        ax.tick_params('both', length=4, width=1, which='major',direction = 'in',labelsize=12)
        ax.tick_params(axis='x',which='both',bottom='on',top='on',labelbottom='off')
    for ax in [ax1,ax2]:
        ax.xaxis.set_ticks_position('top')
    for ax in [ax1,ax3,ax5]:
        ax.set_xticks(np.arange(unique_delays.min(),delay_steps[1],delay_steps[0]))
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.axvline(x= delay_steps[1] ,ls = '--',color = "gray",lw = 1)
        ax.axvline(x= 0 ,ls = '--',color = "gray",lw = 1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim([unique_delays.min(),delay_steps[1]])
    
    for ax in [ax2,ax4,ax6]:
        ax.yaxis.tick_right()
        ax.axvline(x= delay_steps[1] ,ls = '--',color = "gray",lw = 1)
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(delay_steps[1],unique_delays.max(),delay_steps[2]))
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(delay_steps[1],unique_delays.max())
        
    ax5.set_xlabel(r"pump-probe delay t (ps)",fontsize = 16)
    ax5.xaxis.set_label_coords(0.75, -0.1)
    
    plt.text(0.25, 1.16, get_label(ref_file,number)[0] + str(read_param(ref_file,number)[3]) + 'K     ' + str(read_param(ref_file,number)[3]) + 'mJ/cm2' , fontsize = 16, ha='left', va='center', transform=ax1.transAxes)     
    plt.savefig('RawData/' + get_label(ref_file,number)[1] +'/OverviewResults.png',dpi =400,bbox_inches = 'tight')
    plt.show()