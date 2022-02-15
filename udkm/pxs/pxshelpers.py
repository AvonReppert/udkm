# -*- coding: utf-8 -*-
"""
This skript contains all basic functions used to evaluate pxs x-ray diffraction data.

-- Further information --

"""
import numpy as np
import numpy.ma as ma
import udkm.tools.helpers as hel
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import lmfit as lm
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

PXS_WL      = 1.5418           #Cu K-alpha wavelength in [Ang]
PXS_K       = 2*np.pi/PXS_WL   #absolute value of the incident k-vector in [1/Ang]
CRYSTAL_OFF = 0.0211595        #offset of crystal diode in [V]

def read_param(ref_file,number):
    """This function reads the reference data file 'ref_file' and returns all parameters
    of the 'number'. measurement including 'date' and 'time' that are used for unique
    reference on the measurement.
    
    Parameters:
    -----------
    ref_file: string 
            file name of the corresponding reference file in the folder 'ReferenceData'
    number: integer
            line in the reference datafile wich corresponds to the measurement
    
    Returns:
    --------
    date: string
            date of the measurement
    time: string
            time of the measurement
    temperature: float
            temperature during the choosen measurement [K]
    fluence: float
            fluence during the choosen measurement [mJ/cm^2]
    distance: float
            distance between sample and detector for choosen measurement [m]
    
    Example:
    --------
    >>>  20191123 = read_param('Overview',0)[0]
    >>>  235959 = read_param('Overview',0)[1]
    >>>  300 = read_param('Overview',0)[2]
    >>>  5.2 = read_param('Overview',0)[3]
    >>>  0.666 = read_param('Overview',0)[4]"""
    
    paramList   = np.genfromtxt(r'ReferenceData/' + str(ref_file) + '.txt', comments = '#')
    date        = str(int(paramList[number,1]))
    date        = (6-len(date))*'0'+date
    time        = str(int(paramList[number,2]))
    time        = (6-len(time))*'0'+time
    temperature = paramList[number,3]
    fluence     = paramList[number,4]
    distance    = paramList[number,6] 
    return date, time, temperature, fluence, distance

def get_label(ref_file,number):
    """This function create a plot title, a saving path and a saving name for the 'number'.
    measurement of the 'ref_file' based on the corresponding parameters of the measurement.
    Therefore this function uses the function 'read_param'.
    
    Parameters:
    -----------
    ref_file: string 
            file name of the corresponding reference file in the folder 'ReferenceData'
    number: integer
            line in the reference datafile wich corresponds to the measurement
    
    Returns:
    --------
    plot_title: string
            contains the 'number' of measurement, 'date', 'time', 'temperature' and 'fluence'
            of the choosen measurement
    save_path: string
            contains the 'date' and 'time' of the choosen measurement as an order structure to define a 
            saving path like the order structure insaved the folder 'RawData'
    file_name: string
            contains the 'date' and 'time' of the choosen measurement to give saved plots and data
            a usefull name
    Example:
    --------
    >>> '0 / 20191123 / 235959 / 300 K / 5.2 mJ/cm2' = get_label('Overview',0)[0]
    >>> '20191123/235959' = get_label('Overview',0)[1]
    >>> '20191123235959' = get_label('Overview',0)[2]"""
    
    plot_title = str(number) +' | ' + read_param(ref_file,number)[0] + ' | ' + read_param(ref_file,number)[1] + ' | '+ str(read_param(ref_file,number)[2]) + 'K' +' | '+ str(read_param(ref_file,number)[3]) + 'mJ/cm2'
    save_path  = read_param(ref_file,number)[0] + '/' + read_param(ref_file,number)[1]
    file_name  = read_param(ref_file,number)[0] + read_param(ref_file,number)[1]
    return plot_title, save_path, file_name

def read_data_rss(ref_file,number,bad_loops,bad_pixels,crystal_off,threshold,t_0):
    """This function imports the x-ray intensity as function of the detector pixel for the 'number'.
    single angle measurement listed in 'ref_file'. The function returns the measurement angle, the 
    delays in respect to temporal overlapp given by 't_0' and the normed intensity on the crystal
    voltage, where low intensity scans below 'threshold', bad pixels and bad loops are excluded 
    given by 'bad_loops' and 'bad_pixels', respectively.
    This function uses the functions 'get_label' and 'read_param'
    
    Parameters:
    -----------
    ref_file: string 
            file name of the corresponding reference file in the folder 'ReferenceData'
    number: integer
            line in the reference datafile wich corresponds to the measurement
    bad_loops: list
            contain all bad loops that are excluded in further analysis
    bad_pixels: list
            contain bad pixels (0-487) that are excluded in further analysis
    crystal_off: float
            voltage at crystal without photons as reference for normalization
    treshold: float
            minimum of crystalvoltage corresponds to a minimum of photons per puls
    t_0: float
            time when pump and probe arrives at the same time [ps]
            
    Returns:
    --------
    omega_deg: 0D array
            incident angle of measurement [deg]
    omega_rad: 0D array
            incident angle of measurement [rad]
    delays: 1D array
            all measured delays with repetion for different loops in respect to 't_0'
    intensity: 2D array
            detected intensities for detectorpixel(x) and delays(y)
    
    Example:
    --------
    >>> 23.2 = read_data_rss('Overview',0,[],[],0.021,0.1,0)[0]
    >>> 0.405 = read_data_rss('Overview',0,[],[],0.021,0.1,0)[1]
    >>> np.array([-1,0,0.5]) = read_data_rss('Overview',0,[],[],0.021,0.1,0)[2]
    >>> np.array(([0,0.5,1,0.5,0],[0,0.5,1,0.5,0],[0,0.6,0.8,0.3,0])) = read_data_rss('Overview',0,[],[],0.021,0.1,0)[3]"""
    
    print(get_label(ref_file,number)[0] + str(read_param(ref_file,number)[4])+ 'm' )
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