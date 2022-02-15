# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:09:31 2019

@author: Maximilian Mattern
"""

import helpers
h = helpers.helpers()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from tqdm import tqdm  #progress bar
import multiprocessing as mp 

import matplotlib.colors as mcolors
import matplotlib.path as mplPath
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1)."""
    
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)
rvb = make_colormap([(16/256,73/256,113/256), (1,1,1) ,0.5, (1,1,1), (180/256,16/256,22/256)])
#%%

def ReadParameter(sample_name,number):
    """This function returns the parameters of the 'number'. measuerement saved
    in the reference file 'Reference'sample_name''.
    
    Parameters:
    -----------
    sample_name: string 
            name of the measured sample (named reference file)
    number: integer
            line in the reference datafile wich corresponds to the measurement
    
    Returns:
    --------
    identifier: integer
            with timestemp from datetime extracted identification of the measurement  
    date: string
            date of the measurement extracted from the reference datafile
    time: string
            time of the measurement extracted from the reference datafile
    temperature: float
            temperature during the choosen measurement [K]
    fluence: float
            fluence during the choosen measurement [mJ/cm^2]
    distance: float
            distance between sample and detector for choosen measurement [m]
    
    Example:
    --------
    >>>  20191111 = readParameter('SRO',0)[1]"""
    
    paramList   = np.genfromtxt(r'ReferenceData/Measurements' + str(sample_name) + '.dat', comments = '#')
    date        = str(int(paramList[number,1]))
    date        = (6-len(date))*'0'+date
    time        = str(int(paramList[number,2]))
    time        = (6-len(time))*'0'+time
    identifier  = int(h.timeStamp(date,time))
    temperature = paramList[number,3]
    fluence     = paramList[number,4]
    distance    = paramList[number,6] 
    return identifier, date, time, temperature, fluence, distance

def GetLabel(sample_name,number):
    """This function returns the title for plotting the result, the saving path
    and saving name of the data and plots. Therfore the function 'ReadParameter' is used.
    
    Parameters:
    -----------
    sample_name: string 
            name of the measured sample (named reference file)
    number: integer
            line in the reference datafile wich corresponds to the measurement
    
    Returns:
    --------
    plot_title: string
            contains the 'sampleName', 'number' of measurement, 'date', 'time',
            'temperature' and 'fluence' of the choosen measurement
    save_path: string
            contains the 'date' and 'time' of the choosen measurement as an folder
            structure to define a saving path like the folder structure in the 
            saved raw data
    date_file_name: string
            contains the 'date' and 'time' of the choosen measurement to give
            saved figures and data a usefull name
    Example:
    --------
    >>> '20181212/235959' = GetLabel('Dataset',0)[1]
    >>> '20181212235959' = GetLabel('Dataset',0)[2]"""
    
    plot_title     = str(number) +' | ' + ReadParameter(sample_name,number)[1] + ' | ' + ReadParameter(sample_name,number)[2]
    plot_title     = plot_title + ' | '+ str(ReadParameter(sample_name,number)[3]) + 'K' +' | '+ str(ReadParameter(sample_name,number)[4]) + 'mJ/cm2'
    save_path      = ReadParameter(sample_name,number)[1] + '/' + ReadParameter(sample_name,number)[2]
    data_file_name = ReadParameter(sample_name,number)[1] + ReadParameter(sample_name,number)[2]
    return plot_title, save_path, data_file_name

def SelectSingleRSM(omega_loop_list,delay_loop_list,delay_list,omega_loop,delay_loop,delay):
    """This function creates a boolean of the length of the three lists corresponding
    to the first columns of the measurement file from the same length, which give the 
    RSM (intensity for every pixel and every measured omega angle) for a
    single omega loop, delay loop and delay.
    
    Parameters:
    -----------
    omega_loop_list: 1D array
        contains all measured omega loops with their measured delay loops and delays
    delay_loop_list: 1D array
        contains all measured delay loops for every omega loop with all delays
    delay_list: 1D array
        contains all delays measured for every omega and delay loop
    omega_loop: float
        the desired omega Loop
    delay_loop: float
        the desired delay Loop
    delay: float
        the desired delay
        
    Returns:
    --------
    select: boolean
        True where omega loop AND delay loop AND delay is the desired value
        
    Example:
    --------
    >>> SelectSingleRSM(np.array([1,1,1,1]),np.array([1,1,2,2]),np.array([0,5,5,0]),1,1,5)
    >>> array([False,True,True,False])"""

    s_omega_loop = omega_loop_list == omega_loop
    s_delay_loop = delay_loop_list == delay_loop
    s_delay      = delay_list == delay
    select       = s_omega_loop & s_delay_loop & s_delay
    return select

def LoadRSMTimeresolved(sample_name,number,crystal_offset,crystal_threshold,t0):
    """This function imports the raw RSMs from the measurement of the 'number'. line
    of the reference file of the sample 'sampleName' for the different omega loops, 
    delay loops and delays. The first dimension of the RSM denotes the detector pixels
    and the second the omega angles. All angle measurements with a photon number below 
    'crystalThreshold' + 'crystalOffset' are replaced by the intensity of the last delay
    and the intensities are normed by the reduced crystal photon number. The RSMs for
    the different delays of the different loops are appended to a list of the structure
    'fullRSMList[t][l][w]'. Here, w denotes the omega loops, l the delay loops and t the delays.
    
    Parameters:
    -----------
    sample_name: string
        name of the measured sample (named reference file)
    number: integer
        line in the reference datafile wich corresponds to the measurement
    crystal_offset: float
        denotes the voltage of the crystal without xray photons
    crystal_threshold: float
        denotes a threshold, all mangle measurements with voltage below become excluded
    t0: float
        denotes the temporal overlapp of pump and probe of the experiment [ps]
        
    Returns:
    --------
    RSM_list: list
        contains a list of RSMs with indizees: [omegaLoop][delayLoop][delay]
    unique_omega: 1D array
        contains the unique measured omgea angle values
    unique_delays: 1D array
        contains the unique measured pump-probe delays
    omega_loops: integer
        reterns the number of measured omega loops
    delay_loops: integer
        returns the number of measured delay loops
            
    Example:
    --------
    >>> LoadRSMTimeresolved('SRO',17,0.017514,crystal_threshold,1)[0]
    >>> [[[array([[50.2,51.2],[0,0.5],[0.2,0.9]]),array([[50.2,51.2],[0.2,0.5],[0.6,1.4]])]]]"""
             
    print('Loading Data RSM: ' + sample_name + ' ' + ReadParameter(sample_name,number)[1] + ' ' + ReadParameter(sample_name,number)[2])    
    RSM_raw       = np.genfromtxt('RawData/' + GetLabel(sample_name,number)[1] + '/scans' + ReadParameter(sample_name,number)[2] + '.dat', comments = "%",delimiter = "\t")
    unique_delays = np.unique(RSM_raw[:,1]-t0)
    unique_omega  = np.unique(RSM_raw[:,3])
    omega_loops   = np.zeros(len(unique_delays))
    delay_loops   = np.zeros(len(unique_delays))
    for t in range(len(unique_delays)):
        s_delay        = RSM_raw[:,1]-t0 == unique_delays[t]
        omega_loops[t] = max(RSM_raw[s_delay,2])
        delay_loops[t] = max(RSM_raw[s_delay,0])
    
    #sorting the entries of the raw RSM with crystal threshhold
    s_photon_num = RSM_raw[:,4] - crystal_offset > crystal_threshold
    print('Above crystal threshhold: ' + str(sum(s_photon_num)) + '/' + str(len(RSM_raw[:,4])))
    for i in range(np.size(RSM_raw[:,0])):
        if s_photon_num[i] == True:
            RSM_raw[i,9:] = RSM_raw[i,9:]
        else:
            if i > len(unique_omega):
                RSM_raw[i,9:] = RSM_raw[i-len(unique_omega),9:]
                RSM_raw[i,5]  = RSM_raw[i-len(unique_omega),5]
            else:
                RSM_raw[i,9:] = RSM_raw[i+len(unique_omega),9:]
    #generate a list of the RSMs for each loop and delay for evaluation
    RSM_list = list()
    for t in range(len(unique_delays)):
        temp_list_delay = list()
        for l in range(int(delay_loops[t])):
            temp_list_loop = list()
            for w in range(int(omega_loops[t])):
                RSM_angle    = np.zeros((np.size(RSM_raw[0,9:]),np.size(unique_omega)))
                s_single_RSM = SelectSingleRSM(RSM_raw[:,2],RSM_raw[:,0],RSM_raw[:,1],w+1,l+1,unique_delays[t])
                single_RSM   = RSM_raw[s_single_RSM,:]
                single_RSM   = single_RSM[single_RSM[:,3].argsort(),:]
                for i in range(len(single_RSM[:,0])):
                    RSM_angle[:,i] = single_RSM[i,9:]
                    RSM_angle[:,i] = RSM_angle[:,i]/(single_RSM[i,4]-crystalOffset)
                temp_list_loop.append(RSM_angle)
            temp_list_delay.append(temp_list_loop)
        RSM_list.append(temp_list_delay)
    return RSM_list, unique_omega, unique_delays, omega_loops, delay_loops

def PlotRSMRaw(sample_name,number,RSM_angle,unique_omega,log_plot):
    """This function plotted a raw reciprocal space map, i.e. intensity for 
    omega angles and detector pixels, whereby the intensity is plotted logarithmic.
    This plot of the 'RSM_raw' can be saved in the data folder in 'Graphics'.
    
    Parameters:
    -----------
    sample_name: string
        name of the measured sample (named reference file)
    number: integer
        line in the reference datafile wich corresponds to the measurement
    RSM_angle: 2D array
        contains the intensities for the measured omega angles (X) and the detector pixels (Y)
    unique_omega: 1D array
        ontains the unique measured omgea angle values for the x-axis o fthe plot
    log_plot: boolean
        set wether the intensity is plotted logarithmic or not
    
    Example:
    --------
    >>> PlotRSMRaw('SRO',17,fullRSMList[0][0][0],np.array([50.2,51.2]),True)
    >>> PlotRSMRaw('SRO',17,array([[50.2,51.2],[0,0.5],[0.2,0.9]]),np.array([50.2,51.2]),True)"""
    
    print('Plotting RSM_angle') 
    h.makeFolder('RawData/' + GetLabel(sample_name,number)[1] + '/ExportedFigures')
    plt.figure(figsize =(6,4))     
    plt.title(GetLabel(sample_name,number)[0], size=12)
    X,Y = np.meshgrid(unique_omega,range(np.size(RSM_angle[:,0])))
    if log_plot:
        plotted = plt.pcolor(X,Y,RSM_angle/RSM_angle.max(),norm=matplotlib.colors.LogNorm(),cmap=rvb)
    else:
        plotted = plt.pcolor(X,Y,RSM_angle/RSM_angle.max(),cmap=rvb)
    plt.colorbar(mappable=plotted)
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    plt.xlabel('omega (deg)')
    plt.ylabel('pixels ')     
    plt.savefig('RawData/' + GetLabel(sample_name,number)[1] + '/ExportedFigures/RSMAngle', bbox_inches='tight',dpi=200)
    plt.show()

def calcCenterPixel(RSM,center_pixel):
    """This function calculates the centerpixel of the measurement, but only if no positive value is manuell
    given. The centerpixel is given by the pixel with the maximum intensity as sum for all omega angles.
        
    Parameters
    ----------
    RSM: 2D numpy array 
        containing a distorted RSM
    center_pixel:  integer
        input centerpixel: If it is positive this is the return value otherwise the calculation starts

    Returns
    ------
    center_pixel: integer
        the pixel with the maximum intensity of the RSMraw or the used defined value
                    
    Example
    ------- 
    >>> 1 = calcCenterPixel(array([[10,11],[1,2],[5,10],[3,4]]),-1)"""
            
    if center_pixel >=0:
        center_pixel = center_pixel
    else:
        center_pixel = np.argmax(np.sum(RSM,1))        
    return center_pixel

def calcQzQx(omega,theta,k):
    """ Calculates the corresponding values qx and qz for a diffraction geometry of omega, theta and k with qz in out-of-plane direction. 
    The calculation is based on equation (4) of  Ultrafast reciprocal-space mapping with a convergent beam , 2013, Journal of Applied Crystallography, 46, 5, 1372-1377. 
    
    Parameters
    ----------
    omega: float 
        incident angle of the X-rays
    theta:  float
        diffraction angle to the detector
    k: float
        wave vector of the X-rays

    Returns
    ------
    qz : float
        the corrsponding qz coordinate in reciprocal space
    qx : float
        the corrsponding qx coordinate in reciprocal space
                    
    Example
    ------- 
    >>> 6.34, 0 = calcQzQx(51,51,4.075)"""
        
    qx = k*(np.cos(theta) - np.cos(omega))
    qz = k*(np.sin(theta) + np.sin(omega))
    return qz,qx

def CalcRSMQList(sample_name,number,RSM_angle,unique_omega,distance,pixel_size,center_pixel,k,convergence_XRays):
    """This function calculates the transformation of teh Raw RSM to reciprocal
    space qx-qz. The detector pixels relative to the center pixel are associated
    with a different diffraction angle theta according to the pixel size and the
    sample-detector distance. The intensity at a pixel and an omega angle is normed
    by the area in angle space given by delta theta and convergence of the X-rays.
    After the transformation in qx-qz space the intensity density is normed by the
    Jacobian matrix of the transformation.
        
    Parameters
    ----------
    sample_name: string
        name of the measured sample (named reference file)
    number: integer
        line in the reference datafile wich corresponds to the measurement
    RSM_angle: 2D numpy array 
        contains the crystal normalized intensity for pixels and omega angles
    unique_omega: 1D numpy array 
        contains the measured omega angle steps
    distance: float
        distance between sample and detector [m]
    pixel_size: float
        dimension of the detector pixels in real space [m]
    center_pixel:  integer
        input centerpixel: If it is positive this is the return value 
        otherwise the calculation starts
    k: float
        wave vector of the X-rays
    convergence_XRays: float
        convergence angle of incident X-rays [rad]

    Returns
    ------
    RSMQ_list: array
        contains the normed intensity distribution in q-space iterating the pixels
        for each unique omega resulting in len(unique_omega)*pixel num lines. 
        Additionally, it contains angle informations and the jacobian matrix
        
    delta_theta:  float
        describes the angle space due to the real space dimension of the pixels [rad]"""
        
    print('Calculating RSMQ_list: ' + sample_name + ' ' + ReadParameter(sample_name,number)[1] + ' ' + ReadParameter(sample_name,number)[2])
    delta_theta     = np.arctan(pixel_size/distance)
    pixels          = range(len(RSM_angle[:,0]))
    center_pixel    = h.calcCenterPixel(RSM_angle,center_pixel)
    RSMQ_list        = np.zeros((np.size(RSM_angle),13))
    _,omegaL,omegaR = h.calcGridBoxes(unique_omega)
    n = 0
    for i in tqdm(range(len(uniqueOmega))):
        for p in pixels:
            RSMQ_list[n,0]  = n
            RSMQ_list[n,1]  = i
            RSMQ_list[n,2]  = p
            RSMQ_list[n,3]  = h.radians(unique_omega[i])
            RSMQ_list[n,4]  = h.radians(omegaL[i])
            RSMQ_list[n,5]  = h.radians(omegaR[i])
            RSMQ_list[n,6]  = RSMQ_list[n,3] + delta_theta*(pixels[p] - center_pixel)
            RSMQ_list[n,7], RSMQ_list[n,8] = calcQzQx(RSMQ_list[n,3],RSMQ_list[n,6],k)
            RSMQ_list[n,9]  = h.calcJacobiDet(RSMQ_list[n,3],RSMQ_list[n,6],k)
            RSMQ_list[n,10] = RSM_angle[p,i]
            RSMQ_list[n,11] = RSMQ_list[n,10]/(delta_theta*convergence_XRays)
            RSMQ_list[n,12] = RSMQ_list[n,11]/RSMQ_list[n,9]
            n += 1
    return RSMQ_list, delta_theta

def CalcPathList(sample_name,RSMQ_list,delta_theta,k):
    """This function creates a list of the qx-qz pixels resulting from the
    omega bins and the delta theta of the pixels. The four corners of the angle bin
    are transformed into qx-qz-space and form an area in reciprocal space.
        
    Parameters
    ----------
    sample_name: string
        name of the measured sample (named reference file)
    RSMQ_list: 2D numpy array 
        contains the intensity as function of the position in reciprocal space and the angles
    delta_theta:  float
        describes the angle space due to the real space dimension of teh pixels [rad]
    k: float
        wave vector of the X-rays

    Returns
    ------
    path_list : list
        list of the arrays describing an area in reciprocal space"""
            
    print('Calculating Polygon Path List: ' + sample_name + ' ' + ReadParameter(sample_name,number)[1] + ' ' + ReadParameter(sample_name,number)[2])
    path_list = []        
    for n in tqdm(RSMQ_list[:,0].astype(int)):
        path_list.append(mplPath.Path(np.array([calcQzQx(RSMQ_list[n,4],RSMQ_list[n,6] - delta_theta,k),calcQzQx(RSMQ_list[n,5],RSMQ_list[n,6] - delta_theta,k),calcQzQx(RSMQ_list[n,5],RSMQ_list[n,6] + delta_theta,k),calcQzQx(RSMQ_list[n,4],RSMQ_list[n,6] + delta_theta,k)])))
    return path_list

def PlotList(sample_name,RSMQ_list,path_list):
    """This function plots the intensity of the Raw RSM in qx-qz-space distributed
    over the area calculated in the path list. The polygons map the reciprocal space. 
    
    Parameters:
    -----------
    sample_name: string
        name of the measured sample (named reference file)
    RSMQ_list: 2D numpy array 
        contains the intensity as function of the position
        in reciprocal space and the angles
    path_list : list
        last of the arrays describing an area in reciprocal space""" 
    
    h.makeFolder('RawData/' + GetLabel(sample_name,number)[1] + '/ExportedFigures')
    patch_list = []        
    for n in RSMQ_list[:,0].astype(int):        
        patch = patches.PathPatch(path_list[n])
        patch_list.append(patch)    
    p = PatchCollection(patch_list,cmap = rvb , array =RSMQ_list[:,12],edgecolors = 'none',norm=matplotlib.colors.LogNorm() )                    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.add_collection(p)
    ax.plot()
    ax.set_xlabel('qz [1/Ang]')
    ax.set_ylabel('qx [1/Ang]')
    cbar = plt.colorbar(p)  
    cbar.set_label(r'X-Ray reflectivity density $I/I_0$  $[\frac{photons Ang^2}{V}]$', rotation=90)
    ax.set_xlim([np.min(RSMQList[:,7]),np.max(RSMQList[:,7])])        
    ax.set_ylim([np.min(RSMQList[:,8]),np.max(RSMQList[:,8])])      
    plt.savefig('RawData/' + GetLabel(sample_name,number)[1] + '/ExportedFigures/PolygonRSM.png', bbox_inches='tight',dpi=200)
    plt.show()

def createQGrid(unique_omega,pixels,qz_min,qz_max,qx_min,qx_max,qz_oversampling,qx_oversampling):
    """This function creates a grid in qx-qz space with bins along both directions.
    The gird along qz is given by the measured omega angles and the number of pixels
    determines the resolution along qx. 
        
    Parameters
    ----------
    unique_omega: 1D numpy array
        contains the measured omega angles
    pixels: 1D numpy array
        contains the detector pixels
    qz_min: float
        describes the lower border of the analysed part of qx-qz space
    qz_max: float
        describes the upper border of the analysed part of qx-qz space
    qx_min: float
        describes the lower border of the analysed part of qx-qz space
    qx_max: float
        describes the upper border of the analysed part of qx-qz space
    qz_oversampling: float
        describes the additional resolution along qz
    qx_oversampling: float
        describes the additional resolution along qx

    Returns
    ------
    qz_grid: 1D numpy array
        contains the steps along qz
    qx_grid: 1D numpy array
        contains the steps along qx
    qz_bins: list
        the boxes with left and right border to sort intensity
    qx_bins: list
        the boxes with left and right border to sort intensity
    qz_delta: float
        width of the bins along qz
    qx_delta: float
        width of the bins along qx"""
    
    print('Creating the Grid in Qspace: ')
    qz_grid = np.linspace(qz_min,qz_max,int(qz_oversampling*len(unique_omega)))
    qx_grid = np.linspace(qx_min,qx_max,int(qx_oversampling*len(pixels))) 
    qz_delta,qzL,qzR = h.calcGridBoxes(qz_grid) 
    qx_delta,qxL,qxR = h.calcGridBoxes(qx_grid)
    qz_bins = np.append(qzL,qzR[-1])       
    qx_bins = np.append(qxL,qxR[-1])
    return qz_grid, qx_grid, qz_bins, qx_bins, qz_delta, qx_delta

def CreateMapping(qzList,qxList,pathList,RSMQList):
    """The points of the recipracl grid which are in the polygon of the 'pathlist' are associated with 
    the intensity from the Raw RSM corresponding to a pixel and an omega angle. THus every point in the 
    QGrid gets an associated intensity.
        
    Parameters
    ----------
    qzList: 1D numpy array
        contains all qz values for each qx value within the QGrid
    qxList: 1D numpy array
        contains all qx values for each qz value within the QGrid
    pathList: list
        contains arrays describing the polygons in reciprocal space
    RSMQList: 2D numpy array
        contains the intensity as function of the position in reciprocal space and the angles

    Returns
    ------
    QGridListTemp: 2D numpy array
        relates every point in QGrid to information in RSMQList"""
    
    points = np.array([qzList,qxList])
    points = points.transpose()
    QGridListTemp = np.zeros((len(qzList),13))
    for i in tqdm(range(0,np.size(pathList))):
        select = pathList[i].contains_points(points)
        QGridListTemp[select,0:13] = RSMQList[i,0:13]
    return QGridListTemp

def CalcQGridList(qzGrid,qxGrid,pathList,RSMQList):
    """This function creates a list of the QGrid, where each point of the QGrid is associated with 
    an intensity as in the RSMQList.
        
    Parameters
    ----------
    qzGrid: 1D numpy array
        contains the steps along qz
    qxGrid: 1D numpy array
        contains the steps along qx
    pathList: list
        contains arrays describing the polygons in reciprocal space
    RSMQList: 2D numpy array
        contains the intensity as function of the position in reciprocal space and the angles

    Returns
    ------
    QGridList: 2D numpy array
        relates every point in QGrid to information in RSMQList"""
    QGridList = np.zeros((np.size(qzGrid)*np.size(qxGrid),18))
    n = 0 
    for lx in tqdm(range(np.size(qxGrid))):
        for lz in range(np.size(qzGrid)):
            QGridList[n,0] = n
            QGridList[n,1] = lz
            QGridList[n,2] = lx
            QGridList[n,3] = qzGrid[lz]
            QGridList[n,4] = qxGrid[lx]
            n += 1
    QGridListTemp = CreateMapping(QGridList[:,3],QGridList[:,4],pathList,RSMQList)
    QGridList[:,5:18] = QGridListTemp[:,0:13]
    return QGridList

def SortListToRSMQ(qxGrid,qzGrid,QGridList):
    """This function generates on the basis of 'QGrisList' a two dimensional RSM.
        
    Parameters
    ----------
    qzGrid: 1D numpy array
        contains the steps along qz
    qxGrid: 1D numpy array
        contains the steps along qx
    QGridList: 2D numpy array
        relates every point in QGrid to information in RSMQList
        
    Returns
    ------
    RSMQ: 2D numpy array
        intensity in qx (lines) and qz (columns)"""
        
    RSMQ = np.zeros((np.size(qxGrid),np.size(qzGrid)))
    for n in range(np.size(qxGrid)*np.size(qzGrid)):       
        RSMQ[int(QGridList[n,2]),int(QGridList[n,1])] = QGridList[n,17]
    return RSMQ

def plotRSMQ(sampleName,number,RSMQ,qzBins,qxBins,delay):
    """This function plots the RSM with orthogonal bins.
        
    Parameters
    ----------
    sampleName: string 
        name of the measured sample (named reference file)
    number: integer
        line in the reference datafile wich corresponds to the measurement
    RSMQ: 2D numpy array
        intensity in qx (lines) and qz (columns)
    qzBins: list
        the boxes with left and right border to sort intensity
    qxBins: list
        the boxes with left and right border to sort intensity
    delay: float
        the element of uniqueDelays corresponding to RSM""" 
        
    print('Plotting RSMQ ... ') 
    h.makeFolder('RawData/' + GetLabel(sampleName,number)[1] + '/ExportedFigures')       
    plt.figure()
    X,Y= np.meshgrid(qzBins,qxBins)
    plt.pcolor(X,Y,RSMQ,norm=matplotlib.colors.LogNorm(),cmap=rvb,vmin=1e5, vmax=1e8)
    plt.axis([X.min(), X.max(), Y.min(), Y.max()])
    cbar = plt.colorbar()
    cbar.set_label(r'X-Ray reflectivity density $I/I_0$  $[\frac{photons Ang^2}{V}]$', rotation=90)
    plt.xlabel('qz [ 1/Ang ]')
    plt.ylabel('qx [ 1/Ang ]')
    plt.grid('on')
    plt.savefig('RawData/' + GetLabel(sampleName,number)[1] + '/ExportedFigures/RSM.png', bbox_inches='tight',dpi=200)
    plt.show()

def SortRSMRawToQGridList(QGridList,RSMRaw,deltaTheta,convergenceXRays,pixels,uniqueOmega):
    """This function uses 'QGridList' and sorts the new intensities from 'RSMRaw' into the bins.
        
    Parameters
    ----------
    QGridList: 2D numpy array
        relates every point in QGrid to information in RSMQList
    RSMRaw: 2D numpy array
        contains intensity as function of pixels and omega angles
    deltaTheta:  float
        describes the angle space due to the real space dimension of teh pixels [rad]
    convergenceXRays: float
        convergence angle of incident X-rays [rad]
    pixels: 1D numpy array
        contains the detector pixels
    uniqueOmega: 1D array
        contains the unique measured omgea angle values
       
    Returns
    ------
    QGridList: 2D numpy array
        relates every point in QGrid to intensity in RSMRaw""" 
        
    for n in QGridList[:,0].astype(int): 
        if QGridList[n,5] > 0 and  QGridList[n,7] < len(pixels) and QGridList[n,6] < len(uniqueOmega) :
            QGridList[n,15] = RSMRaw[int(QGridList[n,7]),int(QGridList[n,6])]
            QGridList[n,16] = QGridList[n,15]/(deltaTheta*convergenceXRays)
            QGridList[n,17] = QGridList[n,16]/QGridList[n,14]
    return QGridList

def GetRockingCurveRSMQ(RSMQ,direction,qzDelta,qxDelta,qzRange,qxRange):
    if direction == 'z':     
        rockingCurve = np.sum(RSMQ,axis = 0)*qzDelta*qxRange
    else:
        rockingCurve = np.sum(RSMQ,axis = 0)*qxDelta*qzRange
    return rockingCurve

def Write1DArrayToFileWithoutLineBreak(f,Array):
    for i in range(0, len(Array)):
        f.write(str(Array[i]) + '\t')
        
def Write1DArrayToFile(f,Array):
    for i in range(0, len(Array)):
        if i != (len(Array) - 1):
            f.write(str(Array[i]) + '\t')
        else:
            f.write(str(Array[i]) + '\n')

def PrepareRockingCurveFile(sampleName,number,qAxis):        
    """ Starts the files where all Rocking-Curves along qz are stored """
    h.makeFolder('ExportedRockingCurves/' + ReadParameter(sampleName,number)[1])
    header = [ "0 Delay","1 DelayLoop","2 OmeagaLoop", "3 -end Rockingcurve"]    
    rockingCurveFileName = 'ExportedRockingCurves/' + ReadParameter(sampleName,number)[1] +'/qzRockingCurves' + ReadParameter(sampleName,number)[2] + '.dat'             
    rockingCurveFile = open(rockingCurveFileName, 'w')
    rockingCurveFile.write('% ')
    Write1DArrayToFile(rockingCurveFile,header)
    Write1DArrayToFileWithoutLineBreak(rockingCurveFile,[-1,-1,-1])
    Write1DArrayToFile(rockingCurveFile,qAxis)
    return rockingCurveFile
#%%
''' PXS properties influencing the transformation to reciprocal space '''    
pixelsize        = 0.172*1e-3
k                = h.PXSk
convergenceXRays = h.radians(0.3)
crystalOffset    = 0.017514
crystalThreshold = 0.01
centerpixel      = -1
#distance = 1.477

''' Choosing the measurement and time zero '''
number   = 6
t0       = 0
log_plot = True

''' Choosing the omega loop, delay loop and delay '''
w = 0
l = 0 
t = 0


''' Choosing the part of reciprocal space for the rocking curve '''
qzMax = 4.0
qzMin = 4.25
qxMax = 0.05
qxMin = -0.05

'''Data Evaluation Routine'''
# Read in the Raw RSMs for each loop and delay
fullRSMList, uniqueOmega, uniqueDelays, omegaLoopsMeasured, delayLoopMeasured = LoadRSMTimeresolved('FeRh',number,crystalOffset,crystalThreshold,t0) 
# Plot the Raw RSM (pixel-omega-space) of the RSM[omega loop][delay loop][delay]
PlotRSMRaw('FeRh',number,fullRSMList[t][l][w],uniqueOmega,log_plot) 
# Determine the sample-detector distance in meter from the reference file
distance = ReadParameter('FeRh',number)[5]
# Transform the Raw RSM into the qx-qz-space with normalization of the intensity density with Jacobian matrix
RSMQList, deltaTheta = CalcRSMQList('FeRh',number,fullRSMList[t][l][w],uniqueOmega,distance,pixelsize,centerpixel,k,convergenceXRays)
# Calculate an area in reciprocal space corresponding to the omega bins and the delta theta acceptance of the detector pixels
pathList = CalcPathList('FeRh',RSMQList,deltaTheta,k)
# Plot the intensity sorted in the area in reciprocal space of the pathlist. 
PlotList('FeRh',RSMQList,pathList)

# Create the orthogonal grid in qx-qz space
qzGrid, qxGrid, qzBins, qxBins, qzDelta, qxDelta = createQGrid(uniqueOmega,range(len(fullRSMList[w][l][t][:,0])),qzMin,qzMax,qxMin,qxMax,4,1)

# Calculate for every point in the QGrid the intensity according to the polygon where it is in
refQGridList = CalcQGridList(qzGrid,qxGrid,pathList,RSMQList)
# Sort the intensity list to a two-dimensional representation
refRSMQ = SortListToRSMQ(qxGrid,qzGrid,refQGridList)
# Plot the RSM with the intensity sorted into orthogonal bins
plotRSMQ('FeRh',number,refRSMQ,qzBins,qxBins,uniqueDelays[t])

#%%

rockingCurveFile = PrepareRockingCurveFile('FeRh',number,qzGrid)
for t in range(len(uniqueDelays)):
    RSMQmean = np.zeros((len(refRSMQ[:,0]),len(refRSMQ[0,:])))
    for w in range(int(omegaLoopsMeasured[t])):
        counter = 0
        for l in range(int(delayLoopMeasured[t])):
            print('doing step w= ' + str(w+1)+'/'+ str(int(omegaLoopsMeasured[t])) + ' l= '+ str(l+1) + '/' + str(int(delayLoopMeasured[t])) + ' t = ' + str(t+1) +'/' + str(len(uniqueDelays)))
            QGridList = SortRSMRawToQGridList(refQGridList,fullRSMList[t][l][w],deltaTheta,convergenceXRays,range(len(fullRSMList[0][0][0][:,0])),uniqueOmega)
            RSMQ = SortListToRSMQ(qxGrid,qzGrid,QGridList)
            RSMQmean = RSMQmean + RSMQ
            #plotRSMQ('FeRh',number,RSMQ,qzBins,qxBins,uniqueDelays[t])
            rockingCurve = GetRockingCurveRSMQ(RSMQ,'z',qzDelta,qxDelta,qzMax-qzMin,qxMax-qxMin)
            Write1DArrayToFileWithoutLineBreak(rockingCurveFile,[uniqueDelays[t],l,w])            
            Write1DArrayToFile(rockingCurveFile,rockingCurve)
            #plt.plot(qzGrid,rockingCurve,label=str(uniqueDelays[t]))
            counter = counter +1
    h.makeFolder('ExportedRSM/' + GetLabel('FeRh',number)[2])
    RSMQmeanF = RSMQmean/counter
    RSMQExport        = np.zeros((len(RSMQmeanF[:,0])+2,len(RSMQmeanF[0,:])+2))
    RSMQExport[1:,0]  = qxBins
    RSMQExport[0,1:]  = qzBins
    RSMQExport[2:,2:] = RSMQmeanF
    np.savetxt('ExportedRSM/' + GetLabel('FeRh',number)[2] + '/RSMQ_'  + str(int(10*uniqueDelays[t])) + 'ps.dat',RSMQExport,header='qx bins and qz bins',comments='#',delimiter='\t')
    
rockingCurveFile.close()
