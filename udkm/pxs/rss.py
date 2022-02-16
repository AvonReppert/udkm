# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:25:23 2022

@author: matte
"""

import numpy as np
import numpy.ma as ma
import udkm.tools.helpers as h
import numpy as np
import matplotlib.pyplot as plt
import numpy.ma as ma
import lmfit as lm
import matplotlib
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter
import helpers
hel = helpers.helpers()
import matplotlib.colors as mcolors


def ReadParameter(sampleName,number):
    """This function returns the parameters of the 'number'. measuerement saved in the reference file 'Reference'sampleName''.
    
    Parameters:
    -----------
    sampleName: string 
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
    
    paramList   = np.genfromtxt(r'ReferenceData/Parameters' + str(sampleName) + '.txt', comments = '#')
    date        = str(int(paramList[number,1]))
    date        = (6-len(date))*'0'+date
    time        = str(int(paramList[number,2]))
    time        = (6-len(time))*'0'+time
    identifier  = int(h.timeStamp(date,time))
    temperature = paramList[number,3]
    fluence     = paramList[number,4]
    distance    = paramList[number,6] 
    return identifier, date, time, temperature, fluence, distance

def GetLabel(sampleName,number):
    """This function returns the title for plotting the result, the saving path and saving name of the data and plots. 
    Therfore the function 'ReadParameter' with its returns is used.
    
    Parameters:
    -----------
    sampleName: string 
            name of the measured sample (named reference file)
    number: integer
            line in the reference datafile wich corresponds to the measurement
    
    Returns:
    --------
    plotTitle: string
            contains the 'sampleName', 'number' of measurement, 'date', 'time', 'temperature' and 'fluence'
            of the choosen measurement
    savePath: string
            contains the 'date' and 'time' of the choosen measurement as an order structure to define a 
            saving path like the order structure insaved raw data
    dateFileName: string
            contains the 'date' and 'time' of the choosen measurement to give saved plots and data
            a usefull name
    Example:
    --------
    >>> '20181212/235959' = getLabel('Dataset',0)[1]
    >>> '20181212235959' = getLabel('Dataset',0)[2]"""
    
    plotTitle    = str(number) +' | ' + ReadParameter(sampleName,number)[1] + ' | ' + ReadParameter(sampleName,number)[2]
    plotTitle    = plotTitle + ' | '+ str(ReadParameter(sampleName,number)[3]) + 'K' +' | '+ str(ReadParameter(sampleName,number)[4]) + 'mJ/cm2'
    savePath     = ReadParameter(sampleName,number)[1] + '/' + ReadParameter(sampleName,number)[2]
    dataFileName = ReadParameter(sampleName,number)[1] + ReadParameter(sampleName,number)[2]
    return plotTitle, savePath, dataFileName

def ReadDataOneAngle(nameFile,number,excludedLoops,crystalOffset,threshold,t0,delayFirstbool):
    """This function reads the raw data from the single angle measurement and extracts the omega angle
    the delays and the pixel dependent intensities. Bad Loops ('excludedLoops') become excluded and
    the intensities above 'threshold' become normed by the crystal voltage.
    
    Parameters:
    -----------
    sampleName: string 
            name of the reference datafile (named reference file)
    number: integer
            line in the reference datafile wich corresponds to the measurement
    excludedLoops: list
            contain all bad loops
    crystalOffset: float
            voltage at crystal without photons
    treshold: float
            minimum of crystalvoltage corresponds to a minimum of photons per puls
    t0: float
            time when pump and probe arrives at the same time [ps]
    delayFirstbool: boolean
            if true in the first column are the loop delays
            
    Returns:
    --------
    omegaR: 0D array
            incident angle of measurement [rad]
    delays: 1D array
            all measured delays with repetion for different loops reduced by t0
    intensities: 2D array
            detected intensities for detectorpixel(x) and delays(y)
    
    Example:
    --------
    >>> 0.36652 = readDataOneAngle('SRO',0,[],0.01,0.1,0,True)[0]"""
    
    print(GetLabel(nameFile,number)[0]+str(int(t0))+'ps ' + str(crystalOffset)+'V ' + str(ReadParameter(nameFile,number)[5])+ 'm' )
    dataRaw = np.genfromtxt('RawData/' + str(GetLabel(nameFile,number)[1]) + '/scans' + str(ReadParameter(nameFile,number)[2]) + '.dat', comments = "%",delimiter = "\t")
    dataRaw = dataRaw[dataRaw[:,4] - crystalOffset > threshold,:]
    if delayFirstbool:
        for ii in range(np.size(excludedLoops)):
            dataRaw = dataRaw[dataRaw[:,0]!=excludedLoops[ii],:]
        delays = dataRaw[:,1]-t0
        angles = dataRaw[:,3]
    else: 
        for ii in range(np.size(excludedLoops)):
            dataRaw = dataRaw[dataRaw[:,2]!=excludedLoops[ii],:]
        angles = dataRaw[:,1]
        delays = dataRaw[:,3]-t0
    crystalV = dataRaw[:,4] - crystalOffset
    intensities = dataRaw[:,9:]
    for i in range(np.size(delays)):
        intensities[i,:] = dataRaw[i,9:]/crystalV[i]
    omegaR = np.radians(np.unique(angles))
    return omegaR, delays, intensities

def QzAxisOneAngle(sampleName,number,omegaR,intensities):  
    """This function returns an angle and qz axis for the corresponding intensities. The relevant constants are 
    the pixel dimension, the wavenumber of the PXS and the distance between the sample and detector. The wavenumber
    and pixel dimension are constant. The centerpixel is determined by the pixel in the middle of detector. The qz
    values are calculated by the usual nonlinear transformation with 'omegaR' from 'ReadDataOneAngle' and theta.
    
    Parameters:
    -----------
    sampleName: string 
            name of the reference datafile (named reference file)
    number: integer
            line in the reference datafile wich corresponds to the measurement
    omegaR: 0D array
            the incident angle of x-ray beam
    intensities: 2D array
            contains Intensity of detector depends on pixel(x) and timedelay(y)
            
    Returns:
    --------
    thetaAxis: 1D array
            the reflection angles for the different detector pixels
    qzAxis: 1D array
            with incident and refelction angle nonlinear calculated axis with momentum change

    Example:
    --------
    >>> [2.1,2.15,2.21] = qzAxisOneAngle('SRO',0,0.36652,intensity)[1]"""
    
    k                     = h.PXSk
    PixelDimension        = 0.172
    pixelAxis             = np.arange(0,np.size(intensities[0,:]),1)
    centerpixel           = int(np.max(pixelAxis)/2)
    DistancePilatusSample = ReadParameter(sampleName,number)[5]*1000
    DeltaTheta            = np.arctan(PixelDimension/DistancePilatusSample)
    thetaAxis             = omegaR + DeltaTheta*(pixelAxis - centerpixel) 
    qzAxis                = k*(np.sin(thetaAxis) + np.sin(omegaR))
    return thetaAxis, qzAxis

def GetRockingCurvesOneAngle(sampleName,number,omegaR,delays,intensities,badpixels): 
    """ This function returns rocking curves in dependency of theta and qz for every measured time delay and the 
    Intensities for the unique delays are averaged. These rockingccurves are saved as .txt files. The function
    qzAxisOneAngle is used for the calculation of the correspondig axis.
    
    Parameters:
    -----------
    sampleName: string 
            name of the reference datafile
    number: integer
            line in the reference datafile wich corresponds to the measurement
    omegaR: 0D array
            incident angle of measurement [rad]
    delays: 1D array
            all measured delays with repition for different loops
    intensities: 2D array
            detected intensities for detectorpixel(x) and delays(y)
    badpixels: list
            contains the number of all known bad pixels
            
    Returns:
    --------
    rockingCurves: 2D array
            contains the averaged intensities for the unique measured delays(x) in dependency of qz(y1) and the
            angle theta(y2)
    
    Example:
    --------
    >>> RockCurv=
    getRockingCurve('Dataset',0,0.123,t,I,[])"""
    
    uniqueDelays    = np.unique(delays) 
    pixelAxis       = np.arange(0,np.size(intensities[0,:]),1)   
    MaskbadPixels   = np.zeros(np.size(pixelAxis))
    for b in badpixels:
        MaskbadPixels[b] = 1
    selectbadPixel  = MaskbadPixels !=1
    qzAxisExport    = QzAxisOneAngle(sampleName,number,omegaR,intensities)[1][selectbadPixel] 
    rockingCurves      = np.zeros( [len(uniqueDelays)+1,np.size(selectbadPixel)+1])
    rockingCurves[0,:] = np.append([-5000],qzAxisExport)
    rockingCurves[:,0] = np.append(np.array([-5000]),uniqueDelays)
    counter = 0
    for d in uniqueDelays:
        selectDelay = delays == d
        rockingCurves[counter+1,1:] = ma.masked_array(np.mean(intensities[selectDelay,:],0), mask=MaskbadPixels)
        counter +=1
    h.makeFolder('exportedRockingCurves')
    h.writeListToFile('RawData/' + GetLabel(sampleName,number)[1] +'/rockingCurves'+GetLabel(sampleName,number)[2]+'.dat','Col1 = qz, Col2 theta, Row1 Delay, Other cols are Intensites',rockingCurves)
    h.writeListToFile('exportedRockingCurves/'+ GetLabel(sampleName,number)[2] + '.dat','Col1 = qz, Col2 theta, Row1 Delay, Other cols are Intensites',rockingCurves) 
    return rockingCurves

def GetMoments(RockingCurves,qzMin,qzMax):  
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
    
    uniqueDelays = RockingCurves[1:,0]
    selectD      = uniqueDelays<0
    delayst0     = uniqueDelays[selectD]
    qz           = RockingCurves[0,1:]
    Intensities  = RockingCurves[1:,1:]
    qzROI = qz[np.logical_and(qz > qzMin, qz < qzMax)]
    IntensitiesROI = Intensities[:,np.logical_and(qz > qzMin, qz < qzMax)]
    COM          = np.zeros(np.size(uniqueDelays))
    STD          = np.zeros(np.size(uniqueDelays))
    Integral     = np.zeros(np.size(uniqueDelays))
    for i in range(np.size(delayst0)):
        COM[i], STD[i], Integral[i] = h.calcMoments(qzROI,IntensitiesROI[selectD,:][i,:])
    resultFitRef = GaussianBackFit(np.mean(COM[selectD]),np.mean(STD[selectD]),np.max(IntensitiesROI[selectD,:]),0,0,qzROI,np.mean(IntensitiesROI[selectD,:],0))
    for i in range(np.size(uniqueDelays)):
        IntensitiesROI_corr1        = IntensitiesROI[i,:]-(qzROI*resultFitRef.values["slope"]+resultFitRef.values["intercept"])
        IntensitiesROI_corr2        = IntensitiesROI_corr1 - np.min(IntensitiesROI_corr1)
        COM[i], STD[i], Integral[i] = h.calcMoments(qzROI,IntensitiesROI_corr2)
    return COM, STD, Integral, resultFitRef, uniqueDelays

def getFitvalues(NameFile,Number,RockingCurves,qzMin,qzMax,resultFitRef,plotFitResults,PlotSemilogy,fontSize):
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
    
    uniqueDelays = RockingCurves[1:,0]
    qz           = RockingCurves[0,1:]
    Intensities  = RockingCurves[1:,1:]
    qzROI = qz[np.logical_and(qz > qzMin, qz < qzMax)]
    IntensitiesROI = Intensities[:,np.logical_and(qz > qzMin, qz < qzMax)]
    centerFit    = np.zeros(np.size(uniqueDelays))
    widthFit     = np.zeros(np.size(uniqueDelays))
    areaFit      = np.zeros(np.size(uniqueDelays))
    for i in range(np.size(uniqueDelays)):
        resultFit    = GaussianBackFit(resultFitRef.values["g_center"],resultFitRef.values["g_sigma"],resultFitRef.values["g_amplitude"],
                                    resultFitRef.values["slope"],resultFitRef.values["intercept"],qzROI,IntensitiesROI[i,:])
        centerFit[i] = resultFit.values["g_center"] 
        widthFit[i]  = resultFit.values["g_sigma"]
        areaFit[i]   = resultFit.values["g_amplitude"]
        t            = uniqueDelays[i]
        if plotFitResults == True:
            plotPeakFit(NameFile,Number,i,qzROI,resultFit,resultFitRef,t,PlotSemilogy,fontSize)
    return centerFit, widthFit, areaFit, uniqueDelays

def getPeakDynamic(uniqueDelays,center,width,integral,order): 
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
    
    selectD         = uniqueDelays<0
    latticeConstant = order*2*np.pi/center 
    strain          = h.relChange(latticeConstant,np.mean(latticeConstant[selectD]))
    widthrelative   = h.relChange(width,np.mean(width[selectD]))
    arearelative    = h.relChange(integral, np.mean(integral[selectD]))
    return  strain, widthrelative, arearelative


def GaussianBackFit(centerStart,sigmaStart,amplitudeStart,slopeStart,interceptStart,qzROI,IntensityROI):
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
    
    model     = lm.models.GaussianModel(prefix = "g_") + lm.models.LinearModel() 
    pars      = lm.Parameters()                                                      
    pars.add_many(('g_center', centerStart, True, np.min(qzROI), np.max(qzROI)),
                ('g_sigma', sigmaStart, True),
                ('g_amplitude', amplitudeStart , True),
                ('slope', slopeStart, True),
                ('intercept', interceptStart, True))
    resultFit = model.fit(IntensityROI, pars, x = qzROI)
    return resultFit

def GaussianFit(centerStart,sigmaStart,amplitudeStart,qzROI,intensityROI,variationBool):
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
    pars.add_many(('g_center', centerStart, variationBool[0]),
                  ('g_sigma', sigmaStart, variationBool[1]),
                  ('g_amplitude', amplitudeStart , variationBool[2]))
    resultFit = model.fit(intensityROI, pars, x = qzROI)
    return resultFit

def exportPeakDynamic(NameFile,Number,uniqueDelays,strainFit,widthFitrelative,areaFitrelative,strainCOM,STDrelative,Integralrelative):
    """this function exports the results of the peak dynamic. For every delay the relative change of the moments and fit values 
    are saved together with the measuremnt parameters imported with the function readParameter.
    
    Parameters:
    -----------
    NameFile: string 
            name of the reference datafile
    Number: integer
            line in the reference datafile wich corresponds to the measurement
    uniqueDelays: 1D array 
            contains the unique delays of measurement, part of delays without repetition
    strainFit: 1D array
            contains the relative change of latticeconstant for the unique delays (see function getPeakDynamic)
    widthFitrelative: 1D array
            contains the relative change of width of the gaussian fit for the unique delays (see function getPeakDynamic)
    areaFitrelative: 1D array
            contains the relative change of amplitude of gaussain fit for the unique delays (see function getPeakDynamic)
    strainCOM: 1D array
            contains the relative change of center of mass for the unique delays (see function getPeakDynamic)
    STDrelative: 1D array
            contains the relative change of standart derivation for the unique delays (see function getPeakDynamic)
    Integralrelative: 1D array
            contains the relative change of center of integral for the unique delays (see function getPeakDynamic)


    Example:
    --------
    >>> exportPeakDynamic('Dataset',0,delay,sFit,wFit,aFit,sCOM,wCOM,aCOM)""" 
    
    exportFileName1 ='exportedResults/'+GetLabel(NameFile,Number)[2]+'.dat'
    exportFileName2 ='RawData/' + GetLabel(NameFile,Number)[1]+'/results'+GetLabel(NameFile,Number)[2]+'.dat'
    h.writeLinesToFile(exportFileName1,'0ID    1date   2time    3T(K)    4F(mJ/cm²)    5delay(ps) 6strainCOM(perMille)  7STDrelative(%)    8Integralrelative(%)  9strainFit(perMille)   10widthFit(%)    11areaFit(%)',[])
    h.writeLinesToFile(exportFileName2,'0ID    1date   2time    3T(K)    4F(mJ/cm²)    5delay(ps) 6strainCOM(perMille)  7STDrelative(%)    8Integralrelative(%)  9strainFit(perMille)   10widthFit(%)    11areaFit(%)',[])
    for i in range(np.size(uniqueDelays)):
        t = uniqueDelays[i]
        line =str(int(ReadParameter(NameFile,Number)[0]))+'\t'+ReadParameter(NameFile,Number)[1]+'\t'+ReadParameter(NameFile,Number)[2] + '\t'+str(int(ReadParameter(NameFile,Number)[3]))+'\t'+str(int(ReadParameter(NameFile,Number)[4]))+'\t'+str(t)+'\t'+str(strainCOM[i]*1e3)+'\t'+str(STDrelative[i]*1e2)+'\t' +str(Integralrelative[i]*1e2)+'\t'+str(strainFit[i]*1e3)+'\t'+str(widthFitrelative[i]*1e2)+'\t' +str(areaFitrelative[i]*1e2)
        h.appendStringLineToFile(exportFileName1,line)     
        h.appendStringLineToFile(exportFileName2,line)
        
def PlotRockingCurve1DUXRD(sampleName,number,rockingCurves,fontSize,plotSemilogy,additionalProperty): 
    """This function plot the generated Rocking Curves in 1D for every unique delay to control the flow of the script.
    
    Parameters:
    -----------
    sampleName: string 
            name of the reference datafile
    number: integer
            line in the reference datafile wich corresponds to the measurement
    rockingCurves: 2D array
            contains the averaged intensities for the unique measured delays(x) in dependency of qz(y1) and the
            angle theta(y2)
    fontSize: integer
            the fontsize for the plots
    plotSemilogy: boolean
            if the peak should plotted on a logarithmic scale or not
    
    Example:
    --------
    >>> plotRockingCurve1D('Dataset',0,Array,18)"""
    
    uniqueDelays = np.unique(rockingCurves[1:,0])
    qzaxis = rockingCurves[0,3:]
    intensities = rockingCurves[1:,3:]
    plt.figure(figsize =(12,6))
    plt.title(GetLabel(sampleName,number)[0], fontsize = fontSize+2)        
    plt.xlabel(r'$q_z$ ($\mathrm{\AA^{-1}}$)',fontsize = fontSize)
    plt.ylabel('X-Ray Intensity',fontsize = fontSize)
    plt.tick_params(axis='both', which='major', labelsize=fontSize-3)
    plt.xlim(np.min(qzaxis),np.max(qzaxis))
    plt.grid()
    counter = 0
    if plotSemilogy == True:
        for j in range(len(uniqueDelays)):
            plt.semilogy(qzaxis,intensities[j,:],color = CMap(counter/np.size(uniqueDelays)),lw=4)
            plt.semilogy(qzaxis,np.mean(intensities[uniqueDelays < 0,:],0),'--k',linewidth =3 )
            counter +=1
    else:
        for j in range(len(uniqueDelays)):
            plt.plot(qzaxis,intensities[j,:],color =CMap(counter/np.size(uniqueDelays)),lw=4)
            plt.plot(qzaxis,np.mean(intensities[uniqueDelays <0,:],0),'--k',linewidth =3 )
            counter +=1
    plt.savefig('RawData/' + GetLabel(sampleName,number)[1] +'/rockingCurves' + additionalProperty + '.png',dpi =200,bbox_inches = 'tight')
    plt.show()
    
def PlotRockingCurve2DUXRD(sampleName,number,rockingCurves,t_split): 
    qz = rockingCurves[0,1:]
    delays = rockingCurves[1:,0]
    intensity = rockingCurves[1:,1:]
     
    f       = plt.figure(figsize = (8,8))
    gs      = gridspec.GridSpec(2, 1,width_ratios=[1],height_ratios=[1,2],wspace=0.0, hspace=0.0)
    ax1     = plt.subplot(gs[0])
    ax2     = plt.subplot(gs[1])        
    X,Y     = np.meshgrid(qz,delays)
    
    plotted = ax2.pcolormesh(X,Y,intensity/np.max(intensity),norm=matplotlib.colors.LogNorm(vmin = 0.006, vmax = 1),shading='auto',cmap = rvb)
    ax2.axis([qz.min(), qz.max(), delays.min(), t_split])
    ax2.set_xlabel(r'$q_z$ ($\mathrm{\AA^{-1}}$)',fontsize = 13)
    ax2.set_ylabel('delay (ps)', fontsize = 14)
    ax2.yaxis.set_label_coords(-0.07, 0.75) 
    ax2.axhline(y = 0 ,ls = '--',color = "gray",lw = 1)
    ax2.axhline(y = t_split ,ls = '--',color = "gray",lw = 1)

    plotted = ax1.pcolormesh(X,Y,intensity/np.max(intensity),norm=matplotlib.colors.LogNorm(vmin = 0.006, vmax = 1),shading='auto',cmap = rvb)
    ax1.xaxis.set_ticks_position('top')
    ax1.axis([qz.min(), qz.max(), t_split, delays.max()])   
    
    ax1.tick_params(axis='y',which='both',bottom='off',top='on',labelbottom='off')
    ax1.tick_params('both', length=5, width=1, which='major',direction = 'in',labelsize=11)
    ax2.tick_params('both', length=5, width=1, which='major',direction = 'in',labelsize=11)
    ax2.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
          
    cbaxes = f.add_axes([0.925, 0.125, 0.023, 0.775])  
    cbar = plt.colorbar(plotted,cax = cbaxes, orientation='vertical')  
    cbar.set_label(r' X-Ray Intensity', rotation=90,fontsize = 12)
    plt.text(0.15, 1.2, GetLabel(sampleName,number)[0], fontsize = 14, ha='left', va='center', transform=ax1.transAxes)     
    plt.savefig('RawData/' + GetLabel(sampleName,number)[1] +'/rockingCurves2Dplot.png',dpi =400,bbox_inches = 'tight') #Saves the image
    plt.show()   
    
def PlotTransientResults(sampleName,number,exportedData,tlim,widthRatio,ylim,xticks1,xticks2):
    delays    = exportedData[:,0]
    strainCOM = exportedData[:,1]
    strainFit = exportedData[:,4]
    widthRelative = exportedData[:,2]
    widthFitRelative = exportedData[:,5]
    integralRelative = exportedData[:,3]
    areaFitrelative = exportedData[:,6]
    
    plt.figure(figsize = (14,14))
    gs = gridspec.GridSpec(3, 2,width_ratios=widthRatio,height_ratios=[1,1,1],wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])
    ax5 = plt.subplot(gs[4])
    ax6 = plt.subplot(gs[5])
    
    ax1.plot(delays,strainCOM*1e3,'s-',color = 'gray',label= "COM (Dy)",linewidth = 2)
    ax1.plot(delays,strainFit*1e3,'o-',color = 'black',label= "Fit (Dy)",linewidth = 2)
    ax2.plot(delays,strainCOM*1e3,'s-',color = 'gray',label= "COM (Dy)",linewidth = 2)
    ax2.plot(delays,strainFit*1e3,'o-',color = 'black',label= "Fit (Dy)",linewidth = 2)
    ax1.set_ylim(ylim[0],ylim[1])
    ax2.set_ylim(ylim[0],ylim[1])
    ax2.legend(loc = 0,fontsize = 13)
    ax1.set_ylabel('strain ($10^{-3}$)',fontsize=16)
    ax1.yaxis.set_label_coords(-0.07, 0.5)
    
    ax3.plot(delays,widthRelative*1e2,'-o',color = 'gray',label ="width COM",linewidth = 2)
    ax3.plot(delays,widthFitRelative*1e2,'-o',color = 'black',label ="width Fit",linewidth = 2)
    ax4.plot(delays,widthRelative*1e2,'-o',color = 'gray',label ="width COM",linewidth = 2)
    ax4.plot(delays,widthFitRelative*1e2,'-o',color = 'black',label ="width Fit",linewidth = 2)
    ax3.set_ylim(ylim[2],ylim[3])
    ax4.set_ylim(ylim[2],ylim[3])
    ax4.legend(loc = 0,fontsize = 13)
    ax3.set_ylabel('width change ($10^{-2}$)',fontsize=16)
    ax3.yaxis.set_label_coords(-0.07, 0.5)
    
    ax5.plot(delays,integralRelative*1e2,'-s',color = 'gray',label ="Area COM",linewidth = 2) 
    ax5.plot(delays,areaFitrelative*1e2,'-s',color = 'black',label ="Area Fit",linewidth = 2)    
    ax6.plot(delays,integralRelative*1e2,'-s',color = 'gray',label ="Area COM",linewidth = 2)
    ax6.plot(delays,areaFitrelative*1e2,'-s',color = 'black',label ="Area Fit",linewidth = 2)
    ax5.set_ylim(ylim[4],ylim[5])
    ax6.set_ylim(ylim[4],ylim[5])
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
        ax.set_xticks(xticks1)
        ax.spines['right'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.axvline(x= t_split ,ls = '--',color = "gray",lw = 1)
        ax.axvline(x= 0 ,ls = '--',color = "gray",lw = 1)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim([tlim[0],tlim[1]])
    
    for ax in [ax2,ax4,ax6]:
        ax.yaxis.tick_right()
        ax.axvline(x= tlim[1] ,ls = '--',color = "gray",lw = 1)
        ax.set_yticklabels([])
        ax.set_xticks(xticks2)
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim([tlim[1],tlim[2]])
        
    ax5.set_xlabel(r"Pump-probe delay t (ps)",fontsize = 16)
    ax5.xaxis.set_label_coords(0.75, -0.1)
    
    plt.text(0.25, 1.16, GetLabel(sampleName,number)[0] + str(ReadParameter(sampleName,number)[3]) + 'K     ' + str(ReadParameter(sampleName,number)[3]) + 'mJ/cm2' , fontsize = 16, ha='left', va='center', transform=ax1.transAxes)     
    plt.savefig('RawData/' + GetLabel(sampleName,number)[1] +'/OverviewResults.png',dpi =400,bbox_inches = 'tight')
    plt.show()

def plotPeakFit(sampleName,number,index,qzROI,resultFit,resultFitRef,t,PlotSemilogy,SaveFitPlot): 
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
    if PlotSemilogy:
        ax1.semilogy(qzROI,resultFit.data,'s',markersize=4,color='black',label = "data")
        ax1.semilogy(qzROI,resultFitRef.best_fit,'-',color = "grey",lw = 2,label = "Fit t<0")
        ax1.semilogy(qzROI,resultFit.best_fit,'r-',lw = 2,label = "Best Fit")
    else:    
        ax1.plot(qzROI,resultFit.data,'s',markersize=4,color='black',label = "data")
        ax1.plot(qzROI,resultFitRef.best_fit,'-',color = "grey",lw = 2,label = "Fit t<0")
        ax1.plot(qzROI,resultFit.best_fit,'-',lw = 2,color=rvb(0.99),label = "Best Fit")
        
    ax1.set_xlabel(r"$q_{\mathrm{z}}$" + r" ($\AA^{-1}$)",fontsize = 11)        
    ax1.set_ylabel("X-Ray Intensity",fontsize = 11) 
    
    ax1.yaxis.set_ticks_position('both')
    ax1.xaxis.set_ticks_position('both')
    ax1.tick_params('both', length=5, width=1, which='major',direction = 'in',labelsize=11)
    ax1.set_xlim(np.min(qzROI),np.max(qzROI)) 
    ax1.legend(loc = 0)
    
    plt.title(str(np.round(t,2)) +'ps ' + GetLabel(sampleName,number)[0], fontsize = 12)
    if SaveFitPlot:
        h.makeFolder('RawData/' + GetLabel(sampleName,number)[1] +'/Fitgraphics')
        plt.savefig('RawData/' + GetLabel(sampleName,number)[1] +'/Fitgraphics' +'/'+str(int(index))+'.png',dpi =200,bbox_inches = 'tight')
    plt.show()
    
def GetDetectorWidth(qxWidth,qzWidth,angle):
    qDWidth = np.sqrt((qzWidth**2*qxWidth**2)/(qxWidth**2+qzWidth**2*np.tan(h.radians(angle))**2))
    return qDWidth

def GetqzWidthfromDetector(qxWidth,qDWidth,angle):
    qzWidth = np.sqrt((qDWidth**2*qxWidth**2)/(qxWidth**2-qDWidth**2*np.tan(h.radians(angle))**2))
    return qzWidth


def GetPeakProjection(qxMin,qxMax,qzMin,qzMax,dataName):
    """ This function reads in a measured RSM and extracts the intensity distribution of an integral and slice trough
    the center of the selected Bragg peak (choosen q_x and q_z range) along q_x and q_z. It is usefull to choose the 
    q_x and q_z range equal for RSS factor analysis.
        
    Parameters
    ----------
    qxMin : float 
        Lower limit of the ROI along q_x
    qxMax : float 
        Upper limit of the ROI along q_x
    qxMin : float 
        Lower limit of the ROI along q_z
    qxMax : float 
        Upper limit of the ROI along q_z
    dataName : string 
        Name of the file containing the exported RSM (qx = column and qz = line)
    averageLines : integer 
        Number of lines that are averaged for the extraction of the Slice
            
    Returns
    ------
    qxAxis : 1D array
        Choosen q_x range of the Bragg peak
    qzAxis : 1D array
        Choosen q_z range of the Bragg peak
    rsmPeak : 2D array
        Intensity distribution along the choosen q_x and q_z range
    qxIntegral : 1D array
        Integrated intensity along q_z as function of q_x 
    qzIntegral : 1D array
        Integrated intensity along q_x as function of q_z 
    qxCenter : float
        Center of the peak along q_x
    qzCenter : float
        Center of the peak along q_z
    qxSlice : 1D array
        Intensity distribution along q_x through the center of the peak  
    qzSlice : 1D array
        Intensity distribution along q_z through the center of the peak  
            
    Example
    ------ 
        >>> qxAxis,qzAxis,rsmPeak,qxIntegral,qzIntegral,qxCenter,qzCenter,qxSlice,qzSlice = GetPeakProjection(-0.4,0.4,2,2.8,'RSMQ_0',1) """
    
    # Read In the exported RSMQ
    DataRSMQ  = np.genfromtxt(dataName + '.dat')
    RSMQ      = np.transpose(DataRSMQ[1:,1:])
    qxGrid    = DataRSMQ[1:,0]
    qzGrid    = DataRSMQ[0,1:]
    
    # Get the selected Peak and its projections
    qxAxis,qzAxis,rsmPeak,qxIntegral,qzIntegral = h.setROI2D(qxGrid,qzGrid,RSMQ,qxMin,qxMax,qzMin,qzMax)
   
    return qxAxis,qzAxis,rsmPeak,qxIntegral,qzIntegral

def GaussianFunction(axis,intensity,center,sigma,amplitude):
    model = lm.models.GaussianModel(prefix = "g_") 
    pars = lm.Parameters()  
    pars.add_many(('g_center',     center,    False),
                  ('g_sigma',      sigma,     False),
                  ('g_amplitude',  amplitude, False))
    resultFit = model.fit(intensity, pars, x = axis)
    return resultFit

def AdvancedGaussianFit(axis,intensity,excludeROI,leftBorder,rightBorder):
    """ This funtion fits the Intensity as function of an Axis with a Gaussian and
    a linear background. Additionally there is the possibility to exclude a certain region from the fit.
        
    Parameters
    ----------
    axis : 1D array 
        Axis corresponding to the intensity distribution
    intensity : 1D array 
        Normed Intensity distribution over the axis 
    excludeROI : boolean 
        Should a region of the axis be excluded
    leftBorder : float 
        Left border of the excluded region from the fit
    rightBorder : float 
        Right border of the excluded region from the fit
        
    Returns
    ------
    resultFit : lmfit object
        The fit results with different accessible information:
        Data: resultFit.data
        Final fit: resultFit.best_fit
        Parameters: resultFit.values['g_sigma'] 
            
    Example
    ------ 
        >>> resultFit = AdvancedGaussianFit([1,2,3,4,5,6,7],[1,3,6,9,9,9,1],True,5,6) """
    
    COM ,STD ,Integral = h.calcMoments(axis,intensity) 
    model = lm.models.GaussianModel(prefix = "g_") + lm.models.LinearModel()       
    pars = lm.Parameters()  
    pars.add_many(('g_center',     COM, True),
                  ('g_sigma',      STD,	True),
                  ('g_amplitude',  1,   True),
                  ('slope',        0,   True),
                  ('intercept',    0,   True))
    if excludeROI:
        selectROI = (axis < leftBorder) | (axis > rightBorder)
        dummy     = model.fit(intensity[selectROI], pars, x = axis[selectROI])
        resultFit = GaussianFunction(axis,intensity,dummy.values['g_center'],dummy.values['g_sigma'],dummy.values['g_amplitude']) 
    else:
        resultFit = model.fit(intensity, pars, x = axis)
    return resultFit
        
def PlotPeakWidthOverview(qxAxis,qxCenter,qzAxis,qzCenter,rsmQ,qxFitIntegral,qzFitIntegral,folder,RSMName,peakName):
    """ This function plots an overview of the RSS factor calculation and the width of the 
    Bragg peak along q_x and q_z. To get a feeling of the width ratio from the plotted RSM
    it is advantageous if qxAxis and qzAxis have the same range.
        
    Parameters
    ----------
    qxAxis : 1D array 
        Selected Region along q_x
    qxCenter : float 
        Center of the Bragg peak along q_x where the Slice is exctracted
    qzAxis : 1D array 
        Selected Region along q_z
    qzCenter : float 
        Center of the Bragg peak along q_z where the Slice is exctracted
    rsmQ : 2D array
        Intensity distribution in reciprocal space
    qxFitSlice : lmfit object
        Fit of normed intensity of a slice along q_z
    qxFitIntegral : lmfit object
        Fit of normed intensity of an integral along q_x
    qzFitSlice : lmfit object
        Fit of normed intensity of a slice along q_x
    qzFitIntegral : lmfit object
        Fit of normed intensity of an integral along q_z
    sampleName : string
        Name of the sample for the plot title
    peakName : string
        Name of the selected peak for the plot title
              
    Example
    ------ 
        >>> PlotPeakWidthOverview(qxAxis,qxCenter,qzAxis,qzCenter,rsmQ,qxFitSlice,qxFitIntegral,qzFitSlice,qzFitIntegral,'NbDy','Dysposium',15.6) """
    
    
    # Get the peak width and RSS factors
    qxWidthIntegral = qxFitIntegral.values['g_sigma']
    qzWidthIntegral = qzFitIntegral.values['g_sigma']
    
    # Plot the overview of the analysis
    CMap = plt.get_cmap('RdBu_r')
    f    = plt.figure(figsize = (7.5,7.5))
    gs0  = f.add_gridspec(2, 2,width_ratios=[0.68,1],height_ratios=[0.68,1],wspace=0.00, hspace=0.0)
    ax1  = f.add_subplot(gs0[0,1])
    ax3  = f.add_subplot(gs0[1,1])
    ax2  = f.add_subplot(gs0[1,0])
   
    # Plot the qx Integral and Slice
    ax1.plot(qxAxis,qxFitIntegral.data,'o',ms = 5,mew = 0.5,fillstyle = 'none',color = CMap(0.01),label = r'$q_\mathrm{x}$ - integral')
    ax1.plot(qxAxis,qxFitIntegral.best_fit,'--',lw = 2,color = CMap(0.01),label = "FWHM= " + str(np.round(2.355*qxWidthIntegral,3)) +'$\,\mathrm{\AA^{-1}}$')
    ax1.set_xlim(np.min(qxAxis),np.max(qxAxis))
    ax1.set_xticks(np.arange(np.round(np.min(qxAxis),2)+0.01,np.max(qxAxis)+0.01,0.02))
    ax1.set_ylim([-0.05,1.05])
    ax1.set_yticks([0,0.5,1])
    ax1.legend(loc = 0,fontsize = 7)
   
    #Plot the qz Integral and Slice
    ax2.axhline(y = qzCenter,lw = 0.5,ls = '--',color = 'gray')
    ax2.plot(qzFitIntegral.data,qzAxis,'o',ms = 5,mew = 0.5,fillstyle = 'none',color = CMap(0.01),label = r'$q_\mathrm{z}$ - integral')
    ax2.plot(qzFitIntegral.best_fit,qzAxis,'--',lw = 2,color = CMap(0.01),label = "FWHM= " + str(np.round(2.355*qzWidthIntegral,3)) +'$\,\mathrm{\AA^{-1}}$')
    ax2.set_xlabel(r'$I/I_\mathrm{max}$',fontsize = 12,labelpad = 7)
    ax2.set_xlim([1.05,-0.05 ])
    ax2.set_xticks([0,0.5,1])
    ax2.set_ylabel(r'$q_\mathrm{z}$ $\mathrm{\left(\AA^{-1}\right)}$' ,fontsize = 12,labelpad = 1)
    ax2.set_ylim(np.min(qzAxis),np.max(qzAxis))
    ax2.set_yticks(np.arange(np.round(np.min(qzAxis),2)+0.01,np.max(qzAxis)+0.01,0.02))
    ax2.legend(loc = 2,fontsize = 7)
    
    #Plot the RSMQ
    X,Y     = np.meshgrid(qzAxis,qxAxis);
    plotted = ax3.pcolormesh(Y,X,np.transpose(rsmQ)/np.max(rsmQ),shading = 'auto',vmin = 0, vmax = 1,  cmap = CMap)
    ax3.axis([np.min(qxAxis), np.max(qxAxis), np.min(qzAxis), np.max(qzAxis)])
    ax3.axvline(x = qxCenter,lw = 0.5,ls = '--',color = 'gray')
    ax3.axhline(y = qzCenter,lw = 0.5,ls = '--',color = 'gray')
    ax3.set_xlabel(r'$\mathrm{q_x}$ $\mathrm{\left(\AA^{-1}\right)}$',labelpad = 0, fontsize = 12)  
    ax3.tick_params(axis='x', which='major', pad=9)
    ax3.xaxis.set_ticks_position('both')
    ax3.set_xlim([qxAxis.min(), qxAxis.max()])
    ax3.set_xticks(np.arange(np.round(np.min(qxAxis),2)+0.01,np.max(qxAxis)+0.01,0.02)) 
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    ax3.set_ylim([qzAxis.min(), qzAxis.max()])
    ax3.set_yticks(np.arange(np.round(np.min(qzAxis),2)+0.01,np.max(qzAxis)+0.01,0.02))

    cbaxes = f.add_axes([0.7, 0.135, 0.1875, 0.015])  
    cbar   = plt.colorbar(plotted,cax = cbaxes,orientation = 'horizontal') 
    cbar.ax.tick_params('both', length=5, width=0.5, which='major',direction='in')  
    cbar.ax.tick_params(labelsize=9)
    cbar.set_label(r'$I/I_\mathrm{max}$', rotation=0,fontsize = 12)
    cbaxes.xaxis.set_ticks_position("top")
    cbaxes.xaxis.set_label_position("top")
    cbar.set_ticks([0,0.5,1])
    
    #Axis Manipulation and title
    for ax in [ax1,ax2,ax3]:
        ax.tick_params('both', length=3, width=0.5, which='major',direction = 'in')
        ax.tick_params('both', length=1.5, width=0.5, which='minor',direction = 'in')
        ax.xaxis.set_ticks_position('both')
        ax.yaxis.set_ticks_position('both')
        ax.tick_params(axis='both', which='major', labelsize=9 )
        ax.tick_params(axis='x',which='both',bottom='on',top='on',labelbottom='on')
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(0.5)   

    ax1.text(-0.75, 0.90,RSMName, transform=ax1.transAxes,horizontalalignment='left',verticalalignment='top', fontsize=14,color = "black") 
    ax1.text(-0.75, 0.75, peakName +' peak', transform=ax1.transAxes,horizontalalignment='left',verticalalignment='top', fontsize=12,color = "black") 
    ax1.text(-0.75, 0.65, r'$q_z =$' + str(np.round(qzCenter,3)) + r'$\,\AA^{-1}$', transform=ax1.transAxes,horizontalalignment='left',verticalalignment='top', fontsize=10,color = "black") 

    plt.savefig(folder + RSMName+'_'+peakName +'.png',dpi = 600,bbox_inches='tight')
    plt.show()
    return