# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 11:55:33 2022

@author: matte
"""

def GetBackgroundShoulderPeak(rockingCurves,qzPeakMin,qzPeakMax,t0,weightsMin,weightsShoulder,weightsMax):
    """ This function extract the transducer peak from the shoulder of the substrate for better evaluation. 
    The peak is between 'qzPeakMin' and 'qzPeakMax'. The substarte peak is given by a Lorentzian profile, whose
    fit is different weigted below and slightly above the transducer Bragg peak.
    
    Parameters:
    -----------
    rockingCurves: 2D array
        contains the rocking curve for qz(x) and each delay (y1) with peak and background
    qzPeakMin: float
        the estimated left border of the transducer peak
    qzPeakMax: float
        the estimated right border of the transducer peak
    t0: float
        temporal overlapp between pump and probe
    weightsMin: float
        weight of small amplitude left of the peak for fitting (high)
    weightsShoulder: float
        weight of connection between transducer and substrate for fitting (mean)
    weightsMax: float
        weight of point up to maximum of substrate for fitting (small)
    
    Returns:
    --------
    background: 1D array
        contains the values of the best Lorentzian fit for all qz of qzAxis
    fitBackground: 1D array
        contains the values of the best Lorentzian fit for the fitted points
    qzBackgrond: 1D array
        contains the qz values of the fitted points
    qzShoulder: float
        describe the qz value of the minimum between the peak maxima (start of fitting)
    """
    
    qzAxis             = rockingCurves[0,1:]
    timeAxis           = rockingCurves[1:,0]
    intensities        = rockingCurves[1:,1:]
    meanRockingCurvet0 = np.mean(intensities[timeAxis < t0,:],0)
    # Determine the transition of both peaks
    selectPeak       = np.logical_and(qzAxis > qzPeakMin,qzAxis < qzPeakMax)
    qzMaxPeak        = qzAxis[selectPeak][np.argmax(meanRockingCurvet0[selectPeak]-meanRockingCurvet0[selectPeak].max())]
    qzSubstrate      = qzAxis[np.argmax(meanRockingCurvet0-meanRockingCurvet0.max())]
    selectTransition = np.logical_and(qzAxis > qzMaxPeak,qzAxis <= qzSubstrate)
    # Transition coprresponds to minimum between maxima
    qzShoulder = qzAxis[selectTransition][np.argmin(meanRockingCurvet0[selectTransition]-meanRockingCurvet0[selectTransition].min())]
    # Define parts to fit the Lorentzian profile
    selectBackgroundMin = qzAxis < qzPeakMin
    selectBackgroundMax = np.logical_and(qzAxis >= qzShoulder, qzAxis <= qzSubstrate)
    selectBackground    = np.logical_or(selectBackgroundMin, selectBackgroundMax)
    qzFit               = qzAxis[selectBackground]
    intensitiesFit      = meanRockingCurvet0[selectBackground]
    # Define the weighting factors of the different parts
    weightFactors                                                                           = np.ones(np.size(qzAxis[selectBackground]))
    weightFactors[0:np.sum(selectBackgroundMin)-1]                                          = weightsMin
    weightFactors[np.sum(selectBackgroundMin):np.sum(selectBackgroundMin)+20] = weightsShoulder
    #weightFactors[np.sum(selectBackgroundMin):np.sum(selectBackgroundMin)+int((len(qzAxis)/30))] = weightsShoulder
    #weightFactors[np.sum(selectBackgroundMin)+int((len(qzAxis)/30)):]                            = weightsMax
    weightFactors[np.sum(selectBackgroundMin)+20:]                            = weightsMax
    # Fit the Lorentzian Background of Substrate Bragg peak
    #qzCenterMin   = qzAxis[selectTransition][np.argmin(meanRockingCurvet0[selectTransition]-meanRockingCurvet0[selectTransition].min())+int((0*len(qzAxis)/30))]
    qzCenterMin   = qzAxis[selectTransition][np.argmin(meanRockingCurvet0[selectTransition]-meanRockingCurvet0[selectTransition].min())+5]
    backgroundFit = LorentzianFit(qzSubstrate,0.01,meanRockingCurvet0.max(),qzFit,intensitiesFit,[True,True,True],qzCenterMin,weightFactors)
    fitBackground = backgroundFit.best_fit
    # Get the Background over full qz range
    fitValues  = LorentzianFit(backgroundFit.values['L_center'],backgroundFit.values['L_sigma'],backgroundFit.values['L_amplitude'],qzAxis,meanRockingCurvet0,[False,False,False],backgroundFit.values['L_center'],np.zeros(len(meanRockingCurvet0)))
    background = fitValues.best_fit
    return background, fitBackground, qzFit, qzShoulder