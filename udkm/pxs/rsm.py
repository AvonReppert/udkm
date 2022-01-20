
import numpy as np
import udkm.tools.helpers as h

teststring = "Successfully loaded udkm.pxs.rsm"


## detector related
pixelsize   = 0.172*1e-3 # Detector pixelsize of the Pilatus in m
centerpixel = -1       #initializes the centerpixel with a negative value. Once it is set it stays the same
detectorDistance = -2  #initializes the sample detector distance  with a negative value. Once it is set it stays the same

## Normalization detector related
crystalOffset = 0.0197   #Crystal Offset of the Crystal Normalization detector in [V]
crystalThreshold = 0.003 #Measurements where the Crystal Voltage is below this Value are discarded
        
## Source related
wl      = 1.5418           #Cu K-alpha Wavelength in Angstroem
k       = 2*np.pi/wl  #absolute value of the incident k-vector in [1/Ang]
convergenceXRays = h.radians(0.3) #Convergence of the Optics used in our Plasma X-Ray source setup in [rad]      
        
## Calculation Related
existsRefList   = False
existsPathList  = False
qzBGcalculated  = False

def loadParameters(sampleName):
    """ Load the Parameter list into the object. Oversamplingz is the factor 
    that tells how many times the number of omega datapoints are used to sample the qz-Axis. Usually oversamplingx
    should be set to the same value as oversamplingz. 1/oversamplingx is the factor by which the qx axis stepnumber
    is multiplied"""
    sampleName = sampleName        
    parameterListFileName = 'ReferenceData/ParameterStatic' + sampleName + '.txt'          # File Name of the Parameter File
    paramList = np.genfromtxt(parameterListFileName, delimiter='\t',comments = "%")    # Loading the Parameter File
    length = len(paramList)       # Number of Measurements   
    IDs  = paramList[:,0]              
    date = paramList[:,1]         # Date of the measurement
    time = paramList[:,2]         # Time of the measurement
    TA = 300*np.ones(len(paramList))          # Temperature read from sensor A near the expander             
    TB = 300*np.ones(len(paramList))           # Temperature read from sensor B near the sample
    F  = np.zeros(length)          # Excitation Fluence here initializied as 0
    delay = np.nan*np.ones(length)  # Delay of the Rocking curve here initialized as nan
    identifier = h.timeStamp(date,time) # Unique Identifier of each measurement as seconds since 1970
 