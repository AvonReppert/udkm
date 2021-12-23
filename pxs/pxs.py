# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 21:30:38 2015

@author: aleks
"""
import udkm.tools.tools as tools
h = tools.tools()


import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import lmfit as lm

from tqdm import tqdm  #progress bar
import multiprocessing as mp 

import matplotlib.colors as mcolors
import matplotlib.path as mplPath
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
#plt.style.use("matplotlibStyle.mpl")

class TransformStatic:
    """ This class is designed to transform the static measurements:
        What it does:        
        1. Reads in the Parameter List, 
        2. Reads in a dataset that corresponds to a selected line in the Parameter List
        3. Transforms the Data to qz space
        4. Saves the tranformed data to a file in the directory        
    """
    def __init__(self,*args,**kwargs):      
        ## Pilatus detector related
        self.pixelsize = 0.172*1e-3 # Detector pixelsize of the Pilatus in m
        self.centerpixel = -1       #initializes the centerpixel with a negative value. Once it is set it stays the same
        self.detectorDistance = -2  #initializes the sample detector distance  with a negative value. Once it is set it stays the same
        self.number = 0
        ## Normalization detector related
        self.crystalOffset = 0.0197   #Crystal Offset of the Crystal Normalization detector in [V]
        self.crystalThreshold = 0.003 #Measurements where the Crystal Voltage is below this Value are discarded
        
        ## Source related
        self.wl = 1.5418           #Cu K-alpha Wavelength in [Ang]
        self.k = 2*np.pi/self.wl    #absolute value of the incident k-vector in [1/Ang]
        self.convergenceXRays = h.radians(0.3) #Convergence of the Optics used in our Plasma X-Ray source setup in [rad]      
        
        ## Plotting related
        self.dpiValue = 200         #dpi Value for the saved png Graphics
        
        ## Performance related 
        self.cores = mp.cpu_count()-1 #set maximumnumber of CPU-Cores used for grip mapping
        
        ## Calculation Related
        self.existsRefList = False
        self.existsPathList = False
        self.qzBGcalculated = False
        ## Properties of the Peaks that are later Fitted
        self.TbFe = {"name":"TerbiumEisen","shortName":"TbFe",   "qzMin":2.35, "qzMax":2.45, "qzCenter":2.4,  "qzArea":1.0, "qzWidth":0.0073, "order":2, "qxMin":-0.06, "qxMax":0.06, "qxCenter":0, "qxArea":1.0, "qxWidth":0.009}
        self.Al =  {"name":"Aluminium","shortName":"Al",    "qzMin":2.0, "qzMax":2.08, "qzCenter":2.04, "qzArea":0.1, "qzWidth":0.0099, "order":2, "qxMin":-0.05, "qxMax":0.05, "qxCenter":0, "qxArea":0.1, "qxWidth":0.010}
        self.Nb = {"name":"Niobium","shortName":"Nb",   "qzMin":2.63, "qzMax":2.73, "qzCenter":2.68, "qzArea":4.0, "qzWidth":0.0090, "order":2, "qxMin":-0.03, "qxMax":0.03, "qxCenter":0, "qxArea":5.0, "qxWidth":0.009}
        self.Sap ={"name":"Sapphire","shortName":"Sap", "qzMin":2.58, "qzMax":2.63, "qzCenter":2.60, "qzArea":3.0, "qzWidth":0.0073, "order":1, "qxMin":-0.01, "qxMax":0.01, "qxCenter":0, "qxArea":4.0, "qxWidth":0.005}
        self.Au ={"name":"Gold","shortName":"Au", "qzMin":2.6, "qzMax":2.8, "qzCenter":2.70, "qzArea":3.0, "qzWidth":0.1, "order":1, "qxMin":-0.01, "qxMax":0.01, "qxCenter":0, "qxArea":4.0, "qxWidth":0.005}        
        self.Si ={"name":"Silizium","shortName":"Si", "qzMin":2.0, "qzMax":2.1, "qzCenter":2.05, "qzArea":3.0, "qzWidth":0.008, "order":1, "qxMin":-0.01, "qxMax":0.01, "qxCenter":0, "qxArea":4.0, "qxWidth":0.005}
        self.STO ={"name":"Strontium titanate","shortName":"STO", "qzMin":3.1, "qzMax":3.3, "qzCenter":3.21, "qzArea":0.006, "qzWidth":0.003, "qzFraction":0.5, "order":2, "qxMin":-0.01, "qxMax":0.01, "qxCenter":0, "qxArea":4.0, "qxWidth":0.005, "qxFraction":0.5}  
        self.SL1 ={"name":"SRO/BTO SL1","shortName":"SL1", "qzMin":3.05, "qzMax":3.15, "qzCenter":3.1, "qzArea":0.006, "qzWidth":0.01, "qzFraction":0.5, "order":2, "qxMin":-0.01, "qxMax":0.01, "qxCenter":0, "qxArea":4.0, "qxWidth":0.005, "qxFraction":0.5} 
        self.SL2 ={"name":"SRO/BTO SL2","shortName":"SL2", "qzMin":2.85, "qzMax":2.97, "qzCenter":2.93, "qzArea":0.00006, "qzWidth":0.003, "qzFraction":0.5, "order":2, "qxMin":-0.01, "qxMax":0.01, "qxCenter":0, "qxArea":4.0, "qxWidth":0.005, "qxFraction":0.5}
        self.MBE2057 = {'name':'MBE20157','shortName':'MBE2057','qzMin':1.9, 'qzMax':2.8,'qzCenterNb':2.69, 'qzCenterSap':2.625, 'qzCenterHo':2.22, 'qzCenterY':2.18, 'qzCenterP1':2.075, 'qzCenterP2':1.97,'qzAreaNb':0.004,'qzAreaSap':0.003,'qzAreaHo':0.001,'qzAreaY':0.001,'qzAreaP1':0.00002,'qzAreaP2':0.00001,'qzWidthNb': 0.009,'qzWidthSap': 0.0073,'qzWidthHo': 0.0073,'qzWidthY': 0.0099,'qzWidthP1': 0.0073,'qzWidthP2': 0.0073,'qzFractionNb':0.5,'qzFractionSap':0.5,'qzFractionHo':0.5,'qzFractionY':0.5,'qzFractionP1':0.5,'qzFractionP2':0.5,"qxMin":-0.01, "qxMax":0.01, "qxCenter":0, "qxArea":4.0, "qxWidth":0.005, 'qxFraction':0.5}        
        self.peaks = {"TbFe":self.TbFe,"Al":self.Al,"Nb":self.Nb,"Sap":self.Sap,"Au":self.Au,"Si":self.Si,"STO":self.STO,"SL1":self.SL1,"SL2":self.SL2,'MBE2057':self.MBE2057}
        
    def loadParameters(self,sampleName):
        """ Load the Parameter list into the object. Oversamplingz is the factor 
        that tells how many times the number of omega datapoints are used to sample the qz-Axis. Usually oversamplingx
        should be set to the same value as oversamplingz. 1/oversamplingx is the factor by which the qx axis stepnumber
        is multiplied"""
        self.sampleName = sampleName        
        self.parameterListFileName = 'ReferenceData/ParameterStatic' + sampleName + '.txt'          # File Name of the Parameter File
        self.paramList = np.genfromtxt(self.parameterListFileName, delimiter='\t',comments = "%")    # Loading the Parameter File
        self.length = len(self.paramList)       # Number of Measurements                
        self.TA = 300*np.ones(len(self.paramList))          # Temperature read from sensor A near the expander             
        self.TB = 300*np.ones(len(self.paramList))           # Temperature read from sensor B near the sample
        self.date = self.paramList[:,1]         # Date of the measurement
        self.time =self.paramList[:,2]         # Time of the measurement
        self.identifier =h.timeStamp(self.date,self.time) # Unique Identifier of each measurement as seconds since 1970
        self.F = np.zeros(self.length)          # Excitation Fluence here initializied as 0
        self.delay = np.nan*np.ones(self.length)  # Delay of the Rocking curve here initialized as nan
        self.IDs = self.paramList[:,0] 
        self.sampleType = self.paramList[:,3] 
        
    def calcReferenceList(self,line,detectorDistance,centerpixel,crystalThreshold):
        self.currentLine = line # Reads in the current line of the parameterfile that is to be transformed in the future              
        self.loadFileName() #Unzips the Archive containing the data and prepares the FileNames            

        self.detectorDistance = detectorDistance                                #set the detector distance between sample and detector in [m]
        self.deltaTheta = np.arctan(self.pixelsize/self.detectorDistance)       #deltaTheta in radians
        self.crystalThreshold = crystalThreshold             
        print('Calculating Reference List... from '+ str(int(self.date[self.currentLine])) + ' ' +h.timestring(self.time[self.currentLine]))
        
        ########################## Calculating a Reference DataArray ######################################
        self.omegaColumn = 1
        self.dataRocking=  np.genfromtxt(self.FileName, delimiter="\t", comments = "%") # Reads in the current Rocking curve File
        self.pixelNumber = np.size(self.dataRocking[0,:])-5          # Automatically selects the pixelnumber of the selected ROI
        self.pixels = range(self.pixelNumber)                
                   
        self.omegaAll = self.dataRocking[:,self.omegaColumn]                        # Omega: Angle of incidence between the sample surface and the incoming X-Rays 
        self.omegaMeasured = np.unique(self.omegaAll)                               # List of unique omegas
        self.RSMangleRef  = np.ones((self.pixelNumber,np.size(self.omegaMeasured))) #Initializing the RSM filled with ones
        
        
        self.loadRSMRaw(line)
        self.centerpixel = h.calcCenterPixel(self.RSMangle,centerpixel)
        
        ###########################Calculating a Reference DataList ########################################
        n = 0                                                                   #counter variable for the row of the List
        self.RefList = np.zeros((np.size(self.RSMangleRef),13))                       #initialize the list        
        _,omegaL,omegaR = h.calcGridBoxes(self.omegaMeasured)         #calculate the delta omega values of the possibly nonlinear omega grid                
         
        for i in tqdm(range(len(self.omegaMeasured))):    # iterate with i over the measured angles
               for p in self.pixels:                # iteratre with p over the pixels of the detector
                self.RefList[n,0] = n                   # id: unique identifier of this list entry
                self.RefList[n,1] = i               #omegaIndex: Angle of incidence in this measurement
                self.RefList[n,2] = p               #pixelIndex: Pixel on the detector at which it was detected
                self.RefList[n,3] = h.radians(self.omegaMeasured[i])    # omega :Current angle omega of the current RSMangle point in radians
                self.RefList[n,4] = h.radians(omegaL[i])           # omegaL: Left boundary in omega
                self.RefList[n,5] = h.radians(omegaR[i])           # omegaR: Right boundary in omega                
                self.RefList[n,6] = self.RefList[n,3] + self.deltaTheta*(self.pixels[p] - self.centerpixel) # theta: Current angle theta calculated from the pixel                
                self.RefList[n,7], self.RefList[n,8] = h.calcQzQx(self.RefList[n,3],self.RefList[n,6],self.k)  # Qz and Qx  calculated from omega and theta           
                self.RefList[n,9] = h.calcJacobiDet(self.RefList[n,3],self.RefList[n,6],self.k)                # J: Jacobian at this omega and theta which provides the volume distorting fraction of the transformation [(1/Ang^2)*(1/rad^2)]
                self.RefList[n,10] = self.RSMangleRef[p,i]                                                   # P: measured relative Intensity P at that omega and pixel in [photons/CrystalV]
                self.RefList[n,11] = self.RefList[n,10]/(self.deltaTheta*self.convergenceXRays)               # D(w,t): Intensity density in angle Space D(omega,theta) = P/(deltaTheta*sourceConvergence)  [photons/(rad^2 * CrystalV)]              
                self.RefList[n,12] = self.RefList[n,11]/self.RefList[n,9]                                       # D(qx,qz): Intensity density in qspace D(qx,qz)= D(omega,theta)/J(omega,theta) [photons/(CrystalV)*Ang^2] 
                n+=1  
        self.existsRefList = True #Set this indicator to true in order to signify that a Reference List has been calculated
       
        self.sortRSMangleToList()
        self.calcPathList()
        
    def loadRSMRaw(self,line):
        """ Reads in the raw RSM in anglespace from the rockingcurves file that belongs to inserted line in the parameter File. 
        If the Crytal Voltage Value is below the given threshold then the dataline is automatically discarded """
        self.currentLine = line # Reads in the current line of the parameterfile that is to be transformed in the future              
        self.loadFileName() #Unzips the Archive containing the data and prepares the FileNames
        print('Loading Data... ' + self.sampleName +' '+ str(int(self.date[self.currentLine])) + ' ' +h.timestring(self.time[self.currentLine]))
        
        self.dataRockingRaw=  np.genfromtxt(self.FileName, delimiter="\t", comments = "%") # Reads in the current Rocking curve File
         # Remove all Rows where the Crystal Threshold is below the inserted threshold    
        self.dataRocking = self.dataRockingRaw[self.dataRockingRaw[:,3] - self.crystalOffset > self.crystalThreshold,:]  
       
        # Normalize of the data to the Crystal Voltage. The Crystal Voltage is located in the third Colum:
        for i in range(np.size(self.dataRocking[:,0])):
            self.dataRocking[i,5:self.pixelNumber+5] = self.dataRocking[i,5:self.pixelNumber+5]/(self.dataRocking[i,3] - self.crystalOffset)      
        self.loop = self.dataRocking[:,0]                            # Angle Loop Number
        self.omegaLoopsMeasured = int(np.max(self.loop))             # Angle Loops measuremd
        self.omegaAll = self.dataRocking[:,1]                        # Omega: Angle of incidence between the sample surface and the incoming X-Rays 
        self.omegaMeasured = np.unique(self.omegaAll)                # List of unique omegas
        #self.thetaAll = self.dataRocking[:,2]/2                      # Not needed for the analysis Theta: diffraction angle that the sample surface makes with the detector 
        #self.thetaMeasured = np.unique(self.thetaAll)                # Not needed for the analysis List of unique thetas                
        #self.crystalVAll = self.dataRocking[:,3] - self.crystalOffset   # Voltage detected by normalization detector this is used as normalization
        #self.photonsAll = self.dataRocking[:,4]                         # Photons detected by normalization detector : A nonlinear conversion is involved here and it is currently not used      

        self.RSMangle  = np.zeros((self.pixelNumber,np.size(self.omegaMeasured))) #Initializing the RSM
        
        # Here I calculate the RSM in angle space from the average of the RSMs                
        for i in range(np.size(self.omegaMeasured)):                       #iterate with i over the measured angles    
            select    = self.omegaAll == self.omegaMeasured[i];            #selection of the measurements with the current angle of incidence
            self.RSMangle[:,i] = np.average(self.dataRocking[select,5:self.pixelNumber+5],weights = self.dataRocking[select,3],axis = 0) # Calculate a weighted Average of the data
        
        if self.existsRefList:
            self.sortRSMangleToList()
        
    def sortRSMangleToList(self):
    ### Sorting the RSMraw into the List ###
        self.List = self.RefList
        for n in self.RefList[:,0].astype(int):    # iterate with i over the measured angles
                self.List[n,10] = self.RSMangle[int(self.List[n,2]),int(self.List[n,1])]                                                   # P: measured relative Intensity P at that omega and pixel in [photons/CrystalV]
                self.List[n,11] = self.List[n,10]/(self.deltaTheta*self.convergenceXRays)               # D(w,t): Intensity density in angle Space D(omega,theta) = P/(deltaTheta*sourceConvergence)  [photons/(rad^2 * CrystalV)]              
                self.List[n,12] = self.List[n,11]/self.List[n,9]  

                
    def calcPath(self,omegaL,omegaR,theta,dTheta):        
        return 
        
    def calcPathList(self):
        print('Calculating Path List...')
        #Calculates the paths around each element from the Reference List
        self.pathList = []        
        for n in tqdm(self.RefList[:,0].astype(int)):
            self.pathList.append(mplPath.Path(np.array([h.calcQzQx(self.RefList[n,4],self.RefList[n,6] - self.deltaTheta,self.k), h.calcQzQx(self.RefList[n,5],self.RefList[n,6] - self.deltaTheta,self.k), h.calcQzQx(self.RefList[n,5],self.RefList[n,6] + self.deltaTheta,self.k),h.calcQzQx(self.RefList[n,4],self.RefList[n,6] + self.deltaTheta,self.k)]))) 
    
    def createQGrid(self,qzMin,qzMax,qxMin,qxMax,qzOversampling,qxOversampling):
        """ Creates a possibly nonlinear QzGrid by transforming the given Omega Axis at qx = 0 and 
            a possibly nonlinear QxGrid by transforming the theta Axis for a fixed omega value at omega = np.mean(omegaMeasured)
            the number of Gridpoints on the qxAxis is equal to number of detectorpixels/qxDownsampling because only every qxDownsampling point of the qx Axis is taken as gridpoint
            the number of Gridpoints on the qzAxis is eqal  to number of omegasMeasured*(2^qzOversampling) because a point is added in between each two existing qzPoints"""
        print('Creating the Grid in Qspace ...')
        self.qzOversampling = qzOversampling
        self.qxOversampling = qxOversampling
        self.qzGrid=np.linspace(qzMin,qzMax,int(qzOversampling*np.size(self.omegaMeasured))) 
        self.qxGrid=np.linspace(qxMin,qxMax,int(qxOversampling*self.pixelNumber)) 
       
        ## Calculating the bins for the later to use histogram        
        qzDelta,self.qzL,self.qzR = h.calcGridBoxes(self.qzGrid) 
        qxDelta,self.qxL,self.qxR = h.calcGridBoxes(self.qxGrid)
        self.qzDelta = self.qzGrid[1]-self.qzGrid[0]
        self.qxDelta = self.qxGrid[1]-self.qxGrid[0]
        self.qzBins = np.append(self.qzL,self.qzR[-1])       
        self.qxBins = np.append(self.qxL,self.qxR[-1])
        
        self.qxRange =qxMax-qxMin 
        self.RSMQempty = np.zeros((np.size(self.qxGrid),np.size(self.qzGrid)))
        
        self.qListEntries = np.size(self.qzGrid)*np.size(self.qxGrid)
        self.QRefList = np.zeros((self.qListEntries,18))
        n = 0 
        for lx in tqdm(range(np.size(self.qxGrid))):
            for lz in range(np.size(self.qzGrid)):
                self.QRefList[n,0] = n #ID
                self.QRefList[n,1] = lz #index in qz
                self.QRefList[n,2] = lx #index in qx
                self.QRefList[n,3] = self.qzGrid[lz] #qz [1/Ang]
                self.QRefList[n,4] = self.qxGrid[lx] #qz [1/Ang]                              
                n += 1
            
    def createMappingMultiCore(self):
        "This function calculates the mapping between the RSMangle and the RSMQ entries with the arbitrarily chosen grid"
        print('Creating the mapping to the defined Grid in Qspace with ' +str(self.cores)+  ' cores...')
        self.points = np.array([self.QRefList[:,3],self.QRefList[:,4]])
        self.points = self.points.transpose()
        processes = []
        listsize = np.size(self.pathList)
        partsize = int(round(listsize/self.cores,0))
        self.q = mp.Queue()
        for c in range(self.cores):
            if c == self.cores-1:
                p = mp.Process(target=self.createMappingMC,args=(self.q,c*partsize,listsize,False))
                processes.append(p)
            else:
                p = mp.Process(target=self.createMappingMC,args=(self.q,c*partsize,(c+1)*partsize,False))
                processes.append(p)
        [x.start() for x in processes]
        for c in tqdm(range(self.cores)):
            self.QRefList[:,5:18] = self.QRefList[:,5:18] + self.q.get()          
        

    def saveMapping2txt(self):
        "This function saves the mapping as txt"
        header = "0 n Identifier\t 1 lz QzGridIndex\t 2 lx QxGridIndex\t 3 qz Box [1/Ang]\t 4 qx Box [1/Ang]\t 5 nList \t 6 i omegaIndex\t 7 p pixelIndex\t 8 omega\t 9 omegaL\t 10 omegaR\t 11 theta\t 12 qzRSM\t 13 qxRSM\t 14 J Jacobian\t 15 P Pilatus Value\t 16 D(w,t) = P/(dwdt) \t 17 P/(dwdtJ)"        
        h.writeListToFile(self.QRefListFileName+str(int(self.qzOversampling))+str(int(self.qxOversampling))+".dat",header,self.QRefList)
    
    def saveMapping2np(self):
        "This function saves the mapping as numpy array"
        np.savez_compressed(self.QRefListFileName+str(int(self.qzOversampling))+str(int(self.qxOversampling)),self.QRefList)
    
    def createMapping(self):
        "This function calculates the mapping between the RSMangle and the RSMQ entries with the arbitrarily chosen grid"
        print('Creating the mapping to the defined Grid in Qspace with 1 core...')
        self.points = np.array([self.QRefList[:,3],self.QRefList[:,4]])
        self.points = self.points.transpose()
        self.createMappingMC(self.q,0,np.size(self.pathList),True)
 
    def createMappingMC(self,q, start, stop,Print):
        QRefList = np.zeros((self.qListEntries,13))
        if Print:
            for i in tqdm(range(start,stop)):
                select = self.pathList[i].contains_points(self.points)
                QRefList[select,0:13] = self.RefList[i,0:13]
        else:
            for i in range(start,stop):
                select = self.pathList[i].contains_points(self.points)
                QRefList[select,0:13] = self.RefList[i,0:13]    
        self.q.put(QRefList)                 
        
    def loadMappingfromtxt(self,filename):
        "This loads the Mapping from a given datafile. Saves a lot of time but requires that the centerpixel, the scanned angles and the chosen Grid in Q-space stay the same "
        print('load Mapping from datafile')        
        self.QRefList = np.genfromtxt(filename,comments = "#",delimiter = "\t")
        
    def loadMappingfromnp(self,filename):
        "This loads the Mapping from a given npfile. Saves a lot of time but requires that the centerpixel, the scanned angles and the chosen Grid in Q-space stay the same "
        print('load Mapping from npfile')
        self.QRefList = np.load(filename)['arr_0']
    
    def sortRSMangleToQList(self):
        "Sorts the data from the angle List into the according entries in Q-space accordingt to the previously calculated mapping"
        self.QList = self.QRefList
        for n in self.QList[:,0].astype(int):    
            if self.QList[n,5] > 0 and  self.QList[n,7] < np.shape(self.RSMangleRef)[0] and self.QList[n,6] < np.shape(self.RSMangleRef)[1] :
                self.QList[n,15] = self.RSMangle[int(self.QList[n,7]),int(self.QList[n,6])]
                self.QList[n,16] = self.QList[n,15]/(self.deltaTheta*self.convergenceXRays)
                self.QList[n,17] = self.QList[n,16]/self.QList[n,14]
        
    def sortListToRSMQ(self,subtractBG):
        """ This sorts the List Values to the previously created Grid in Qspace, 
            in a second step it integrates over qx in order to get to the rocking Curve
            if the Background has not been calculated previously it is calculated by mapping column 10 of the List to the Qgrid
            if subtractBG is set to True a Background is calculated and scaled to the first 10 points on the left side of the
            rocking curve """
        print('Sorting List to QGrid ... ')
        self.RSMQ = self.RSMQempty
        for n in self.QList[:,0].astype(int):       
            self.RSMQ[int(self.QList[n,2]),int(self.QList[n,1])] = self.QList[n,17]        
        self.rockingQz = np.sum(self.RSMQ,axis = 0)*self.qzDelta*self.qxRange # Integrates over the qxDimension of RSMQ to get the rockingQz
        
        ## Background rescaling takes place here but only once
        if (self.qzBGcalculated == False) and (subtractBG == True):  
            self.rockingQzBG = self.rockingQzBG*np.mean(self.rockingQz[0:10])/np.mean(self.rockingQzBG[0:10])       
            self.qzBGcalculated = True
            
        #Background subtraction from the rocking Curve
        if subtractBG == True:        
            self.rockingQz = self.rockingQz-self.rockingQzBG
            
        self.thetaAxis = h.calcThetaAxis(self.qzGrid)
        header = 'qz Axis [ 1/Ang ]\t thetaAxis\t  X-Ray Intensity I*dOmega*dTheta*Jacobidet [photons/(sec 1/Ang^2)]'
        h.writeColumnsToFile(self.RockingCurveDataFileName,header,self.qzGrid,self.thetaAxis,self.rockingQz)
        
    def calcQzBackground(self):
        # Sort the RSMangleRef matrix to a Q List
        print('Calculating the background')
        self.BGList = self.QRefList
        for n in self.BGList[:,0].astype(int):    
             if self.BGList[n,5] > 0 and  self.BGList[n,7] < np.shape(self.RSMangleRef)[0] and self.BGList[n,6] < np.shape(self.RSMangleRef)[1]:
                self.BGList[n,15] = self.RSMangleRef[int(self.BGList[n,7]),int(self.BGList[n,6])]
                self.BGList[n,16] = self.BGList[n,15]/(self.deltaTheta*self.convergenceXRays)
                self.BGList[n,17] = self.BGList[n,16]/self.BGList[n,14]
                # Sort this Q List to a Background RSM and integrate over it to get a homogenous background RSM 
        self.RSMQBG = self.RSMQempty
        for n in self.BGList[:,0].astype(int):     
            self.RSMQBG[int(self.BGList[n,2]),int(self.BGList[n,1])] = self.BGList[n,17] 
        self.rockingQzBG = np.sum(self.RSMQBG,axis = 0)*self.qzDelta*self.qxRange
        
            
    def createReference(self,peakIdentifier):
        """ This function creates a Reference Curve Fit that is then displayed together with the other fits to see a change"""
        # Selecting the peak Area and extract the data:
        peakData = self.peaks[peakIdentifier] #select the chosen peak data from the peak dictionary
        qz,qx,ROI,qzRocking,qxRocking = h.setROI2D(self.qzGrid,self.qxGrid,self.RSMQ,peakData["qzMin"],peakData["qzMax"],peakData["qxMin"],peakData["qxMax"]) # Calculated the 2D ROI
        
        self.peaks[peakIdentifier]["qzNormalization"] = np.max(qzRocking)
        self.peaks[peakIdentifier]["qxNormalization"] = np.max(qxRocking)
        
        qzRocking = qzRocking/np.max(qzRocking) # Normalize the Integrals to their Gridbox sizes actually only necessary in regions where the grid box size changes       
        qxRocking = qxRocking/np.max(qxRocking) # Normalize the Integrals to their Gridbox sizes actually only necessary in regions where the grid box size changes               
        #self.qz = qz; self.qzRocking = qzRocking;
        ## Parameter modeling: ################################################
        ## Parameter modeling: ################################################
        #model =  lm.models.GaussianModel() + lm.models.GaussianModel() + lm.models.LinearModel()
        modelqz =  lm.models.PseudoVoigtModel(prefix='ka1_Nb_') + lm.models.PseudoVoigtModel(prefix='ka2_Nb_')+lm.models.PseudoVoigtModel(prefix='ka1_Sap_') + lm.models.PseudoVoigtModel(prefix='ka2_Sap_')+lm.models.PseudoVoigtModel(prefix='ka1_Ho_') + lm.models.PseudoVoigtModel(prefix='ka2_Ho_')+lm.models.PseudoVoigtModel(prefix='ka1_Y_') + lm.models.PseudoVoigtModel(prefix='ka2_Y_')+lm.models.PseudoVoigtModel(prefix='ka1_P2_') + lm.models.PseudoVoigtModel(prefix='ka2_P2_')+lm.models.PseudoVoigtModel(prefix='ka1_P1_') + lm.models.PseudoVoigtModel(prefix='ka2_P1_')+ lm.models.LinearModel()
        modelqx =  lm.models.PseudoVoigtModel() + lm.models.LinearModel()
        parsQz = lm.Parameters()
        parsQx = lm.Parameters()
        # Here you can set the initial values and possible boundaries on the fitting parameters
                          #Name       Value                 Vary     Min                   Max                                     
        parsQz.add('ka1_Nb_center',     peakData["qzCenterNb"],    True)
        parsQz.add('ka1_Sap_center',    peakData["qzCenterSap"],    True)
        parsQz.add('ka1_Ho_center',     peakData["qzCenterHo"],    True)
        parsQz.add('ka1_Y_center',      peakData["qzCenterY"],    True)
        parsQz.add('ka1_P2_center',     peakData["qzCenterP2"],    False)
        parsQz.add('ka1_P1_center',     peakData["qzCenterP1"],    False)
        parsQz.add('ka1_Nb_sigma',      peakData["qzWidthNb"],     True)
        parsQz.add('ka1_Sap_sigma',     peakData["qzWidthSap"],     True)
        parsQz.add('ka1_Ho_sigma',      peakData["qzWidthHo"],     True)
        parsQz.add('ka1_Y_sigma',       peakData["qzWidthY"],     True)
        parsQz.add('ka1_P2_sigma',      peakData["qzWidthP2"],     True)
        parsQz.add('ka1_P1_sigma',      peakData["qzWidthP1"],     True)
        parsQz.add('ka1_Nb_amplitude',  peakData["qzAreaNb"] ,     True)
        parsQz.add('ka1_Sap_amplitude', peakData["qzAreaSap"] ,     True)
        parsQz.add('ka1_Ho_amplitude',  peakData["qzAreaHo"] ,     True)
        parsQz.add('ka1_Y_amplitude',   peakData["qzAreaY"] ,     True)
        parsQz.add('ka1_P2_amplitude',  peakData["qzAreaP2"] ,     False)
        parsQz.add('ka1_P1_amplitude',  peakData["qzAreaP1"] ,     False)        
        parsQz.add('ka1_Nb_fraction',   peakData["qzFractionNb"],  True, 0, 1)
        parsQz.add('ka1_Sap_fraction',  peakData["qzFractionSap"],  True, 0, 1)      
        parsQz.add('ka1_Ho_fraction',   peakData["qzFractionHo"],  True, 0, 1)
        parsQz.add('ka1_Y_fraction',    peakData["qzFractionY"],  True, 0, 1)
        parsQz.add('ka1_P2_fraction',   peakData["qzFractionP2"],  True, 0, 1)
        parsQz.add('ka1_P1_fraction',   peakData["qzFractionP1"],  True, 0, 1)       
 
        parsQz.add('ka2_Nb_center',     expr='8047.78/8027.83*ka1_Nb_center')
        parsQz.add('ka2_Sap_center',    expr='8047.78/8027.83*ka1_Sap_center')
        parsQz.add('ka2_Ho_center',     expr='8047.78/8027.83*ka1_Ho_center')
        parsQz.add('ka2_Y_center',      expr='8047.78/8027.83*ka1_Y_center')
        parsQz.add('ka2_P2_center',     expr='8047.78/8027.83*ka1_P2_center')
        parsQz.add('ka2_P1_center',     expr='8047.78/8027.83*ka1_P1_center')
        parsQz.add('ka2_Nb_sigma',      expr='8047.78/8027.83*ka1_Nb_sigma')
        parsQz.add('ka2_Sap_sigma',     expr='8047.78/8027.83*ka1_Sap_sigma')
        parsQz.add('ka2_Ho_sigma',      expr='8047.78/8027.83*ka1_Ho_sigma')
        parsQz.add('ka2_Y_sigma',       expr='8047.78/8027.83*ka1_Y_sigma')
        parsQz.add('ka2_P2_sigma',      expr='8047.78/8027.83*ka1_P2_sigma')
        parsQz.add('ka2_P1_sigma',      expr='8047.78/8027.83*ka1_P1_sigma')
        parsQz.add('ka2_Nb_amplitude',  expr='51/100*ka1_Nb_amplitude')
        parsQz.add('ka2_Sap_amplitude', expr='51/100*ka1_Sap_amplitude')
        parsQz.add('ka2_Ho_amplitude',  expr='51/100*ka1_Ho_amplitude')
        parsQz.add('ka2_Y_amplitude',   expr='51/100*ka1_Y_amplitude')
        parsQz.add('ka2_P2_amplitude',  expr='51/100*ka1_P2_amplitude')
        parsQz.add('ka2_P1_amplitude',  expr='51/100*ka1_P1_amplitude')     
        parsQz.add('ka2_Nb_fraction',   expr='ka1_Nb_fraction')
        parsQz.add('ka2_Sap_fraction',  expr='ka1_Sap_fraction') 
        parsQz.add('ka2_Ho_fraction',   expr='ka1_Ho_fraction')
        parsQz.add('ka2_Y_fraction',    expr='ka1_Y_fraction')
        parsQz.add('ka2_P2_fraction',   expr='ka1_P2_fraction')
        parsQz.add('ka2_P1_fraction',   expr='ka1_P1_fraction')                  
        parsQz.add('slope',         0,    True)
        parsQz.add('intercept',     0, True)
        
                          #Name       Value                 Vary     Min                   Max                                     
        parsQx.add_many(('center',    peakData["qxCenter"],    True),
                        ('sigma',     peakData["qxWidth"],     True),
                        ('amplitude', peakData["qxArea"],     True),
                        ('fraction',  peakData["qxFraction"],  True),
                        ('slope',     0, True),
                        ('intercept', 0, True))
        ## Fitting takes place here
        resultQz = modelqz.fit(qzRocking, parsQz, x = qz)
        resultQx = modelqx.fit(qxRocking, parsQx, x = qx)       
        ## Writing the reference results into the peaks dictionary takes place here this includes also boundaries so that the fitplots are not rescaled in the y direction
        self.peaks[peakIdentifier]["qzReferenceFit"]  = resultQz
        self.peaks[peakIdentifier]["qzYMin"]  = np.min(qzRocking)-0.1*np.min(qzRocking)
        self.peaks[peakIdentifier]["qzYMax"]  = np.max(qzRocking)+0.1*np.max(qzRocking)        
        self.peaks[peakIdentifier]["qxReferenceFit"]  = resultQx        
        self.peaks[peakIdentifier]["qxYMin"]  = np.min(qxRocking)-0.1*np.min(qxRocking)
        self.peaks[peakIdentifier]["qxYMax"]  = np.max(qxRocking)+0.1*np.max(qxRocking)  
        
        ## Plot the Reference Fit
        self.plotFitResult(qz,resultQz,peakData["shortName"],"qz",self.fitGraphicsFolder,self.plotLog)        
        self.plotFitResult(qx,resultQx,peakData["shortName"],"qx",self.fitGraphicsFolder,self.plotLog) 
                    

    def fitPeak(self,peakIdentifier):
        """ This function fits a peak from the RSM for that it goes through the 3 steps:
            1. Cut ROI from the Reciprocal Spacemap according to the boundaries specified in the peaks dictionary for the peakIdentifier
            2. Set the initial fitting parameters from the peaks dictionary (Currently a Gaussian with a linear background is fitted)
            3. Fit the Model
            4. Extract the Fit results into a Dataline"""
        
        # Selecting the peak Area and extract the data:
        peakData = self.peaks[peakIdentifier] #select the chosen peak data from the peak dictionary
        qz,qx,ROI,qzRocking,qxRocking = h.setROI2D(self.qzGrid,self.qxGrid,self.RSMQ,peakData["qzMin"],peakData["qzMax"],peakData["qxMin"],peakData["qxMax"]) # Calculated the 2D ROI
        
        qzRocking = qzRocking/self.peaks[peakIdentifier]["qzNormalization"] # Normalize the Integrals to the maximum value that occured in the chosen reference Measurement
        qxRocking = qxRocking/self.peaks[peakIdentifier]["qxNormalization"] # Normalize the Integrals to the maximum value that occured in the chosen reference Measurement
        
        ## Parameter modeling: ################################################
        #model =  lm.models.GaussianModel() + lm.models.GaussianModel() + lm.models.LinearModel()
        modelqz =  lm.models.PseudoVoigtModel(prefix='ka1_Nb_') + lm.models.PseudoVoigtModel(prefix='ka2_Nb_')+lm.models.PseudoVoigtModel(prefix='ka1_Sap_') + lm.models.PseudoVoigtModel(prefix='ka2_Sap_')+lm.models.PseudoVoigtModel(prefix='ka1_Ho_') + lm.models.PseudoVoigtModel(prefix='ka2_Ho_')+lm.models.PseudoVoigtModel(prefix='ka1_Y_') + lm.models.PseudoVoigtModel(prefix='ka2_Y_')+lm.models.PseudoVoigtModel(prefix='ka1_P2_') + lm.models.PseudoVoigtModel(prefix='ka2_P2_')+lm.models.PseudoVoigtModel(prefix='ka1_P1_') + lm.models.PseudoVoigtModel(prefix='ka2_P1_')+ lm.models.LinearModel()
        modelqx =  lm.models.GaussianModel() + lm.models.LinearModel()
        parsQz = lm.Parameters()
        parsQx = lm.Parameters()
        # Here you can set the initial values and possible boundaries on the fitting parameters
                          #Name       Value                 Vary     Min                   Max                                     
        parsQz.add('ka1_Nb_center',     peakData["qzCenterNb"],    True)
        parsQz.add('ka1_Sap_center',    peakData["qzCenterSap"],    True)
        parsQz.add('ka1_Ho_center',     peakData["qzCenterHo"],    True)
        parsQz.add('ka1_Y_center',      peakData["qzCenterY"],    True)
        parsQz.add('ka1_P2_center',     peakData["qzCenterP2"],    False)
        parsQz.add('ka1_P1_center',     peakData["qzCenterP1"],    False)
        parsQz.add('ka1_Nb_sigma',      peakData["qzWidthNb"],     True)
        parsQz.add('ka1_Sap_sigma',     peakData["qzWidthSap"],     True)
        parsQz.add('ka1_Ho_sigma',      peakData["qzWidthHo"],     True)
        parsQz.add('ka1_Y_sigma',       peakData["qzWidthY"],     True)
        parsQz.add('ka1_P2_sigma',      peakData["qzWidthP2"],     True)
        parsQz.add('ka1_P1_sigma',      peakData["qzWidthP1"],     True)
        parsQz.add('ka1_Nb_amplitude',  peakData["qzAreaNb"] ,     True)
        parsQz.add('ka1_Sap_amplitude', peakData["qzAreaSap"] ,     True)
        parsQz.add('ka1_Ho_amplitude',  peakData["qzAreaHo"] ,     True)
        parsQz.add('ka1_Y_amplitude',   peakData["qzAreaY"] ,     True)
        parsQz.add('ka1_P2_amplitude',  peakData["qzAreaP2"] ,     False)
        parsQz.add('ka1_P1_amplitude',  peakData["qzAreaP1"] ,     False)        
        parsQz.add('ka1_Nb_fraction',   peakData["qzFractionNb"],  False)
        parsQz.add('ka1_Sap_fraction',  peakData["qzFractionSap"],  False)      
        parsQz.add('ka1_Ho_fraction',   peakData["qzFractionHo"],  False)
        parsQz.add('ka1_Y_fraction',    peakData["qzFractionY"],  False)
        parsQz.add('ka1_P2_fraction',   peakData["qzFractionP2"],  False)
        parsQz.add('ka1_P1_fraction',   peakData["qzFractionP1"],  False)
 
        parsQz.add('ka2_Nb_center',     expr='8047.78/8027.83*ka1_Nb_center')
        parsQz.add('ka2_Sap_center',    expr='8047.78/8027.83*ka1_Sap_center')
        parsQz.add('ka2_Ho_center',     expr='8047.78/8027.83*ka1_Ho_center')
        parsQz.add('ka2_Y_center',      expr='8047.78/8027.83*ka1_Y_center')
        parsQz.add('ka2_P2_center',     expr='8047.78/8027.83*ka1_P2_center')
        parsQz.add('ka2_P1_center',     expr='8047.78/8027.83*ka1_P1_center')
        parsQz.add('ka2_Nb_sigma',      expr='8047.78/8027.83*ka1_Nb_sigma')
        parsQz.add('ka2_Sap_sigma',     expr='8047.78/8027.83*ka1_Sap_sigma')
        parsQz.add('ka2_Ho_sigma',      expr='8047.78/8027.83*ka1_Ho_sigma')
        parsQz.add('ka2_Y_sigma',       expr='8047.78/8027.83*ka1_Y_sigma')
        parsQz.add('ka2_P2_sigma',      expr='8047.78/8027.83*ka1_P2_sigma')
        parsQz.add('ka2_P1_sigma',      expr='8047.78/8027.83*ka1_P1_sigma')
        parsQz.add('ka2_Nb_amplitude',  expr='51/100*ka1_Nb_amplitude')
        parsQz.add('ka2_Sap_amplitude', expr='51/100*ka1_Sap_amplitude')
        parsQz.add('ka2_Ho_amplitude',  expr='51/100*ka1_Ho_amplitude')
        parsQz.add('ka2_Y_amplitude',   expr='51/100*ka1_Y_amplitude')
        parsQz.add('ka2_P2_amplitude',  expr='51/100*ka1_P2_amplitude')
        parsQz.add('ka2_P1_amplitude',  expr='51/100*ka1_P1_amplitude')     
        parsQz.add('ka2_Nb_fraction',   expr='ka1_Nb_fraction')
        parsQz.add('ka2_Sap_fraction',  expr='ka1_Sap_fraction') 
        parsQz.add('ka2_Ho_fraction',   expr='ka1_Ho_fraction')
        parsQz.add('ka2_Y_fraction',    expr='ka1_Y_fraction')
        parsQz.add('ka2_P2_fraction',   expr='ka1_P2_fraction')
        parsQz.add('ka2_P1_fraction',   expr='ka1_P1_fraction')                  
        parsQz.add('slope',         0,    True)
        parsQz.add('intercept',     0, True)
        
                          #Name       Value                 Vary     Min                   Max                                     
        parsQx.add_many(('center',    peakData["qxCenter"],    True),
                        ('sigma',     peakData["qxWidth"],     True),
                        ('amplitude', peakData["qxArea"] ,     True),
                        ('fraction',  peakData["qxFraction"],  False),
                        ('slope',     0,   False),
                        ('intercept', 0,   False))
        
        ## Fitting takes place here
        resultQz = modelqz.fit(qzRocking, parsQz, x = qz)
        resultQx = modelqx.fit(qxRocking, parsQx, x = qx)       
        
        ## Writing the results into the peaks dictionary takes place here
        peakData["qzWidthNb"]     = resultQz.values["ka1_Nb_sigma"]
        peakData["qzWidthSap"]     = resultQz.values["ka1_Sap_sigma"]
        peakData["qzWidthHo"]     = resultQz.values["ka1_Ho_sigma"]
        peakData["qzWidthY"]     = resultQz.values["ka1_Y_sigma"]
        peakData["qzWidthP2"]     = resultQz.values["ka1_P2_sigma"]
        peakData["qzWidthP1"]     = resultQz.values["ka1_P1_sigma"]        
        peakData["qzCenterNb"]    = resultQz.values["ka1_Nb_center"]              
        peakData["qzCenterSap"]    = resultQz.values["ka1_Sap_center"] 
        peakData["qzCenterHo"]    = resultQz.values["ka1_Ho_center"] 
        peakData["qzCenterY"]    = resultQz.values["ka1_Y_center"] 
        peakData["qzCenterP2"]    = resultQz.values["ka1_P2_center"] 
        peakData["qzCenterP1"]    = resultQz.values["ka1_P1_center"]         
        peakData["qzAreaNb"]      = resultQz.values["ka1_Nb_amplitude"]
        peakData["qzAreaSap"]      = resultQz.values["ka1_Sap_amplitude"]
        peakData["qzAreaHo"]      = resultQz.values["ka1_Ho_amplitude"]
        peakData["qzAreaY"]      = resultQz.values["ka1_Y_amplitude"]
        peakData["qzAreaP2"]      = resultQz.values["ka1_P2_amplitude"]
        peakData["qzAreaP1"]      = resultQz.values["ka1_P1_amplitude"]
        peakData["qzFractionNb"]      = resultQz.values["ka1_Nb_fraction"]  
        peakData["qzFractionSap"]      = resultQz.values["ka1_Sap_fraction"] 
        peakData["qzFractionHo"]      = resultQz.values["ka1_Ho_fraction"] 
        peakData["qzFractionY"]      = resultQz.values["ka1_Y_fraction"] 
        peakData["qzFractionP2"]      = resultQz.values["ka1_P2_fraction"] 
        peakData["qzFractionP1"]      = resultQz.values["ka1_P1_fraction"] 
        peakData["qzSlope"]     = resultQz.values["slope"]        
        peakData["qzIntercept"] = resultQz.values["intercept"]
        
#        stdDeviationQz             = np.sqrt(np.diag(resultQz.covar))        
#        peakData["qzWidthErr"]     = stdDeviationQz[0]
#        peakData["qzCenterErr"]    = stdDeviationQz[1]
#        peakData["qzAreaErr"]      = stdDeviationQz[2]
#        peakData["qzSlopeErr"]     = stdDeviationQz[3]
#        peakData["qzInterceptErr"] = stdDeviationQz[4]
                
        peakData["qxWidth"]     = resultQx.values["sigma"]
        peakData["qxCenter"]    = resultQx.values["center"]              
        peakData["qxArea"]      = resultQx.values["amplitude"]
        peakData["qxSlope"]     = resultQx.values["slope"]        
        peakData["qxIntercept"] = resultQx.values["intercept"]
        
        #stdDeviationQx             = np.sqrt(np.diag(resultQx.covar))        
        peakData["qxWidthErr"]     = 0#stdDeviationQx[0]
        peakData["qxCenterErr"]    = 0#stdDeviationQx[1]
        peakData["qxAreaErr"]      = 0#stdDeviationQx[2]
       #  peakData["qxSlopeErr"]     = stdDeviationQx[3]
        peakData["qxInterceptErr"] = 0#stdDeviationQx[3]
        
#        ## Plotting all the fit results       
        self.plotFitResult(qz,resultQz,peakData["shortName"],"qz",self.fitGraphicsFolder,self.plotLog)        
        #self.plotFitResult(qx,resultQx,peakData["shortName"],"qx",self.fitGraphicsFolder) 
        
        ## Writing the fit results to a Dataline
#        self.qzCOM,self.qzSTD,self.qzIntegral = h.calcMoments(qz,qzRocking)
#        self.qxCOM,self.qxSTD,self.qxIntegral = h.calcMoments(qx,qxRocking)      
        self.cAxisFitNb  = h.convertQtoD(peakData["qzCenterNb"],2)
        self.cAxisFitSap = h.convertQtoD(peakData["qzCenterSap"],1)
        self.cAxisFitHo  = h.convertQtoD(peakData["qzCenterHo"],2)
        self.cAxisFitY   = h.convertQtoD(peakData["qzCenterY"],2)
#        self.cAxisCOM = h.convertQtoD(self.qzCOM,peakData["order"])
        self.dataLine = np.array([self.identifier[self.currentLine],self.date[self.currentLine],self.time[self.currentLine],self.TB[self.currentLine],self.F[self.currentLine],peakData["qzCenterNb"],self.cAxisFitNb,peakData["qzWidthNb"],peakData["qzCenterSap"],self.cAxisFitSap,peakData["qzWidthSap"],peakData["qzCenterHo"],self.cAxisFitHo,peakData["qzWidthHo"],peakData["qzCenterY"],self.cAxisFitY,peakData["qzWidthY"]])      
   
    def plotFitResult(self,xAxis,result,peak,direction,fitGraphicsFolder,PlotSemilogy):
        ## This function plots the fit Results and saves them to a file
        COM, std,_ = h.calcMoments(xAxis,result.data)
        plt.figure(figsize = (8,5))
        plt.suptitle(self.peaks[peak]["name"] + " " + str(direction)+ self.titletext,fontsize =10)    
        if PlotSemilogy:
                plt.semilogy(xAxis,result.data,'bo',label = "data")
                plt.semilogy(xAxis,self.peaks[peak][str(direction)+"ReferenceFit"].best_fit,"-k",lw = 2,label = "referenceFit")
                plt.semilogy(xAxis,result.init_fit,'k--',label = "initial Guess")
                plt.semilogy(xAxis,result.best_fit,'r-',lw = 2,label = "Best Fit")
                #plt.ylim([1e-4,1])
        else:    
                plt.plot(xAxis,result.data,'bo',label = "data")
                plt.plot(xAxis,self.peaks[peak][str(direction)+"ReferenceFit"].best_fit,"-k",lw = 2,label = "referenceFit")
                plt.plot(xAxis,result.init_fit,'k--',label = "initial Guess")
                plt.plot(xAxis,result.best_fit,'r-',lw = 2,label = "Best Fit")
                plt.ylim([self.peaks[peak][str(direction)+"YMin"],self.peaks[peak][str(direction)+"YMax"]])
        plt.xlabel(str(direction) + r" \left($frac{1}{\AA}$\right)")        
        plt.ylabel("X-Ray Intensity") 
        
        plt.legend(loc = 0)  
        plt.axvline(x=COM,ls = '-',color = 'gray',lw=1, label = "COM = "+str(np.round(COM,4)))
        plt.axvspan(COM-std,COM+std, facecolor='g', alpha=0.1,label = "std =" + str(np.round(std,4))) 
        h.makeFolder(str(fitGraphicsFolder)+"/"+str(peak)+"/"+str(direction))
        plt.savefig(str(fitGraphicsFolder)+"/"+str(peak)+"/"+str(direction)+"/"+str(peak)+"peakFit"+str(direction)+str(self.currentLine)+".png", bbox_inches='tight',dpi=self.dpiValue) 
        plt.show()    
        
    def exportAllRockingCurves(self,subtractBG):
        """ This function contains all methods to export all Rockingcurves of one series to one file """
     
        self.rockingCurveMatrix = np.zeros((self.length+2,np.size(self.qzGrid)+5))
        self.rockingCurveMatrix[0,:] = np.append([-2, -2, -2, -2, -2 ],self.qzGrid)
        self.rockingCurveMatrix[1,:] = np.append([-1, -1, -1, -1, -1 ],h.calcThetaAxis(self.qzGrid))
        
        for i in tqdm(np.arange(0,self.length,1)):                               
            self.loadRSMRaw(i)       #Loads the RSM of a selected Parameterline that will serve as Reference Plot in all Fit Plots
            self.sortRSMangleToQList()    
            self.sortListToRSMQ(subtractBG)     #Sorts the QList to the created Grid and integrates over it, Takes a bolean as argument whether or not a Background should be subtracted                   
            self.rockingCurveMatrix[i+2,:] = np.append([self.identifier[i],self.date[i],self.time[i],self.TB[i],self.F[i]],self.rockingQz)
            print("Completed Step " + str(i) +" / "+ str(self.length-1))
        h.writeListToFile("rockingCurves"+self.sampleName+".dat","identifier\t day\t time\t Temperature [K] \t Fluence [K] \t RockingCurve",self.rockingCurveMatrix)     
    
    def exportRSMQ(self,i):
        """ This function contains all methods to export teh RSMQ to a file """
        savestring = "RSMQS/"+h.timestring(self.number)+".txt"
        np.savetxt(savestring, self.RSMQ)
        h.writeColumnsToFile("RSMQS/qx.dat",u"q_x  (1/Ang) \t q_x  (1/Ang)" ,self.qxBins,self.qxBins)
        h.writeColumnsToFile("RSMQS/qz.dat",u"q_z  (1/Ang) \t q_z  (1/Ang)" ,self.qzBins,self.qzBins)
    
    def fitAllPeaks(self,subtractBG,ReferenceLine):
        """ This function contains the methods to fit all peaks after the initialization has been done """
        self.loadRSMRaw(ReferenceLine)                              #Loads the RSM of a selected Parameterline that will serve as Reference Plot in all Fit Plots
        self.sortRSMangleToQList()    
        self.sortListToRSMQ(subtractBG)     #Sorts the QList to the created Grid and integrates over it, Takes a bolean as argument whether or not a Background should be subtracted                   
        for p in ["MBE2057"]:
            self.createReference(p)            
            h.writeHeaderToFile(p+"FitData"+self.sampleName+".dat","0 identifier\t 1 day\t 2 time\t 3 Temperature [K] \t 4 Fluence [K] \t 5 qzCenterSRO [1/Ang]\t 6 cAxisFitSRO [Ang]\t 7 qzWidthSRO [1/Ang]\t 8 qzCenterNNO [1/Ang]\t 9 cAxisFitNNO [Ang]\t 10 qzWidthNNO [1/Ang]\t 11 qzCenterSTO [1/Ang]\t 12 cAxisFitSTO [Ang]\t 13 qzWidthSTO [1/Ang]\t")
        
        for i in tqdm(range(self.length)):                               
            self.loadRSMRaw(i)                 #Loads the distorted Reciprocal spacemap. Takes the line of the Parameter file as argument
            self.sortRSMangleToQList()    #Creates a List of all Datapoints in omega,theta and qx,qz space, Takes the Detector distance in m as argument
            self.sortListToRSMQ(subtractBG)     #Sorts the List to the created Grid and integrates over it, Takes a bolean as argument whether or not a Background should be subtracted
                        
            for p in ["MBE2057"]:
                self.fitPeak(p)
                h.appendSingleLineToFile(p+"FitData"+self.sampleName+".dat",self.dataLine)
            print("Completed Step " + str(i+1) +" / "+ str(self.length))
      
############ Helper Functions  of this Module: They are only called within other Methods ###########################
            
    def loadFileName(self):
        """ This function creates the relevant FileNames used in the Transformation process. 
        It takes the parameter line from the param list as input"""
        ####### Input Files ############################
        self.Folder = str(int(self.date[self.currentLine]))+ '/' + h.timestring(self.time[self.currentLine])  # Archive in which the input data is located
        h.unzip(self.Folder,self.Folder) # Unzips the Archive if not already done        
        self.FileName = self.Folder + '/rockingcurves' + h.timestring(self.time[self.currentLine]) + '.dat'   # Filename of the File that contains the rockingcurves measured for each angle by PilatusTheta2Theta        
        self.peakRangeFileName = "ReferenceData/fitRanges"+self.sampleName+".txt"        
        
        ###### Output Files ############################        
        ## Graphics ###
        self.GraphicsFolder = self.Folder + '/Graphics'                                 #Folder in which the Graphics files are stored
        h.makeFolder(self.GraphicsFolder) # Creates a Graphics Folder to which all the created Plots are exported if not already done
        self.fitGraphicsFolder = 'fitGraphics'        
        h.makeFolder(self.fitGraphicsFolder)        
        self.RSMRawFileName =  self.GraphicsFolder  +'/RSMRaw.png'                          #Picture FileName for the distorted Reciprocal Spacemap in Angle space
        self.RSMQFileName =  self.GraphicsFolder  +'/RSMQ.png'                          #Picture FileName for the distorted Reciprocal Spacemap in Angle space
        self.ListPlotFileName =  self.GraphicsFolder  +'/ListPlot.png'                      #Picture FileName for the scatter plot of the List
        self.ListPlot2FileName =  self.GraphicsFolder  +'/ListPlot2.png'                      #Picture FileName for the scatter plot of the List
        
        self.RockingCurveAngleFigFileName = self.GraphicsFolder  +'/RockingCurveAngle.png'  #Picture FileName for the RockingCurve in Angle space       
        self.RockingCurveQFigFileName = self.GraphicsFolder  +'/RockingCurveQ.png'
        
        ## Data ##              
        self.RockingCurveDataFileName = self.Folder +'/RockingCurve.dat'   # Contains the RSM integrated over Qx
        self.QRefListFileName = "ExportData/QRefList"
        ## Titletext for Plots ##
        #self.titletext =  self.sampleName + '  ' + str(int(self.IDs[self.currentLine])) +  '  '+ str(int(self.date[self.currentLine])) + '  ' +h.timestring(self.time[self.currentLine])
        
        self.titletext =  self.sampleName 
            
    def make_colormap(self, seq):
        """Return a LinearSegmentedColormap
        seq: a sequence of floats and RGB-tuples. The floats should be increasing
        and in the interval (0,1).
        """
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

    def plotRSMRaw(self):
        """Plots distorted RSM in with omega and detector pixels as axis space.
        The dashed line indicates the current choice of the centerpixel where omega  = theta is assumed """
        print('Plotting RSMraw...')
        
        #Calculating the centerpixel if that has not been set previously
        if self.centerpixel<=0:
            self.centerpixel = h.calcCenterPixel(self.RSMangle,-1)
            
        plt.figure()    
        plt.title(self.titletext, size=10, verticalalignment='top')
        X,Y= np.meshgrid(self.omegaMeasured,self.pixels)
        plt.pcolormesh(X,Y,self.RSMangle/(self.convergenceXRays*self.deltaTheta),norm=matplotlib.colors.LogNorm(),cmap = h.fireice())
        cbar = plt.colorbar()
        cbar.set_label(r'X-Ray reflectivity', rotation=90)
        plt.axis([X.min(), X.max(), Y.min(), Y.max()])
        plt.xlabel(r' angle of incidence $\omega$ [ $^\circ$ ]')
        plt.ylabel('Horizontal Detector pixel ')
        plt.axhline(y=self.centerpixel,lw = '1' ,ls = '--',c = 'k')
        #plt.grid('on')
        plt.savefig(self.RSMRawFileName, dpi=self.dpiValue)
        plt.show()

    def plotList2(self):
        """ Plots the  as a polygonplot"""
        print('Plotting RSMQspace using polygons from List')                    
        print('Step1: Calculating the Polygon paths')                   
                                             
        rvb = self.make_colormap([(16/256,73/256,113/256), (1,1,1) ,0.5, (1,1,1), (180/256,16/256,22/256)])
        
        patchList = []        
        for n in self.List[:,0].astype(int):        
            patch = patches.PathPatch(self.pathList[n])
            patchList.append(patch)    
        p = PatchCollection(patchList,cmap = h.fireice() , array =self.List[:,12],edgecolors = 'none',norm=matplotlib.colors.LogNorm() )

        print('Step2: Plotting the Figure')                    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        fig.title(self.titletext, size=10)
        ax.add_collection(p)
        ax.plot()
        plt.xlabel(r'$\mathrm{q_z}$ $\left( \mathrm{\frac{1}{\AA}} \right)$',fontsize = 14)
        plt.ylabel(r'$\mathrm{q_x}$ $\left( \mathrm{\frac{1}{\AA}} \right)$',fontsize = 14)
        cbar = plt.colorbar(p)  
        cbar.set_label(r'X-Ray reflectivity', rotation=90)
        ax.set_xlim([np.min(self.List[:,7]),np.max(self.List[:,7])])        
        ax.set_ylim([np.min(self.List[:,8]),np.max(self.List[:,8])])
#        plt.grid('on')        
        plt.savefig(self.ListPlot2FileName, bbox_inches='tight',dpi = 1200)
        plt.show()

    
    def plotList(self):
        """ Plots the  as a scatterplot of measured points in Qspace where the color of the points corresponds to the detected X-Ray intensity in photons/second. """
        print('Plotting RSMQspace from List')
        rvb = self.make_colormap([(16/256,73/256,113/256), (1,1,1) ,0.5, (1,1,1), (180/256,16/256,22/256)])
        
        plt.figure()
        plt.title(self.titletext, size=10)
        plt.scatter(self.List[:,7],self.List[:,8],c = self.List[:,12], cmap=h.fireice(), norm=matplotlib.colors.LogNorm(),edgecolor='None',s = 2)
        cbar = plt.colorbar()
        cbar.set_label(r'X-Ray reflectivity', rotation=90)
        plt.xlabel(r'$\mathrm{q_z}$ $\left( \mathrm{\frac{1}{\AA}} \right)$',fontsize = 14)
        plt.ylabel(r'$\mathrm{q_x}$ $\left( \mathrm{\frac{1}{\AA}} \right)$',fontsize = 14)
        plt.xlim([np.min(self.List[:,7]),np.max(self.List[:,7])])        
        plt.ylim([np.min(self.List[:,8]),np.max(self.List[:,8])])
        #plt.grid('on')
        plt.savefig('RSMList/'+h.timestring(self.number)+'.png', bbox_inches='tight',dpi = self.dpiValue)
        plt.show()
        
    
        
    def plotRSMQ(self):
        """ This function plots the RSM in Q-Space that was previously created"""
        print('Plotting RSMQ ... ')        
        plt.figure()
        plt.title(self.titletext, size=14, verticalalignment='top')
        X,Y= np.meshgrid(self.qzBins,self.qxBins)
        plt.pcolormesh(X,Y,self.RSMQ,norm=matplotlib.colors.LogNorm(), cmap = h.fireice())
        cbar = plt.colorbar()
        cbar.set_label(r'X-Ray reflectivity ', rotation=90)
        #plt.yticks([-0.2,-0.1,0,0.1,0.2])
        plt.xlabel(r'$\mathrm{q_z}$ $\left( \mathrm{\frac{1}{\AA}} \right)$',fontsize = 14)
        plt.ylabel(r'$\mathrm{q_x}$ $\left( \mathrm{\frac{1}{\AA}} \right)$',fontsize = 14)
        #plt.grid('on')
        plt.savefig('RSMQS/'+h.timestring(self.number)+'.png', bbox_inches='tight',dpi=self.dpiValue)
        plt.show()
        
    def plotRockingCurveQz(self,PlotSemilogy):
        """ Used to plot the created Rocking Curve """
        print('Plotting Rockingcurve in Q-Space')
        plt.figure()      
        plt.title(self.titletext, size=10, verticalalignment='top')
        if PlotSemilogy:              
            plt.semilogy(self.qzGrid,self.rockingQz,'o-g',lw = 2,label="RockingCurve BG subtracted")
            plt.semilogy(self.qzGrid,self.rockingQz+self.rockingQzBG,'-b',lw = 2,label="RockingCurve Raw")            
            plt.semilogy(self.qzGrid,self.rockingQzBG,'-k',lw = 2,label="Background")
        else:
            plt.plot(self.qzGrid,self.rockingQz,'o-g',lw = 2,label="RockingCurve BG subtracted")
            plt.plot(self.qzGrid,self.rockingQz+self.rockingQzBG,'-b',lw = 2,label="RockingCurve Raw")            
            plt.plot(self.qzGrid,self.rockingQzBG,'-k',lw = 2,label="Background")
        plt.xlabel('qz [ 1/Ang ]')
        plt.ylabel( r'X-Ray reflectivity')
        plt.xlim([np.min(self.qzGrid),np.max(self.qzGrid)])
        plt.legend(loc = 0)
       
        plt.grid('on')
        plt.savefig(self.RockingCurveQFigFileName, bbox_inches='tight',dpi=self.dpiValue)    
        plt.show()
    def plotPeak(self,peakIdentifier,PlotSemilogy):
        """ Plots the peak region of the RSM that corresponds to the given Identifiere from the self.peaks dictionary
            This includes integrals over """
        
        peakData = self.peaks[peakIdentifier] #select the chosen peak data from the peak dictionary
        qz,qx,ROI,qzRocking,qxRocking = h.setROI2D(self.qzGrid,self.qxGrid,self.RSMQ,peakData["qzMin"],peakData["qzMax"],peakData["qxMin"],peakData["qxMax"]) # Calculated the 2D ROI
        qzRocking = qzRocking/np.max(qzRocking)
        qxRocking = qxRocking/np.max(qxRocking)
        #_,qzRockingWithoutBG = h.setROI1D(self.qzGrid,self.rockingQz,peakData["qzMin"],peakData["qzMax"]) # Get the rocking curve of the same region where the Background has been removed
        ## Plotting takes place here
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize = (8,6))
        f.suptitle(peakData["name"] + "-Peak", size=16, verticalalignment='top')
        X,Y= np.meshgrid(qz,qx)
        if PlotSemilogy == True:
            ax1.pcolor(X,Y,ROI,norm=matplotlib.colors.LogNorm())            
            ax2.semilogx(qxRocking,qx,'o-b',lw = 2,label="Integrated over qz")                
            ax3.semilogy(qz,qzRocking,'o-b',lw = 2,label="Integrated over qx")
            #ax3.semilogy(qz,qzRockingWithoutBG,'o-g',lw = 2,label="Background subtracted")
        else:
            ax1.pcolor(X,Y,ROI)            
            ax2.plot(qxRocking,qx,'o-b',lw = 2,label="Integrated over qz")                           
            ax3.plot(qz,qzRocking,'o-b',lw = 2,label="Integrated over qx")
            #ax3.plot(qz,qzRockingWithoutBG,'o-g',lw = 2,label="Background subtracted")
        
        ### Top Left Color Plot of the Peak #####
        ax1.axis([X.min(), X.max(), Y.min(), Y.max()])
        ax1.xaxis.tick_top()            
        ax1.xaxis.set_label_position("top") 
        ax1.set_xlabel('qz [ 1/Ang ]')
        ax1.set_ylabel('qx [ 1/Ang ]')
        ax1.grid('on')        
        
        ## Top Right qx-Rocking Curve of the Peak
        ax2.set_ylabel("qx [1/Ang]")        
        ax2.yaxis.tick_right()      
        ax2.yaxis.set_label_position("right")
        ax2.set_xlabel("Normalized X-Ray Reflectivity")
        ax2.grid('on')           
        
        ## Bottom Left qz-Rocking Curve of the Peak        
        ax3.set_xlabel("qz [1/Ang]") 
        ax3.set_ylabel("Normalized X-Ray Reflectivity")                 
        ax3.grid('on')  
        #ax3.legend(loc = (1.05,0.0))         
        ax4.axis('off')
        plt.savefig(self.GraphicsFolder + "/Overview" +peakData["shortName"] + "-Peak.png", bbox_inches='tight',dpi=self.dpiValue) 
        plt.show()
        
    def exportGrid(self):
        """ This function plots the RSM in Q-Space that was previously created"""
        print('Exporting Grid ')        
        plt.figure()
        plt.title(self.titletext, size=14, verticalalignment='top')
        X,Y= np.meshgrid(self.qzBins,self.qxBins)
        plt.pcolormesh(X,Y,self.RSMQ,norm=matplotlib.colors.LogNorm(), cmap = h.fireice())
        plt.axis([X.min(), X.max(), Y.min(), Y.max()])
        cbar = plt.colorbar()
        cbar.set_label(r'X-Ray reflectivity ', rotation=90)
        #plt.yticks([-0.2,-0.1,0,0.1,0.2])
        plt.xlabel(r'$\mathrm{q_z}$ $\left( \mathrm{\frac{1}{\AA}} \right)$',fontsize = 14)
        plt.ylabel(r'$\mathrm{q_x}$ $\left( \mathrm{\frac{1}{\AA}} \right)$',fontsize = 14)
        #plt.grid('on')
        plt.savefig('RSMQS/'+h.timestring(self.number)+'.png', bbox_inches='tight',dpi=self.dpiValue)
        plt.show()


if __name__ == '__main__':
    # Necessary Parameters go here        
    sampleName = 'SROSL5'
    detectorDistance =  1.33   # Distance between sample and detector in m. Should be adjusted so that the peaks appear straigth in the RSMQ
    centerpixel = -1            # External specification of a centerpixel. If set to a negative value it is automatically determined by the Reference List calculation
    crystalThreshold = 0        # This is the crystal Voltage change(!) threshold. Values with a lower Crystal Voltage change are neglected 
    subtractBG = False           # Do you wish to subtract a background that corresponds to a homogenous illumination in real space
    plotLog = True             # Should the graphcis be displayed in semilogarithmic mode
    referenceLine = 1           # Selection to the Line of the Parameterfile that is supposed to be used as Reference Measurement
    dataLine      = 1               # Selection of a specific Dataline that should be plotted
    # Initializing the Routine
    x= TransformStatic()            #Loads the class
#     x.cores = 5                # Set max. CPU Cores (not set: cores-2)
#     x.plotLog = plotLog
#     x.loadParameters(sampleName)    #Reads in the parameter file that contains all measurements that are to be evaluated
#     x.calcReferenceList(referenceLine,detectorDistance,centerpixel,crystalThreshold) # Does some initial calculations concerning the steps in Q and Omega space that need to be done only one time
#     x.plotList()                   #Optional: Plots all points of the list in a scatterplot: Visualizes potential nonlinear sampling
# #    x.plotList2()    
#     x.createQGrid(2.9,3.5,-0.04,0.04,1,6)             #Creates a Grid in Q space to which the List is later sorted. Takes QxOversampling and QzOversampling as arguments. 
#     #x.createMappingMultiCore() # This function creates a Mapping
#     #x.saveMapping2txt()
     
#     #x.saveMapping2np()
    
#     x.loadMappingfromtxt("ExportData/QRefList16.dat")  # This function loads a mapping from the Hard disk if it was already created

# #    x.loadMappingfromnp("ExportData/QRefList16.npz")
#     x.calcQzBackground() # This function calculates a Background from a previously specified region of interest Only necessary in case you want to do a BG-subtraction
# # #    
# # #    
# # #    ### Possible set of commands ##############
# # #    #Plot the RSMQ of a certain Dataline
     
#     x.loadRSMRaw(dataLine)          #Loads the Data from angle space. Takes the line of the Parameter file as argument
#     x.sortRSMangleToQList()         #Sorts the RSM from angle space into a List of Data in Q-space
#     x.sortListToRSMQ(False)         #Sorts the List to the created Grid and integrates over it, Takes a bolean as argument whether or not a Background should be subtracted
#     x.plotRSMQ()
#     x.plotRockingCurveQz(plotLog)

# # #    for p in ["MBE2057"]:
# # #        x.plotPeak(p,True)         #Optional: Plots the RSM-cut of a selected Peak and the integrals along both directions, Takes a Bolean as second argument that indicates logarithmic plotting
    
# # #    for p in ["MBE2057"]:
# # #       x.createReference(p)         #Optional: Plots the RSM-cut of a selected Peak and the integrals along both directions, Takes a Bolean as second argument that indicates logarithmic plotting

# # #    for i in range(x.length):
# # #%%
#     for i in range(x.length):
#         x.number = i
#         print(i)
#         x.loadRSMRaw(i)                 #Loads the distorted Reciprocal spacemap. Takes the line of the Parameter file as argument
#         x.plotRSMRaw()
#         x.plotList()
#         x.sortRSMangleToQList()
#         x.sortListToRSMQ(False)          #Sorts the List to the created Grid and integrates over it, Takes a bolean as argument whether or not a Background should be subtracted
#         x.plotRSMQ()
#         x.exportRSMQ(i)

# # #        for p in ["MBE2057"]:
# # #            x.fitPeak(p)
# # #            h.appendSingleLineToFile(p+"FitData"+x.sampleName+".dat",x.dataLine)
    
# #     #### Possible Plotting commands    ###########################################
# #     #x.plotRSMRaw()                 #Optional: Plots the raw Data
# #     #x.plotList()                   #Optional: Plots all points of the list in a scatterplot: Visualizes potential nonlinear sampling
# #     #x.plotList2()                   #Optional: Plots all points of the list in using the corresponding polygons:
# #     #x.plotRSMQ() 
# #     #for p in ["Si","Au"]:
# #     #   x.plotPeak(p,True)         #Optional: Plots the RSM-cut of a selected Peak and the integrals along both directions, Takes a Bolean as second argument that indicates logarithmic plotting
    
# #     #### What you want to do
#     x.exportAllRockingCurves(False)
# #    x.fitAllPeaks(subtractBG,referenceLine)
    