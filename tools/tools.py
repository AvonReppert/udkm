import os
import numpy as np
import zipfile 
import datetime 
import matplotlib


class tools:
    '''
    Version 2: added the possibility to append to file
        clear distinction between adding a single line to the file 
        --WriteSingleLineOfListToFile or AppendSingleLineOfListToFile-- 
        and the full list --WriteFullListToFile--(first case being useful 
        for a series of fits of results coming from different files, the second
        to save plot data)
    Version 2.1 MultiColumnMultiArray added
    Version 3   Added a uniform Documentation style
                Changed Comment Marker % to # for better LaTeX readability
                Changed name MultiColumnMultiArray to WriteColumnsToFile
                Changed the Function unzip
                Added the Function makeFolder
                Changed name AppendSingleOfListToFile to appendSingleLineToFile
                Changed the order of Arguments to be consistent to: Filename, *Header, *Data
                Added the function writeLinesToFile as a replacement to WriteRockingCurveToFile
                Added the function Timestamp
                Added the function removeBadData
                Added the function removeBadData
                Added the function regridArray
                Added the function NormalizeColumnsToValue
                Added the function Smooth
    Version 3.1 Added the function analyzeParameters

                
    '''
    __version__='4'
    
    def __init__(self):
        self.PXSwl = 1.5418              #Cu K-alpha Wavelength in [Ang]
        self.PXSk = 2*np.pi/self.PXSwl    #absolute value of the incident k-vector in [1/Ang]

    def analyzeParameters(self,filename):
        """returns the the parameters saved in the parameter-file
        as dictonary
        
        Parameters
        ----------
        filename : filename
            interplanarDistance in Angström 
        
        Returns
        ------
            parameters : dictonary
            
        Example
        ------- """
        with open (filename, "r") as myfile:
            data=myfile.readlines()
            param = {}
            comment=[]
            
        for i in range(len(data)):
        
            if data[i].find('Date Time:') != -1:
                line = data[i]
                datetime = line[11:len(line)-1]
                param.update({'Date Time':datetime})
            
            elif data[i].find('Delay Vector:') != -1:
                line = data[i]
                delaysstr1 = line[14:len(line)-1]
                delaysstr2 = delaysstr1.split(',')
                delays=[]
                for j in range(len(delaysstr2)):
                    delaysstr3 = delaysstr2[j].split(':')
                    if len(delaysstr3) == 1:   
                        delays.append(float(delaysstr3[0]))    
                    else:
                        delaysstr4 = [float(k) for k in delaysstr3]
                        delaysstr5 = list(np.arange(delaysstr4[0],delaysstr4[2]+1,delaysstr4[1]))
                        delays.extend(delaysstr5)              
                delays = np.transpose(delays)
                delays = np.unique(delays)
                param.update({'Delay Vector':delays})
                
            elif data[i].find('Delay Loop:') != -1:
                line = data[i]
                delayloop = int(line[12:len(line)-1])
                param.update({'Delay Loop':delayloop})
                
            elif data[i].find('Delay Order:') != -1:
                line = data[i]
                delayorder = line[13:len(line)-1]
                param.update({'Delay Order':delayorder})        
                
            elif data[i].find('Theta Vector:') != -1:
                line = data[i]
                thetavec1 = line[14:len(line)-1]
                thetavec2 = thetavec1.split(',')
                thetas=[]
                for j in range(len(thetavec2)):
                    thetavec3 = thetavec2[j].split(':')
                    if len(thetavec3) == 1:   
                        thetas.append(float(thetavec3[0]))    
                    else:
                        thetavec4 = [float(k) for k in thetavec3]
                        thetavec5 = list(np.arange(thetavec4[0],thetavec4[2]+1,thetavec4[1]))
                        thetas.extend(thetavec5)              
                delays = np.transpose(thetas)
                delays = np.unique(thetas)
                param.update({'Theta Vector':thetas})
                
                
            elif data[i].find('Theta Loop:') != -1:
                line = data[i]
                thetaloop = int(line[12:len(line)-1])
                param.update({'Theta Loop':thetaloop})
        
            elif data[i].find('Theta Order:') != -1:
                line = data[i]
                thetaorder = line[13:len(line)-1]
                param.update({'Theta Order':thetaorder})
                
            elif data[i].find('2 Theta Correction:') != -1:
                line = data[i]
                line = line.replace(',','.')
                thetacorr = float(line[20:len(line)-1])
                param.update({'2 Theta Correction':thetacorr})
           
            elif data[i].find('Inner Parameter:') != -1:
                line = data[i]
                innerParam = line[17:len(line)-1]
                param.update({'Inner Parameter':innerParam})
                
            elif data[i].find('Exposure Time:') != -1:
                line = data[i]
                line = line.replace(',','.')               
                exposure = float(line[15:len(line)-2])
                param.update({'Exposure Time':exposure})
                
            elif data[i].find('Calibration:') != -1:
                line = data[i]
                line = line.replace(',','.')
                line = line[13:len(line)-1]
                cal = [float(j) for j in line.split('\t')]
                param.update({'Calibration':cal})
                   
            elif data[i].find('ROI') != -1:
                line = data[i][data[i].find('('):len(data[i])]
                start, stop = line.split('; ')
                start = start[1:len(start)-1]
                stop = stop[1:len(stop)-1]
                startx, starty = start.split(', ')
                stopx, stopy = stop.split(', ')
                ROI = [int(startx),int(starty),int(stopx),int(stopy)]
                param.update({'ROI':ROI})
         
            elif data[i].find('File Name:') != -1:
                line = data[i]
                filename = int(line[11:len(line)-1])
                param.update({'File Name':filename})
                
            elif data[i].find('Tmp Path:') != -1:
                line = data[i]
                tmppath = line[10:len(line)-1]
                param.update({'Tmp Path:':tmppath})
                
            elif data[i].find('Data Path:') != -1:
                line = data[i]
                datapath = line[11:len(line)-1]
                param.update({'Data Path':datapath})
               
            else:
                if data[i] != '\n':
                    comm = data[i].split('\n')
                    comment.append(comm[0])
    #                print(i)
    #                print(comm)
                    param.update({'Comment':comment})       
        return param        
        
    def calcFourierLimit(self, wl,deltaWl):
        c = 299792458
        return wl**2*0.441/(c*deltaWl)
    
    def calcFluence(self,power,FWHMx,FWHMy,angle,repRate):
        """returns the fluence         
        Parameters
        ----------
        power : float 
            incident power in mW 
        FWHMx : float 
            Full Width at Half Maximum of the (gaussian) pump beam in x-direction in cm
        FWHMy : float 
            Full Width at Half Maximum of the (gaussian) pump beam in y-direction in cm
        angle : float 
            angle of incidence of the pump beam in degree relative to the sample surface
        repRate : float 
            repetition rate of the used laser
            
        Returns
        ------
            fluence : float
              calculated fluence in mJ/cm^2
            
        Example
        ------ 
            >>> calcFluence(50,500*1e-4,400*1e-4,45,1000) """
         #Umrechung von Grad in Bogenmaß
        angleRad = np.radians(angle)
     #Berechnung der Fluenz
        x0 = FWHMx/(2*np.sqrt(np.log(2)))
        y0 = FWHMy/(2*np.sqrt(np.log(2)))
        A0 = np.pi*x0*y0
        fluence= power*np.sin(angleRad)/(repRate*A0) #in mJ/cm^2   
        return fluence
    
        
    def sortArray(self,array,column):
        """ Sorts a 2D numpy array by the specified column so that the values in this column are increasing"""
        return array[array[:,column].argsort()]
        
    def fireice(self):
        """Returns a self defined analog of the colormap fireice"""
        cdict = {'red':    ((0.0,  0.75, 0.75),
                    (1/6,  0, 0),
                    (2/6,  0, 0),
                    (3/6,  0, 0),
                    (4/6,  1.0, 1.0),
                    (5/6,  1.0, 1.0),
                    (1.0,  1.0, 1.0)),

         'green': ((0.0,  1, 1),
                    (1/6,  1, 1),
                    (2/6,  0, 0),
                    (3/6,  0, 0),
                    (4/6,  0, 0),
                    (5/6,  1.0, 1.0),
                    (1.0,  1.0, 1.0)),

         'blue':  ((0.0,  1, 1),
                    (1/6,  1, 1),
                    (2/6,  1, 1),
                    (3/6,  0, 0),
                    (4/6,  0, 0),
                    (5/6,  0, 0),
                    (1.0,  0.75, 0.75))}
        fireice = matplotlib.colors.LinearSegmentedColormap('fireice', cdict)
        return(fireice)
        
    def calcAngle(self,d,orderOfReflex):
        """returns the angle of incidence at which a reflex with a given Order
        and interplanar distance to occur at our Plasma X-Ray source
        
        Parameters
        ----------
        d : float
            interplanarDistance in Angström 
        
        orderOfReflex : integer
            orderOfReflex   
        
        Returns
        ------
            omega : float
            angle of incidence in a theta2theta scan in degrees 
            
        Example
        ------- 
            >>> omegaSap = calcAngle(dSap,1)"""
        
        omega = self.degrees(np.arcsin((orderOfReflex*self.PXSwl)/(2*d)))
        return(omega)
        
    def relChange(self,dataArray,fixpoint):
        """returns relative change of a measured quantity relative to a given value
        
        Parameters
        ----------
        dataArray : 1D numpy array
            Array containing the data 
        
        fixpoint : float
            Quantity to which the relative change is calculated   
        
        Returns
        ------
            relChangeArray : 1D numpy array with same length as dataArray
            contains (dataArray-fixpoint)/fixpoint
        Example
        ------- 
            >>> change = relChange(cAxisDy,cAxisDy[T==300])"""
        relChangeArray = (dataArray-fixpoint)/fixpoint        
        return(relChangeArray)
    
    def convertQtoD(self,q,orderOfReflex):
        """converts the qAxis to an interatomic distance
        Parameters
        ----------
        q : 1D numpy array
            qAxis usually in 1/Ang 
        
        orderOfReflex : integer
            order of the Bragg reflex
               
        Returns
        ------
            d : 1D numpy array with same length as q
        Example
        ------- 
            >>> d = convertQtoD(q,2)"""
        d = orderOfReflex*2*np.pi/q
        return(d)
        
    def calcMoments(self,xAxis,yValues):
        """ calculates the Center of Mass, standard Deviation and Integral of a given Distribution
        
        Parameters
        ----------
            xAxis : 1D numpy array 
                numpy array containing the x Axis
            yValues : 1D numpy array
                numpy array containing the according y Values
            
        Returns
        ------
        in that order
            COM : float
                xValue of the Center of Mass of the data
                
            STD : float
                xValue for the standard deviation of the data around the center of mass
                
            integral : 
                integral of the data
                
        Example
        ------- 
            >>> COM,std,integral = calcMoments([1,2,3],[1,1,1])
                sould give a COM of 2, a std of 1 and an integral of 3 """
        
        COM     = np.average(xAxis,axis=0,weights=yValues)
        STD     = np.sqrt(np.average((xAxis-COM)**2, weights=yValues)) 
        delta = self.calcGridBoxes(xAxis)[0]
        integral = np.sum(yValues*delta)
        return COM,STD,integral
    
    def setROI2D(self,xAxis,yAxis,Matrix,xMin,xMax,yMin,yMax):
        """ selects a rectangular region of intrest ROI from a 2D Matrix based on the 
        passed boundaries xMin,xMax, yMin and yMax. x stands for the columns of the 
        Matrix, y for the rows
        
        Parameters
        ----------
            xAxis, yAxis : 1D numpy arrays 
                numpy arrays containing the x and y grid respectively
            Matrix : 2D numpy array
                2D array with the shape (len(yAxis),len(xAxis))
            xMin,xMax,yMin,yMax : inclusive Boundaries for the ROI
            
        Returns
        ------
        in that order
            xROI : 1D numpy array slice of xAxis between xMin and xMax
                   
            yROI : 1D numpy array slice of yAxis between yMin and yMax
            
            ROI : 2D numpy array of same length as vector
            
            xIntegral : 1D numpy arrays with the same length as xROI
                array containing the sum of ROI over the y direction
                
            yIntegral : 1D numpy arrays with the same length as yROI
                array containing the sum of ROI over the x direction

        Example
        ------- 
            >>> qzCut,qxCut,ROI,xIntegral,yIntegral = setROIMatrix(qzGrid,qxGrid,RSMQ,2.1,2.2,-0.5,0.5)"""
        
        selectX = np.logical_and( xAxis >= xMin, xAxis <= xMax)
        selectY = np.logical_and(yAxis >= yMin,yAxis <= yMax)
        
        xROI = xAxis[selectX]
        yROI = yAxis[selectY]
        ROI  = Matrix[selectY,:]
        ROI = ROI[:,selectX]
        xIntegral = np.sum(ROI,0)
        yIntegral = np.sum(ROI,1)
        return xROI,yROI,ROI,xIntegral,yIntegral 
    
    def setROI1D(self,xAxis,values,xMin,xMax):
        """ selects a rectangular region of intrest ROI from a 2D Matrix based on the 
        passed boundaries xMin,xMax, yMin and yMax. x stands for the columns of the 
        Matrix, y for the rows
        
        Parameters
        ----------
            xAxis: 1D numpy arrays 
                numpy array containing the x grid
            Values : 1D numpy array same length as xAxis
               
            xMin,xMax: inclusive Boundaries for the ROI
            
        Returns
        ------
        in that order
            xROI : 1D numpy array slice of xAxis between xMin and xMax
            
            ROI : 1D numpy array of same length as vector
            
           
        Example
        ------- 
            >>> qzCut,ROI = setROIMatrix(qzGrid,RockingQz,2.1,2.2)"""
        
        selectX = np.logical_and( xAxis >= xMin, xAxis <= xMax)
        xROI = xAxis[selectX]
        ROI  = values[selectX]        
        return xROI,ROI
    
    def calcGridBoxes(self,grid):
         """This function calculates the size of a grid cell and the left and right boundaries of each gridcell in a monotonous but
         possibly nonlinear grid.
        
        Parameters
        ----------
            vector : 1D numpy array 
                     numpy array containing a monotonous grid with points
        Returns
        ------
        in that order
            delta : 1D numpy array of same length as vector
                    the distance to the left and right neighbor devided by 2
            l : 1D numpy array of same length as vector
                left boundaries of the grid cell
            r : 1D numpy array of same length as vector
                right boundaries of the grid cell        
        Example
        ------- 
            >>> delta,right,left = calcGridBoxes([0,1,2,4])
                (array([ 1. ,  1. ,  1.5,  2. ]),
                 array([-0.5,  0.5,  1.5,  3. ]),
                 array([ 0.5,  1.5,  3. ,  5. ]))"""
            
         delta = np.zeros(len(grid)) 
         r =  np.zeros(len(grid)) 
         l =  np.zeros(len(grid)) 
         for n in range(len(grid)):
             if (n == 0):                           
                 delta[n] = grid[n+1] -grid[n]
                 l[n] = grid[n]-delta[n]/2
                 r[n] = np.mean([grid[n],grid[n+1]])
             elif n == (len(grid)-1):
                 delta[n] = grid[n] - grid[n-1]                
                 l[n] = np.mean([grid[n],grid[n-1]])
                 r[n] = grid[n] +delta[n]/2
             else:
                 l[n] = np.mean([grid[n],grid[n-1]])
                 r[n] = np.mean([grid[n],grid[n+1]])
                 delta[n] = np.abs(r[n]-l[n])
         return delta,l,r
         
    def calcCenterPixel(self,RSMRaw,centerpixel):
        """ Calculates the centerpixel from a raw Reciprocal Spacemap but only if no positive value for centerpixel is manually given
        
        Parameters
        ----------
            RSMRaw : 2D numpy array 
                    Numpy array containing a distorted RSMRaw
            centerpixel :  integer
                    input centerpixel: If it is positive this is the return value. If not 
                    the centerpixel is calculated as to be the pixel with the maximum intensity of the RSMraw

        Returns
        ------
            centerpixel : integer
                    The pixel with the maximum intensity of the RSMraw or the used defined value
                    
        Example
        ------- 
            >>> centerpixel = h.returnTimeZero(RSMraw,-1)"""
            
        if centerpixel >=0:
            centerpixel = centerpixel
        else:
            centerpixel = np.argmax(np.sum(RSMRaw,1))        
            return centerpixel
    def returnTimeZero(self,fileName,date,time):
        """This function returns t0 and and the crystalOffset measured at a particular day
            
            Parameters
            ----------
            fileName: String 
                    Filename of the timeZero file
            date :  integer
                    date at which the most recent time zero measurement is asked
            time : integer
                   time at which the most recent time zero measurement is asked
            

            Returns
            ------
            t0 : float
                 the most recent timezero matching the inputs
            CrystalOffset : float
                the most recent recorded crystal offset of that day
               
            Example
            ------- 
            >>> t0,offset = h.returnTimeZero("TimeZero.dat",20140801,120000)"""
        Data = np.genfromtxt(fileName,comments = "#",delimiter = "\t")
        timeStamp = self.timeStamp(str(date),self.timestring(time))
        if np.sum(Data[:,0]<=timeStamp) > 0:
            t0 = Data[Data[:,0]<=timeStamp,3][-1]
            CrystalOffset = Data[Data[:,0]<=timeStamp,4][-1]
        else:
            t0 = Data[0,3]
            CrystalOffset = Data[0,4]
            print("Warning ! Time zero has been requested at a day before the first measurement was taken. I take the first available time")
        return t0,CrystalOffset

    def returnExtremum(self,Data,identifier):
        """This function returns the data belonging to an input identifier. It selects the according
            
            Parameters
            ----------
            Data : 2D numpy array 
                    numpy array containing a Datamatrix
            identifier :  float
                    identifier of the Dataset
            
            Returns
            ------
            tMin : float
                    extracted Minimum
            Min: float
                 amplitude of the extracted Minimum
            tMax :float
                  extracted Maximum timing
            Max: float
                amplitude of the extracted Maximum
               
            Example
            ------- 
            >>> tMin,Min,tMax,Max = h.returnExtremumg(DataExt,ID)"""
        select = Data[:,0]==identifier
        tMin = Data[select,5]
        Min = Data[select,6]
        tMax = Data[select,7]
        Max = Data[select,8]
        return tMin,Min,tMax,Max
    
    def returnData(self,Data,identifier,colx,coly,colT,colF):
        """This function returns the data belonging to an input identifier. It selects the according
            
            Parameters
            ----------
            Data : 2D numpy array 
                    numpy array containing a Datamatrix
            identifier :  float
                    identifier of the Dataset
            colx : integer
                   Column from which the x-Values are be selected usually the delay column
            coly : integer
                   Column from which the y-Values are be selected usually one of the following
                   dcc0 column 10 
                   dww0 column 13
                   dcc0COM column 20
            colT: integer 
                  Column containing the temperature values of a measurement
                  
            colF: integer 
                  Column containing the Fluence values of a measurement 

            Returns
            ------
            x : 1D numpy array
                usually the delay vector
            y : 1D numpy vector
                the quantity on a y axis
            Tplot : float
                The temperature of the selected measurement
            Fplot : 
                The fluence of the selected measurement
            date :
                The date of the selected measurement
            time :
                The time of the selected measurement
               
            Example
            ------- 
            >>> x,y,Tplot,Fplot,date,time = h.returnData(Data,i,6,coldcc0,colT,colF)"""
        select = Data[:,0]==identifier
        Tplot = np.unique(Data[select,colT])
        Fplot = np.unique(Data[select,colF])
        date = np.unique(Data[select,1])
        time = np.unique(Data[select,2])
        x = Data[select,colx]
        y = Data[select,coly]
        return x,y,Tplot,Fplot,date,time

    def returnIds2(self,Data,temperatures,fluences,dates,times,wavelengths,spot,power):   
        """This function returns the identifiers that belong to datasets that match the given 
        temperature, fluence, and dates specified in the input. If one quantity is not specified 
        then all data of that quantity are returned.
            
            Parameters
            ----------
            Data : 2D numpy array 
                    numpy array containing a Datamatrix
            temperatures :  1D numpy array
                    temperatures in K that are to be included in the selection
            fluences : 1D numpy array
                    fluences in mJ that are to be included in the selection 
            dates : 1D numpy array
                    dates that are to be included in the selection    
            
            
            Returns
            ------
            ids :  1D numpy array
                    A numpy array with the unique identifiers where all input requirements are met
               
            Example
            ------- 
            >>> DataArray = np.array([ 0,9,9],
                                     [ 1,9,9],
                                     [ 2,8,8]]
            >>> ids = returnIDs(Data,[250,200],[3,6],[20140801])
            will return for example [0,1] """

        colIDs = 0
        
        colDate = 1
        colTime = 2
        colT = 3
        colF = 4
        colWL = 6
        colSpot = 7 
        colPower = 8
        
        idT     = self.findIDs(Data,temperatures,colT,colIDs)
        idF     = self.findIDs(Data,fluences,colF,colIDs)
        idDate  = self.findIDs(Data,dates,colDate,colIDs)
        idTime  = self.findIDs(Data,times,colTime,colIDs)
        idWL    = self.findIDs(Data,wavelengths,colWL,colIDs)
        idSpot  = self.findIDs(Data,spot,colSpot,colIDs)
        idPower  = self.findIDs(Data,power,colPower,colIDs)
        ids = np.intersect1d(idT,idF)
        ids = np.intersect1d(ids,idDate)
        ids = np.intersect1d(ids,idTime)
        ids = np.intersect1d(ids,idWL)
        ids = np.intersect1d(ids,idSpot)
        ids = np.intersect1d(ids,idPower)
        
        return(ids)

    def returnIds(self,Data,temperatures,fluences,dates):   
        """This function returns the identifiers that belong to datasets that match the given 
        temperature, fluence, and dates specified in the input. If one quantity is not specified 
        then all data of that quantity are returned.
            
            Parameters
            ----------
            Data : 2D numpy array 
                    numpy array containing a Datamatrix
            temperatures :  1D numpy array
                    temperatures in K that are to be included in the selection
            fluences : 1D numpy array
                    fluences in mJ that are to be included in the selection 
            dates : 1D numpy array
                    dates that are to be included in the selection    
            
            
            Returns
            ------
            ids :  1D numpy array
                    A numpy array with the unique identifiers where all input requirements are met
               
            Example
            ------- 
            >>> DataArray = np.array([ 0,9,9],
                                     [ 1,9,9],
                                     [ 2,8,8]]
            >>> ids = returnIDs(Data,[250,200],[3,6],[20140801])
            will return for example [0,1] """

        colIDs = 0
        colDate = 1

        colT = 3
        colF = 4

        
        idT = self.findIDs(Data,temperatures,colT,colIDs)
        idF = self.findIDs(Data,fluences,colF,colIDs)
        idDate = self.findIDs(Data,dates,colDate,colIDs)

        ids = np.intersect1d(idT,idF)
        ids = np.intersect1d(ids,idDate)
        
        return(ids)
        
    def findIDs(self,DataArray,keys,colKeys,colIDs):
        """This function returns the unique identifiers that belong to datasets 
        where one key value is found in the specified key column. If the keys are
        an empty vector all unique Ids in colIDs are returned
        
Parameters
----------
DataArray : 2D numpy array 
        numpy array containing a Datamatrix
keys :  1D numpy array
        numpy array containing the keys for which the routine looks in DataArray[:,colKeys]
colKeys :  integer
        column number of the Datamatrix in which the routine looks for the keys
colIDs :   integer
        column number of the Datamatrix from which the routine takes the IDs

Returns
------
IDs :  1D numpy array
        A numpy array with the unique identifiers where the values of in DataArray[:,colKeys] match 
        one of the keys
   
Example
------- 
>>> DataArray = np.array([ 0,9,9],
                         [ 1,9,9],
                         [ 2,8,8]]
>>> ids = findIDs(DataArray,[9],2,0)
will return [0,1]"""
        if np.size(keys) !=0:
            ids = np.array([])
            for i in keys:
                ids = np.append(ids,np.unique(DataArray[DataArray[:,colKeys] == i,colIDs]))
        else:
            ids = np.unique(DataArray[:,colIDs])
        return ids
        
    def finderA(self,array,key):
        """This little function returns the index of the array where the array value is closest to the key   
        
Parameters
----------
array : 1D numpy array 
        numpy array with values
key :   float value
        The value that one looks for in the array

Returns
------
index :     integer 
            index of the array value closest to the key    
   
Example
------- 
>>> index = finderA(np.array([1,2,3]),3)
will return 2"""
        index = (np.abs(array-key)).argmin()
        return index
        
    def smooth(self,x,window_len,window):
        """smooth the data using a window with requested size.
    
     This method is based on the convolution of a scaled window with the signal.
     The signal is prepared by introducing reflected copies of the signal 
     (with the window size) in both ends so that transient parts are minimized
     in the begining and end part of the output signal.
     
Parameters
----------
x :     1D numpy array
        the input signal numpy array
window_len : odd integer
             the dimension of the smoothing window
window :    string 
            the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing. 
            
Returns
------
result : numpy Array of the smoothed signal      
   
Example
------- 
>>> t=linspace(-2,2,0.1)
>>> x=sin(t)+randn(len(t))*0.1
>>> y=smooth(x)   
  
see also:      
numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
scipy.signal.lfilter  
TODO: the window parameter could be the window itself if an array instead of a string """    
        if x.ndim != 1:
            raise (ValueError, "smooth only accepts 1 dimension arrays.") 
        if x.size < window_len:
            raise (ValueError, "Input vector needs to be bigger than window size.")
         
         
        if window_len<3:
            return x
         
         
        if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
            raise(ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'" ) 
        s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
         #print(len(s))
        if window == 'flat': #moving average
            w=np.ones(window_len,'d')
        else:
            w=eval('np.'+window+'(window_len)')
         
        y=np.convolve(w/w.sum(),s,mode='valid')
        return y[(window_len/2-1):-(window_len/2)]
    
    def StepExpDecaySinus(self, x, x0, tau,A, omega,  Offset,tau2):
        """This method defines the fitfunction  
         f(x) = 1 + Step(x-x0)*[A*exp(-(x-x0/tau))*(sin(omega*t))^2+Offset*exp(-(x-x0/tau2))]"""         
        StepExpDecaySinus = 0.5 *(np.sign(x-x0)+1 )*(A*np.exp(-((x-x0)/tau))*(np.sin(omega*(x-x0)))**2  + Offset*np.exp(-((x-x0)/tau2)) )
        return StepExpDecaySinus
        
    def regridArray(self,Array,Column,Grid,RemoveColumns):
        """ 
        This function returns a new version of a numpy Array where data are 
        averaged within boxes of a given grid. The new array contains the average
        value of the datapoints within each gridbox. A list of Columns that are removed
        in the new array or left untouched should be provided. If the datapoints coincide
        with the gridpoints this function reduces to a simple average. 
        Columns with the corresponding standard deviations are appended as columns
        to the end of the array in the same order as the average value columns. 
        
Parameters
----------
Array : numpy Array
        numpy Array that contains separate data in columns
Column : integer
        integer number of the column to which the array will be regridded
Grid:  1D numpy Array
        Array that contains the gridpoints
        select = Grid(i)<= Array(:,Column) & Array(:,Column)< Grid(i+1)
RemoveColumns : List of integers
        List of Columns that will be removed in the outpout

Returns
------
result : numpy Array
    a numpy array that contains the regridded Array with specified columns untouched and removed. 
    The end of the array contains the standard deviation vectors. 
    
Example
-------

>>> AllDataAveraged =regridArray(AllDataNormalized,0,TimeGrid,[0,2,5,6])

This returns a numpy array that is regridded according to column 1 and the specified TimeGrid. 
Columns 0,2,5 and 6 are removed in the output. Only columns 1,3 and 4 remain and they will be column
0,1,2 of the new array. Column 3,4 and 5 five will contain the standard deviation of the values in column
0,1,2 respectively.
"""


        time = Array[:,Column]                                      # This is the column according to which I regrid
        ArrayTemp = np.delete(Array,RemoveColumns,1)                # This temparry array only contains the remaining columns
        ColsNew = np.size(ArrayTemp,1)
        ArrayNew = np.zeros((np.size(Grid)-1,ColsNew*2))                 # This is the new Array that will eventually be returned
        
        counter = 0    
        for t in range(np.size(Grid)-1):
            select1 = time>= Grid[t]
            select2 = time< Grid[t+1]
            select = select1 & select2
            if np.sum(select) <= 0.:
                ArrayNew = np.delete(ArrayNew,t - counter,0)
                counter += 1
            else:
                for c in range(ColsNew):
                    ArrayNew[t-counter,c] = np.mean(ArrayTemp[select,c])
                    ArrayNew[t-counter,c+ColsNew] = np.std(ArrayTemp[select,c])
        return(ArrayNew)

    
    def radians(self,degrees):
        """ 
        Convert degrees to radians
        
Parameters
----------
degrees: 1D numpy Array or Float
         input value in degrees
Returns
------
radians : numpy Array
        a numpy array with the corresponding radian values of the degrees input
Example
-------

>>> rad =radians(180)
returns the numerical value of pi

"""
        radians = np.pi/180*degrees
        return(radians)

    def degrees(self,radians):
        """ 
        Converts radians to degrees
        
Parameters
----------
radians: 1D numpy Array or Float
         input value in radians
Returns
------
degrees : numpy Array
        a numpy array with the corresponding degree values of the radian input
Example
-------

>>> deg =degrees(np.pi)
returns 180

"""
        degrees = 180/np.pi*radians
        return(degrees)

        
    def normalizeColumnsToValue(self,Array,Value,ColumnsToNormalize,*ReturnRelativeChange):
        """ 
        This function returns a new version of an numpy Array where the specified 
        columns are devided by a the specified value. It has the possibility to 
        return the relativeChangeValues instead. 
        
Parameters
----------
Array : numpy Array
        numpy Array that contains separate data in columns
Value: float
        Value by which the columns are divided
ColumnsToNormalize : List of Integers  
       List of integers that contains the columns which should be normalized to the 
       specified column
ReturnRelativeChange: Bolean
        True False Value if the data should return a relative change. 
        True Case: Data will be normalized and 

Returns
------
result : numpy Array
    a numpy array that contains the same data but the specified columns are
    normalized to a value
    
Example
-------
Suppose your Array looks like::

    1.0     1.0    2.0  1.0
    0.0     0.0    2.0  0.0
    1.0     1.0    1.0  1.0 
    1.0     1.0    1.0  1.0 

 
And you apply:

>>> ArrayNew = NormalizeColumnsToValue(Array,2,[0,3])

ArrayNew will look contain only the following data::

    0.5     1.0    2.0  0.5
    0.0     0.0    2.0  0.0
    0.5     1.0    1.0  0.5 
    0.5     1.0    1.0  0.5 
    
"""
        ArrayNew = Array
        for i in ColumnsToNormalize:
            if ReturnRelativeChange:
                ArrayNew[:,i] = Array[:,i]/Value - 1
            else:
                ArrayNew[:,i] = Array[:,i]/Value
                
        return(ArrayNew)
    

    def normalizeColumnsToColumn(self,Array,Column,Offset,ColumnsToNormalize):
        """ 
        This function returns a new version of an numpy Array where the specified 
        columns are normalized to the values in a specified column minus a fixed
        Offset.
        
Parameters
----------
Array : numpy Array
        numpy Array that contains separate data in columns
Column : integer
        integer number of the column to which the array will be normalized to
Offset: float or integer
        Newvalue = oldvalue/(columnvalue - Offset)
ColumnsToNormalize : List of Integers  
       List of integers that contains the columns which should be normalized to the 
       specified column

Returns
------
result : numpy Array
    a numpy array that contains the same data which are normalized to a specified column
    
Example
-------
Suppose your Array looks like::

    1.0     1.0    2.0  1.0
    0.0     0.0    2.0  0.0
    1.0     1.0    1.0  1.0 
    1.0     1.0    1.0  1.0 
    1.0     1.0    2    1.0 
 
And you apply:

>>> ArrayNew = NormalizeColumnsToColumn(Array,2,[0,1,3],0)

ArrayNew will look contain only the following data::

    0.5     0.5    2.0  0.5
    0.0     0.0    2.0  0.0
    2.0     2.0    0.5  2.0 
    2.0     2.0    0.5  2.0 
    0.5     0.5    2    0.5 
    
"""
        ArrayNew = Array
        for i in ColumnsToNormalize:
            ArrayNew[:,i] = Array[:,i]/(Array[:,Column]-Offset)
        return(ArrayNew)
        
    def removeBadData(self,Array,Column,Threshold):
        """ 
        This function returns a new version of an numpy Array that contains only 
        the rows where the values in the specified column are larger or equal than
        the specified threshold
Parameters
----------
Array : numpy Array
        numpy Array that contains separate data in columns
Column : integer
        integer number of the column of the Array which are compared to the thresholdvalue
Threshold : float  
       numerical threshold value. If the data are larger than this threshold than they are kept
       otherwise the entire row of the matrix is removed

Returns
------
result : numpy Array
    a numpy array that contains the same data as the input array except for the 
    rows where the value is smaller than the specified threshold
    
Example
-------
Suppose your Array looks like::

    1.0     1.0    2.0  1.0
    0.0     0.0    0.0  0.0
    1.0     1.0    1.0  1.0 
    1.0     1.0    1.0  1.0 
    1.0     1.0    0.05 1.0 
 
And you apply:

>>> ArrayNew = removeBadData(Array,2,0.1)

ArrayNew will look contain only the following data::

    1.0     1.0    2.0  1.0
    1.0     1.0    1.0  1.0 
    1.0     1.0    1.0  1.0 
    
"""
        ArrayNew = np.zeros((np.sum(Array[:,Column] >= Threshold),len(Array[0,:])))
        counter = 0
        for j in range(0,len(Array[:,0])):
            if Array[j,Column] >= Threshold:
                ArrayNew[counter,:] = Array[j,:]
                counter += 1
        return(ArrayNew)
        
    def timeStamp(self,DateStr,TimeStr):
        """ 
        This function returns a timeStamp (seconds since "1970-1-1 00:00:00") 
        for a date given in the commonly used format in the data aquisition in 
        the UDKM Group. The timestamps can be used as unique identifier for datasets
        and they can also be compared
Parameters
----------
Datestr : string
        String that contains the date in the format: "yyyymmdd""
TimeStr : string
        String that contains the time in the format "hhmmss""
        If the TimeString is to short it is automatically extended with leading zeros

Returns
------
result : integer
    An integer value of the timestamp that corresponds to the date. It is the number of 
seconds since the epoch "1970-1-1 00:00:00"

Example
-------
>>> TimeStamp("20150515","175850")

returns the value 1431705530
    
"""
        if np.size(DateStr) ==1:
            DateStr = str((int(DateStr)))
            TimeStr = str(int(TimeStr))
            TimeStr = (6-len(TimeStr))*'0'+TimeStr
            temp = datetime.datetime(int(DateStr[0:4]),int(DateStr[4:6]),int(DateStr[6:8]),int(TimeStr[0:2]),int(TimeStr[2:4]),int(TimeStr[4:6]))
            TimeStamp = int(temp.timestamp())
        else:
            TimeStamp = np.zeros(np.size(DateStr))
            for i in range(np.size(TimeStamp)):                        
                TimeStri = (6-len(str(int(TimeStr[i]))))*'0'+str(int(TimeStr[i]))
                DateStri = str(int(DateStr[i]))
                temp = datetime.datetime(int(DateStri[0:4]),int(DateStri[4:6]),int(DateStri[6:8]),int(TimeStri[0:2]),int(TimeStri[2:4]),int(TimeStri[4:6]))
                TimeStamp[i] = int(temp.timestamp())    
        return TimeStamp

    def dateFromTimeStamp(self,TimeStamp):
        """ 
        This function returns the DateString and a TimeString from a timestamp 
        (seconds since "1970-1-1 00:00:00"). in the common format yymmdd and hhmmss
Parameters
----------
Timestamp : integer
        integer value of the seconds since 1970-1-1 00:00:00

Returns
------
result = DateStr,TimeStr

Datestr : string
        String that contains the date in the format: "yyyymmdd""
TimeStr : string
        String that contains the time in the format "hhmmss""
        If the TimeString is to short it is automatically extended with leading zeros

Example
-------
>>> Datestr,TimeStr = dateFromTimeStamp(1431705530)

returns Datestr = "20150515" and TimeStr = "175850"
    
"""                 
        return datetime.datetime.fromtimestamp(TimeStamp).strftime('%Y%m%d'),datetime.datetime.fromtimestamp(TimeStamp).strftime('%H%M%S')   

    def writeLinesToFile(self,FileName,Header,*Data):
        """ 
        This procedure writes an arbitrary number of 1D arrays into a single file as lines with the specified header as first line. 
        Tabs are used as separation sign. New Lines indicate new arrays
Parameters
----------
Filename : string
        Path to the filename where the file will be stored
Header : string
        First line of the datafile that will be written
data : 1D arrays separated by ","  
       1D arrays of the same length that will be written to the file as lines
Example
-------
>>> a=np.ones(4)
>>> b=np.zeros(4)
>>> c=np.ones(4) 
>>> writeLinesToFile('EraseMe.txt','Time \t FirstPeak \t SecondPeak',a,b,c)
Will write the following content to the file 'EraseMe.txt' ::

    # Time  FirstPeak   SecondPeak
    1.0     1.0    1.0  1.0
    0.0     0.0    0.0  0.0
    1.0     1.0    1.0  1.0
    
"""
    
        f = open(FileName, 'w')
        f.write('# ') 
        f.write(Header+'\n')
        for i in range(len(Data)):
            self.write1DArrayToFile(f,Data[i])
        f.close

    def writeHeaderToFile(self,FileName,Header):
        """ 
         This procedure writes a string as it is to a specified file. 
Parameters
----------
FileName : string
        Path to the filename where the file will be stored
Header : string
        First line of the datafile that will be written

Example
-------
>>> h.writeHeaderToFile("test.dat","#temp \t x0 \t sigma")

Creates a new file and a line in the file
The created file looks like:: 
 
    #temp   x0  sigma


"""
        f = open(FileName, 'w') 
        f.write(Header+'\n')
        f.close 

    def writeListToFile(self,FileName,Header,List):
        """ 
         This procedure writes a 2D Array or List to a unopened file. 
         rows are written as lines. The elements are separated by tabs. Rows are 
         separated by linebreaks. The first line is preceded by a #
Parameters
----------
FileName : string
        Path to the filename where the file will be stored
Header : string
        First line of the datafile that will be written
List : list, or 2D numpy array
        List which is written to the File

Example
-------
>>> h.WriteListToFile("test.dat","temp \t x0 \t sigma", np.array([[170, 12.3, 1.34],[ 180, 11.53, 1.4]]))

Creates a new file and writes the full list in your file (plus header)    
The created file looks like:: 
 
    # temp   x0  sigma
    170     12.3    1.34 
    180     11.53    1.4

"""
        f = open(FileName, 'w')
        f.write('# ') 
        f.write(Header+'\n')
        self.write2DArrayToFile(f,List)
        f.close 
            
    def writeSingleLineToFile(self,FileName,Header,Line):
        """ 
        This procedure writes a List or a 1D-numpy array as single line to a file 
Parameters
----------
Filename : string
        relative (or absolute) path to the file to which the line will written 
Header : string
        First line of the datafile that will be written. It is preceded by a '#'        
Line : list, 1D-numpy-array
        the elements of the line will be appended to the specified file columns
Example
-------
>>> writeSingleLineToFile('test.dat',' Number [ ]\t  Fitresults ', [1,1.2, 12, 14, "a"])

Creates a file that will look like:: 

    # Number [ ]    Fitresults
    1   1.2     12  14  a

"""
        f = open(FileName, 'w')
        f.write('# ') 
        f.write(Header+'\n')
        self.write1DArrayToFile(f,Line)
        f.close 
    def appendSingleLineToFile(self,FileName,Line):
        """
        This procedure appends a List or a numpy array as single line to a file 
Parameters
----------
Filename : string
        relative (or absolute) path to the file to which the line will be appended 
Line : list, 1D-numpy-array
        the elements of the line will be appended to the specified file columns
Example
-------
Suppose you have the file List.dat that looks like::

    # Time  FirstPeak   SecondPeak
    1.0     0.0    1.0 
    1.0     0.0    1.0 

and you call:    

>>> appendSingleLineToFile([1,2,3.2],'List.dat')

then it will look like:: 

    # Time  FirstPeak   SecondPeak
    1.0     0.0    1.0 
    1.0     0.0    1.0 
    1       2      3.2 
    
"""
        f = open(FileName, 'a')
        self.write1DArrayToFile(f,Line)
        f.close 
        
    def appendStringLineToFile(self,FileName,Line):
        """
        This procedure appends a string as single line to a file 
        Parameters
        ----------
        Filename : string
                relative (or absolute) path to the file to which the line will be appended 
        Line : string
                String
            
        """
        f = open(FileName, 'a')
        f.write(Line + '\n')
        f.close     

    def write1DArrayToFile(self,f,Array):
        """ 
         This procedure writes a 1D Array or List to an already opened file with the 
         identifier f as a line. The elements are separated by tabs.
Parameters
----------
f : file identifier 
       file identifier of the already opened file
Array : 1D-numpy-Array or List
        Data Line 

Example
-------
This is how you typically would use the function:

>>> f = open("test.dat", 'a')
>>> Line = [1,2,3]
>>> write1DArrayToFile(f,Line)
>>> f.close

This appends the Line::
 
    1   2   3
    
to the file "test.dat"
"""

        for i in range(0, len(Array)):
                    if i != (len(Array) - 1):
                        f.write(str(Array[i]) + '\t')
                    else:
                        f.write(str(Array[i]) + '\n')
    def write2DArrayToFile(self,f,Array):        
        """ 
         This procedure writes a 2D Array or List to an already opened file with 
         the identifier f. The elements are separated by tabs and new rows are 
         indicated by line breaks.
Parameters
----------
f : file identifier 
       file identifier of the already opened file
Array : 2D-numpy-Array or List
        Data Line 

Example
-------
This is how you typically would use the function:

>>> f = open("test.dat", 'a')
>>> Array = [1,2,3;4,5,6]
>>> write2DArrayToFile(f,Array)
>>> f.close

This appends the lines::
 
    1   2   3
    4   5   6
    
to the file "test.dat"
"""
        for i in range(0, len(Array)):
            for j in range(0, len(Array[i,:])):
                if j != (len(Array[i,:]) - 1):
                    f.write(str(Array[i,j]) + '\t')
                else:
                    f.write(str(Array[i,j]) + '\n') 
                    
    def writeColumnsToFile(self,Filename, Header, *data):
        """
        This procedure writes an arbitrary number of 1D arrays into a single file as columns with the specified header as first line. 
        Tabs are used as separation sign
Parameters
----------
Filename : string
        Path to the filename where the file will be stored
Header : string
        First line of the datafile that will be written
data : 1D arrays separated by ","  
       1D arrays of the same length that will be written to the file as columns
Example
-------
>>> a=np.ones(4)
>>> b=np.zeros(4)
>>> c=np.ones(4) 
>>> WriteColumnsToFile('EraseMe.txt','Time \t FirstPeak \t SecondPeak',a,b,c)
Will write the following content to the file 'EraseMe.txt' ::

    # Time  FirstPeak   SecondPeak
    1.0     0.0    1.0 
    1.0     0.0    1.0 
    1.0     0.0    1.0 
    1.0     0.0    1.0 
"""
        f = open(Filename, 'w')
        f.write('# ') 
        f.write(Header+'\n')
        for j in range(len(data[0])):
            for i in range(len(data)):
                if i != (len(data) - 1):
                    f.write(str(data[i][j]) + '\t')
                else:
                    f.write(str(data[i][j]) + '\n')
        f.close

    
    def unzip(self,Archive,Target):
        """
        This procedure unzips a specified archive "Archive.zip" and stores its contents to the specified to the folder "target". It only acts if a folder named "Target" does not already exist. 
Parameters
----------
Archive : string
        Path to the .zip archive relative to the working directory or absolute
Target : string
        Path to the folder into which the contents of archive will be unpacked

Example        
-------
>>> unzip("a","b")
Extracts the contents of the archive archive "a" + ".zip" to a folder "b" if
b does not alread exist. 
>>> unzip("a","a")
Extracts an archive into the same locaction of the file and keeps "a" as the 
name of the target folder if the folder a does not already exist.
"""
        if (os.path.isdir(Target) == False):
            z = zipfile.ZipFile(Archive +'.zip')
            z.extractall(Target)
                        
    def makeFolder(self,PathToFolder):
        """
        This procedure creates an empty folder if the target Folder does not already exist. 
Parameters
----------
PathToFolder : string
    Path to the newly created folder

Example
-------
>>> makeFolder("SubFolder/NewFolderName")
Creates a new folder with the name "NewFolder" in the directory "SubFolder" 
relative to the working path but only if this folder does not already exist.
"""                
        if not os.path.exists(PathToFolder):
            os.makedirs(PathToFolder)

    def calcThetaAxis(self,qzAxis):
        """ This function calculates the theta axis in degrees when it is given the qzAxis assuming Cu-Kalpha radiation
        Parameters
        ----------
        qzAxis : 1d numpy array
            qzAxis in Angström
        
        Returns
        ----------
        thetaAxis : 1d numpy array with same length as qzAxis
        
        Example
        -------
        >>> calcThetaAxis(qzAxis)
        Will return a the theta axis that would be necessary for a in a theta-2theta rocking scan in degrees     
        """
        wl = 1.392 # Cu-Kalpha1 wavelength in Angström
        thetaAxis = np.arcsin(qzAxis*wl/(4*np.pi))
        thetaAxis = self.degrees(thetaAxis)
        return thetaAxis

    def calcQxQz(self,omega,theta,k):
        """ Calculates the corresponding values qx and qz for an input pair omega, theta and a wavevector k. 
        The calculation is based on equation (4) of  Ultrafast reciprocal-space mapping with a convergent beam , 2013, Journal of Applied Crystallography, 46, 5, 1372-1377 
        omega is the incident angle of the X-Rays onto the sample in radians, theta is the detector angle at which the photons are detected in radians """
        
        qx = k*(np.cos(theta) - np.cos(omega))
        qz = k*(np.sin(theta) + np.sin(omega))
        return qx,qz
    def calcQzQx(self,omega,theta,k):
        """ Calculates the corresponding values qx and qz for an input pair omega, theta and a wavevector k. 
        The calculation is based on equation (4) of  Ultrafast reciprocal-space mapping with a convergent beam , 2013, Journal of Applied Crystallography, 46, 5, 1372-1377 
        omega is the incident angle of the X-Rays onto the sample in radians, theta is the detector angle at which the photons are detected in radians """
        
        qx = k*(np.cos(theta) - np.cos(omega))
        qz = k*(np.sin(theta) + np.sin(omega))
        return qz,qx
    
    def calcJacobiDet(self,omega,theta,k):
        """This calculates the Jacobian of the coordinate transformation given by equation (4) of  Ultrafast reciprocal-space mapping with a convergent beam , 2013, Journal of Applied Crystallography, 46, 5, 1372-1377 """  
        Jacobi = k**2*np.abs(np.sin(omega)*np.cos(theta)+np.sin(theta)*np.cos(omega))
        return Jacobi
    
    def timestring(self,time):
        """ Returns a timestring made of 6 digits for the inserted time adding leading zeros if necessary. works also with arrays."""
                
        if np.size(time) == 1:
            time = int(time)
            result = (6-len(str(time)))*'0'+str(time)
        else:
            
            result = np.zeros(np.size(time))
            for i in range(np.size(time)):
                result[i] = (6-len(str(int(time[i]))))*'0'+str(int(time[i]))
        return result
    
