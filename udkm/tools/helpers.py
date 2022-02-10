# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:29:02 2021

@author: Aleks
"""

import numpy      as np
import os         as os
import zipfile    as zipfile


teststring = "Successfully loaded udkm.tools.helpers"

    
PXSwl   = 1.5418           #Cu K-alpha Wavelength in [Ang]
PXSk    = 2*np.pi/PXSwl    #absolute value of the incident k-vector in [1/Ang]


def calcMoments(xAxis,yValues):
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
    delta   = calcGridBoxes(xAxis)[0]
    integral = np.sum(yValues*delta)
    return COM,STD,integral
    
def calcGridBoxes(grid):
     """ calculates the size of a grid cell and the left and right boundaries of each gridcell in a monotonous but
     possibly nonlinear grid
    
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
        
def radians(degrees):
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

def degrees(radians):
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

def finderA(array,key):
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

def relChange(dataArray,fixpoint):
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
        
def setROI1D(xAxis,values,xMin,xMax):
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


def setROI2D(xAxis,yAxis,Matrix,xMin,xMax,yMin,yMax):
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



def smooth(x,window_len,window):
    """
    smooth the data using a window with requested size.
   
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
    the window parameter could be the window itself as an array instead of a string """
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
    return y[int(window_len/2-1):-int(window_len/2)]



def writeListToFile(FileName,Header,List):
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
    write2DArrayToFile(f,List)
    f.close 


def writeSingleLineToFile(FileName,Header,Line):
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
    write1DArrayToFile(f,Line)
    f.close 

def writeColumnsToFile(Filename, Header, *data):
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
    

def write1DArrayToFile(f,Array):
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

    
def write2DArrayToFile(f,Array):        
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


def appendSingleLineToFile(FileName,Line):
    
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
    write1DArrayToFile(f,Line)
    f.close 

def unzip(Archive,Target):
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
            

def makeFolder(PathToFolder):
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

def timestring(time):
    """ Returns a timestring made of 6 digits for the inserted time adding leading zeros if necessary. works also with arrays."""
            
    if np.size(time) == 1:
        time = int(time)
        result = (6-len(str(time)))*'0'+str(time)
    else:
        result = np.zeros(np.size(time))
        for i in range(np.size(time)):
            result[i] = (6-len(str(int(time[i]))))*'0'+str(int(time[i]))
    return result


def imagestring(time):
    """ Returns a timestring made of 5 digits for the inserted time adding leading zeros if necessary. works also with arrays."""
            
    if np.size(time) == 1:
        time = int(time)
        result = (5-len(str(time)))*'0'+str(time)
    else:
        result = np.zeros(np.size(time))
        for i in range(np.size(time)):
            result[i] = (5-len(str(int(time[i]))))*'0'+str(int(time[i]))
    return result