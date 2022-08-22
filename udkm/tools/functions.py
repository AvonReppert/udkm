# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 21:29:02 2021
@author: Aleks
"""

import numpy as np
import os as os
import pandas as pd
import pickle as pickle
import shutil
import zipfile as zipfile

teststring = "Successfully loaded udkm.tools.functions"


def fft(x_data, y_data):
    """ returns fft of the y_data and frequency  axis"""
    length = len(x_data)

    if length != len(y_data):
        print('x-axis and first dimension of the data matrix do not have identical size!')
        return

    dx = np.mean(np.diff(x_data))
    if length % 2 == 0:
        length_2 = length
    else:
        length_2 = length-1
        print('last data point omitted!')

    fft_size = length_2/2+1
    ft_data = np.abs(2*np.fft.rfft(y_data[0:length_2])/length_2)
    ft_x_axis = 0.5/dx*np.linspace(0, 1, int(fft_size))
    return ft_x_axis, ft_data


def calc_fluence(power, fwhm_x, fwhm_y, angle, rep_rate):
    """returns the fluence
    Parameters
    ----------
    power : float
        incident power in mW
    fwhm_x : float
        Full Width at Half Maximum of the (gaussian) pump beam in x-direction in microns
    fwhm_y : float
        Full Width at Half Maximum of the (gaussian) pump beam in y-direction in microns
    angle : float
        angle of incidence of the pump beam in degree relative to surface normal
    rep_rate : float
        repetition rate of the used laser

    Returns
    ------
        fluence : float
          calculated fluence in mJ/cm^2

    Example
    ------
        >>> calc_fluence(50,500*1e-4,400*1e-4,45,1000) """
    # Umrechung von Grad in Bogenmaß
    angle_rad = np.radians(angle)
    # Berechnung der Fluenz
    x0 = fwhm_x/(2*np.sqrt(np.log(2)))*1e-4
    y0 = fwhm_y/(2*np.sqrt(np.log(2)))*1e-4
    area = np.pi*x0*y0
    fluence = power*np.cos(angle_rad)/(rep_rate*area)  # in mJ/cm^2
    return np.round(fluence, 2)


def calc_moments(x_axis, y_values):
    """ calculates the center of mass, standard deviation and integral of a given distribution

    Parameters
    ----------
        x_axis : 1D numpy array
            numpy array containing the x Axis
        y_values : 1D numpy array
            numpy array containing the according y Values

    Returns
    ------
    in that order
        com : float
            xValue of the Center of Mass of the data

        std : float
            xValue for the standard deviation of the data around the center of mass

        integral :
            integral of the data

    Example
    -------
        >>> com,std,integral = calcMoments([1,2,3],[1,1,1])
            sould give a com of 2, a std of 1 and an integral of 3 """

    com = np.average(x_axis, axis=0, weights=y_values)
    std = np.sqrt(np.average((x_axis-com)**2, weights=y_values))
    delta = calc_grid_boxes(x_axis)[0]
    integral = np.sum(y_values*delta)
    return com, std, integral


def calc_grid_boxes(grid):
    """ calculates the size of a grid cell and the left and right boundaries of each gridcell
    in a monotonous but possibly nonlinear grid

   Parameters
   ----------
       vector : 1D numpy array
                numpy array containing a monotonous grid with points
   Returns
   ------
   in that order
       delta : 1D numpy array of same length as vector
               the distance to the left and right neighbor devided by 2
       left : 1D numpy array of same length as vector
           left boundaries of the grid cell
       right : 1D numpy array of same length as vector
           right boundaries of the grid cell
   Example
   -------
       >>> delta,right,left = calc_grid_boxes([0,1,2,4])
           (array([ 1. ,  1. ,  1.5,  2. ]),
            array([-0.5,  0.5,  1.5,  3. ]),
            array([ 0.5,  1.5,  3. ,  5. ]))"""

    delta = np.zeros(len(grid))
    right = np.zeros(len(grid))
    left = np.zeros(len(grid))
    for n in range(len(grid)):
        if (n == 0):
            delta[n] = grid[n+1] - grid[n]
            left[n] = grid[n]-delta[n]/2
            right[n] = np.mean([grid[n], grid[n+1]])
        elif n == (len(grid)-1):
            delta[n] = grid[n] - grid[n-1]
            left[n] = np.mean([grid[n], grid[n-1]])
            right[n] = grid[n] + delta[n]/2
        else:
            left[n] = np.mean([grid[n], grid[n-1]])
            right[n] = np.mean([grid[n], grid[n+1]])
            delta[n] = np.abs(right[n]-left[n])
    return delta, left, right


def find(array, key):
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
    >>> index = find(np.array([1,2,3]),3)
    will return 2"""
    index = (np.abs(array-key)).argmin()
    return index


def rel_change(data_array, fixpoint):
    """returns relative change of a measured quantity relative to a given value

    Parameters
    ----------
    data_array : 1D numpy array
        array containing the data

    fixpoint : float
        Quantity to which the relative change is calculated

    Returns
    ------
        rel_change_array : 1D numpy array with same length as data_array
        contains (data_array-fixpoint)/fixpoint
    Example
    -------
        >>> change = relChange(cAxisDy,cAxisDy[T==300])"""
    rel_change_array = (data_array-fixpoint)/fixpoint
    return(rel_change_array)


def set_roi_1d(x_axis, values, x_min, x_max):
    """ selects a rectangular region of intrest roi from a 2D matrix based on the
    passed boundaries x_min,x_max, y_min and y_max. x stands for the columns of the
    matrix, y for the rows

    Parameters
    ----------
        x_axis: 1D numpy arrays
            numpy array containing the x grid
        Values : 1D numpy array same length as x_axis

        x_min,x_max: inclusive Boundaries for the roi

    Returns
    ------
    in that order
        x_roi : 1D numpy array slice of x_axis between x_min and x_max

        roi : 1D numpy array of same length as vector


    Example
    -------
        >>> qz_cut,roi = setroimatrix(qzgrid,RockingQz,2.1,2.2)"""

    select_x = np.logical_and(x_axis >= x_min, x_axis <= x_max)
    x_roi = x_axis[select_x]
    roi = values[select_x]
    return x_roi, roi


def set_roi_2d(x_axis, y_axis, matrix, x_min, x_max, y_min, y_max):
    """ selects a rectangular region of intrest roi from a 2D matrix based on the
    passed boundaries x_min,x_max, y_min and y_max. x stands for the columns of the
    matrix, y for the rows

    Parameters
    ----------
        x_axis, y_axis : 1D numpy arrays
            numpy arrays containing the x and y grid respectively
        matrix : 2D numpy array
            2D array with the shape (len(y_axis),len(x_axis))
        x_min,x_max,y_min,y_max : inclusive Boundaries for the roi

    Returns
    ------
    in that order
        x_roi : 1D numpy array slice of x_axis between x_min and x_max

        y_roi : 1D numpy array slice of y_axis between y_min and y_max

        roi : 2D numpy array of same length as vector

        x_integral : 1D numpy arrays with the same length as x_roi
            array containing the sum of roi over the y direction

        y_integral : 1D numpy arrays with the same length as y_roi
            array containing the sum of roi over the x direction

    Example
    -------
        >>> qz_cut,qx_cut,roi,x_integral,y_integral = setroimatrix(qzgrid,qxgrid,rsm_q,2.1,2.2,-0.5,0.5)"""

    select_x = np.logical_and(x_axis >= x_min, x_axis <= x_max)
    select_y = np.logical_and(y_axis >= y_min, y_axis <= y_max)

    x_roi = x_axis[select_x]
    y_roi = y_axis[select_y]
    roi = matrix[select_y, :]
    roi = roi[:, select_x]
    x_integral = np.sum(roi, 0)
    y_integral = np.sum(roi, 1)
    return x_roi, y_roi, roi, x_integral, y_integral


def regrid_measurement(x, y, x_grid):
    signal_matrix = np.zeros((len(x), 2))
    signal_matrix[:, 0] = x
    signal_matrix[:, 1] = y
    signal_matrix_regridded = regrid_array(signal_matrix, 0, x_grid, [])
    x_regridded = signal_matrix_regridded[:, 0]
    y_regridded = signal_matrix_regridded[:, 1]
    std_x = signal_matrix_regridded[:, 2]
    std_y = signal_matrix_regridded[:, 3]
    return x_regridded, y_regridded, std_x, std_y


def regrid_array(array, column, grid, remove_columns):
    """
    This function returns a new version of a numpy array where data are
    averaged within boxes of a given grid. The new array contains the average
    value of the datapoints within each gridbox. A list of columns that are removed
    in the new array or left untouched should be provided. If the datapoints coincide
    with the gridpoints this function reduces to a simple average.
    columns with the corresponding standard deviations are appended as columns
    to the end of the array in the same order as the average value columns.

    Parameters
    ----------
    array : numpy array
            numpy array that contains separate data in columns
    column : integer
            integer number of the column to which the array will be regridded
    grid:  1D numpy array
            array that contains the gridpoints
            select = grid(i)<= array(:,column) & array(:,column)< grid(i+1)
    remove_columns : list of integers
            list of columns that will be removed in the outpout

    Returns
    ------
    result : numpy array
        a numpy array that contains the regridded array with specified columns untouched and removed.
        The end of the array contains the standard deviation vectors.

    Example
    -------

    >>> all_data_avaraged =regrid_array(all_data_normalized,0,time_grid,[0,2,5,6])

    This returns a numpy array that is regridded according to column 1 and the specified time_grid.
    columns 0,2,5 and 6 are removed in the output. Only columns 1,3 and 4 remain and they will be column
    0,1,2 of the new array. column 3,4 and 5 five will contain the standard deviation of the values in column
    0,1,2 respectively.
    """

    time = array[:, column]  # This is the column according to which I regrid
    array_temp = np.delete(array, remove_columns, 1)  # This temporary array only contains the remaining columns
    cols_new = np.size(array_temp, 1)
    array_new = np.zeros((np.size(grid)-1, cols_new*2))  # This is the new array that will eventually be returned

    counter = 0
    for t in range(np.size(grid)-1):
        select_1 = time >= grid[t]
        select_2 = time < grid[t+1]
        select = select_1 & select_2
        if np.sum(select) <= 0.:
            array_new = np.delete(array_new, t - counter, 0)
            counter += 1
        else:
            for c in range(cols_new):
                array_new[t-counter, c] = np.mean(array_temp[select, c])
                array_new[t-counter, c+cols_new] = np.std(array_temp[select, c])
    return(array_new)


def smooth(x, window_len, window):
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
    result : numpy array of the smoothed signal

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

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise(ValueError, "Window is none of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.'+window+'(window_len)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[int(window_len/2-1):-int(window_len/2)]


def write_list_to_file(file_name, header, list):
    """
    This procedure writes a 2D array or list to an unopened file.
    rows are written as lines. The elements are separated by tabs.
    Rows are separated by linebreaks. The first line is preceded by a "#"
    Parameters
    ----------
    file_name : string
            Path to the file_name where the file will be stored
    header : string
            First line of the datafile that will be written
    list : list, or 2D numpy array
            list which is written to the File

    Example
    -------
    >>> h.write_list_to_file("test.dat","temp \t x0 \t sigma", np.array([[170, 12.3, 1.34],[ 180, 11.53, 1.4]]))

    Creates a new file and writes the full list in your file (plus header)
    The created file looks like:

        # temp   x0  sigma
        170     12.3    1.34
        180     11.53    1.4

    """
    f = open(file_name, 'w')
    f.write('# ')
    f.write(header+'\n')
    write_2d_array_to_file(f, list)
    f.close


def write_single_line_to_file(file_name, header, line):
    """
    This procedure writes a list or a 1D-numpy array as single line to a file
    Parameters
    ----------
    file_name : string
            relative (or absolute) path to the file to which the line will written
    header : string
            First line of the datafile that will be written. It is preceded by a '#'
    line : list, 1D-numpy-array
            the elements of the line will be appended to the specified file columns
    Example
    -------
    >>> write_single_line_to_file('test.dat',' Number [ ]\t  Fitresults ', [1,1.2, 12, 14, "a"])

    Creates a file that will look like:
    # Number [ ]    Fitresults
    1   1.2     12  14  a
    """
    f = open(file_name, 'w')
    f.write('# ')
    f.write(header+'\n')
    write_1d_array_to_file(f, line)
    f.close


def write_columns_to_file(file_name, header, *data):
    """
    This procedure writes an arbitrary number of 1D arrays into a single file as columns with
    the specified header as first line.
    Tabs are used as separation sign
    Parameters
    ----------
    file_name : string
    Path to the file_name where the file will be stored
    header : string
    First line of the datafile that will be written
    data : 1D arrays separated by ","
       1D arrays of the same length that will be written to the file as columns
    Example
    -------
    >>> a=np.ones(4)
    >>> b=np.zeros(4)
    >>> c=np.ones(4)
    >>> write_columns_to_file('EraseMe.txt','Time \t FirstPeak \t SecondPeak',a,b,c)
    Will write the following content to the file 'EraseMe.txt':

    # Time  FirstPeak   SecondPeak
    1.0     0.0    1.0
    1.0     0.0    1.0
    1.0     0.0    1.0
    1.0     0.0    1.0
    """
    f = open(file_name, 'w')
    f.write('# ')
    f.write(header+'\n')
    for j in range(len(data[0])):
        for i in range(len(data)):
            if i != (len(data) - 1):
                f.write(str(data[i][j]) + '\t')
            else:
                f.write(str(data[i][j]) + '\n')
    f.close


def write_1d_array_to_file(f, array):
    """
     This procedure writes a 1D array or list to an already opened file with the
     identifier f as a line. The elements are separated by tabs.

    Parameters
    ----------
    f : file identifier
       file identifier of the already opened file
    array : 1D-numpy-array or list
    Data line

    Example
    -------
    This is how you typically would use the function:

    >>> f = open("test.dat", 'a')
    >>> line = [1,2,3]
    >>> write_1d_array_to_file(f,line)
    >>> f.close

    This appends the line:

    1   2   3

    to the file 'test.dat'

    """

    for i in range(0, len(array)):
        if i != (len(array) - 1):
            f.write(str(array[i]) + '\t')
        else:
            f.write(str(array[i]) + '\n')


def write_2d_array_to_file(f, array):
    """
     This procedure writes a 2D array or list to an already opened file with
     the identifier f. The elements are separated by tabs and new rows are
     indicated by line breaks.

    Parameters
    ----------
    f : file identifier
       file identifier of the already opened file
    array : 2D-numpy-array or list
        data line

    Example
    -------
    This is how you typically would use the function:

    >>> f = open("test.dat", 'a')
    >>> array = [1,2,3;4,5,6]
    >>> write_2d_array_to_file(f,array)
    >>> f.close

    This appends the lines:

    1   2   3
    4   5   6

    to the file 'test.dat'

    """

    for i in range(0, len(array)):
        for j in range(0, len(array[i, :])):
            if j != (len(array[i, :]) - 1):
                f.write(str(array[i, j]) + '\t')
            else:
                f.write(str(array[i, j]) + '\n')


def append_single_line_to_file(file_name, line):
    """
     This procedure appends a list or a numpy array as single line to a file

    Parameters
    ----------
    file_name : string
            relative (or absolute) path to the file to which the line will be appended
    line : list, 1D-numpy-array
            the elements of the line will be appended to the specified file columns
    Example
    -------
    Suppose you have the file list.dat that looks like:

        # Time  FirstPeak   SecondPeak
        1.0     0.0    1.0
        1.0     0.0    1.0

    and you call:

    >>> append_single_line_to_file([1,2,3.2],'list.dat')

    then it will look like:

        # Time  FirstPeak   SecondPeak
        1.0     0.0    1.0
        1.0     0.0    1.0
        1       2      3.2

    """

    f = open(file_name, 'a')
    write_1d_array_to_file(f, line)
    f.close


def save_dictionary(dictionary, file_name):
    pickle.dump(dictionary, open(file_name, "wb"))


def load_dictionary(file_name):
    return pickle.load(open(file_name, "rb"))


def unzip(archive, target):
    """
    This procedure unzips a specified archive "archive.zip" and stores its contents
    to the specified to the folder "target". It only acts if a folder named "target" does not already exist.
    Parameters
    ----------
    archive : string
            Path to the .zip archive relative to the working directory or absolute
    target : string
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
    if (os.path.isdir(target) is False):
        z = zipfile.ZipFile(archive + '.zip')
        z.extractall(target)


def make_folder(path_to_folder):
    """
    creates an empty folder if the target folder does not already exist

    Parameters
    ----------
    path_to_folder : string
    Path to the newly created folder

    Example
    -------
    >>> make_folder("subfolder/new_folder_name")
    Creates a new folder with the name "NewFolder" in the directory "subfolder"
    relative to the working path but only if this folder does not already exist.
    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)


def timestring(time):
    """ Returns a timestring made of 6 digits for the inserted time adding leading zeros if necessary.
    works also with arrays."""

    if np.size(time) == 1:
        time = int(time)
        result = (6-len(str(time)))*'0'+str(time)
    else:
        result = np.zeros(np.size(time))
        for i in range(np.size(time)):
            result[i] = (6-len(str(int(time[i]))))*'0'+str(int(time[i]))
    return result


def imagestring(time):
    """ Returns a timestring made of 5 digits for the inserted time adding leading zeros if necessary.
    works also with arrays."""

    if np.size(time) == 1:
        time = int(time)
        result = (5-len(str(time)))*'0'+str(time)
    else:
        result = np.zeros(np.size(time))
        for i in range(np.size(time)):
            result[i] = (5-len(str(int(time[i]))))*'0'+str(int(time[i]))
    return result


def copy_files(files, target_directory):
    """
    Copy files from list to a directory.

    Parameters
    ----------
    files : list of strings
            list of file names
    target_directory : string
                       name of destination folder
    """
    if os.path.isdir(target_directory):
        for f in files:
            shutil.copy(f, target_directory)
    else:
        print("Warning: destination folder " + target_directory + " does not " +
              "exist. Please create it first.")


def get_files(data_directory, pattern=''):
    """
    Getting all files in the directory and subdirectories matching some pattern ('.txt' for example)

    Parameters
    ----------
    data_directory : string
                     name of folder that will be searched through
    pattern: string
             string that should be substring of relevant filenames

    Returns
    ------
    filenames : list of strings
                list of file names
    """
    filenames = []
    for path, _, files in os.walk(data_directory):
        for name in files:
            if pattern in name:
                filenames.append(os.path.join(path, name))
    return filenames


def reduce_file_size(filename, decimals, filename_suffix='_compressed'):
    """
    Reduce number of decimal places for all entries in a data file. If filename_suffix equals ''
    the existing file will be overwritten.

    Parameters
    ----------
    filename : string
               name of file that will be read
    decimals: integer
              number of decimal places
    filename_suffix: string
                     string that will be added at the end of the original filename before saving
    """

    data = pd.read_csv(filename, sep=" ", header=None)
    data = data.round(decimals)
    if decimals == 0:
        data = data.astype(int)
    splitted_filename = filename.split('.')
    new_filename = splitted_filename[0] + filename_suffix + '.' + splitted_filename[1]
    data.to_csv(new_filename, sep=" ", index=False, header=False)
