#!/usr/bin/env python

import os
import re
from operator import xor
import tables as tab
import numpy as np
import scipy as sp
from scipy.stats import nanmean, nanmedian
from scipy.optimize import fmin
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt

class Run():
    '''
    The fundamental class for a run.

    '''
    def __init__(self, runid, datafile, forceRecalc=False):
        '''Loads all parameters if available otherwise it generates them.

        Parameters
        ----------
        runid : int or string
            The run id: 5 or '00005'.
        datafile : pytable object of an hdf5 file
            This file must contain the run data table and the calibration data
            table.
        forcecalc : boolean
            If true then it will force a recalculation of all the the non raw
            data.

        '''

        # get the run data table
        datatable = datafile.root.data.datatable

        # these are the columns that may need to be calculated
        processedCols = ['SteerAngle', 'SteerRate', 'RollAngle', 'RollRate',
                         'RearWheelRate', 'SteerTorque', 'YawRate',
                         'PitchRate', 'FrameAccelerationX',
                         'FrameAccelerationY', 'FrameAccelerationZ',
                         'PullForce', 'tau']

        # make a container for the data
        self.data = {}

        # get the row number for this particular run id
        rownum = get_row_num(runid, datatable)

        # if the data hasn't been time shifted, then do it now
        curtau = datatable[rownum]['tau']
        print 'curtau', curtau
        Fs = get_cell(datatable, 'NISampleRate', rownum)
        if curtau == 0. or forceRecalc == True:
            # calculate tau for this run
            NIacc = get_cell(datatable, 'FrameAccelY', rownum)
            VNacc = get_cell(datatable, 'AccelerationZ', rownum)
            tau, error = find_timeshift(NIacc, VNacc, Fs)
            print 'This is tau', tau

            # store tau in the the table
            datatable.cols.tau[rownum] = tau
            datatable.flush()
        else:
            tau = curtau

        cdat = get_calib_data(os.path.join('..', 'BicycleDAQ', 'data',
                'CalibData', 'calibdata.h5'))

        for col in datatable.colnames:
            # grab the data for the run number and column name
            coldata = get_cell(datatable, col, rownum)
            # now check to see if we need to process the data
            if col in processedCols: # and np.sum(coldata) == 0.:
                print col, 'needs some processing'
                if col == 'tau':
                    pass
                elif col in cdat.keys():
                    print 'Processing', col
                    voltage = get_cell(datatable, cdat[col]['signal'], rownum)
                    scaled = linear_calib(voltage, cdat[col])
                    coldata = truncate_data(scaled, 'NI', Fs, tau)

            self.data[col] = coldata

    def plot(self, *args):
        '''
        Plots the time series of various signals.

        Parameters
        ----------
        args* : string
            These should be strings that correspond to processed data
            columns.

        '''
        samplerate = self.data['NISampleRate']
        #n = len(self.data['SteerAngle'])
        n = len(self.data['SteerPotentiometer'])
        t = np.linspace(0., n/samplerate, num=n)

        for i, arg in enumerate(args):
            if arg == 'SteerTorque':
                plt.plot(t, size_vector(self.data[arg]*10., n))
            else:
                plt.plot(t, size_vector(self.data[arg], n))

        plt.legend(args)

        plt.title('Rider: ' + self.data['Rider'] +
                  ', Speed: ' + str(self.data['Speed']) + 'm/s\n' +
                  'Maneuver: ' + self.data['Maneuver'] +
                  ', Environment: ' + self.data['Environment'] + '\n' +
                  'Notes: ' + self.data['Notes'])
        plt.grid()

        plt.show()

    def video(self):
        '''
        Plays the video of the run.

        '''
        # get the 5 digit string version of the run id
        runid = pad_with_zeros(str(self.data['RunID']), 5)
        viddir = os.path.join('..', 'Video')
        abspath = os.path.abspath(viddir)
        # check to see if there is a video for this run
        if (runid + '.mp4') in os.listdir(viddir):
            path = os.path.join(abspath, runid + '.mp4')
            os.system('vlc "' + path + '"')
        else:
            print "No video for this run"

def split_around_nan(sig):
    '''
    Returns the sections of an array not polluted with nans.

    Parameters
    ----------
    sig : ndarray, shape(n,)
        A one dimensional array that may or may not contain m nan values where
        0 <= m <= n.

    Returns
    -------
    indices : list, len(indices) = k
        List of tuples containing the indices for the sections of the array.
    arrays : list, len(indices) = k
        List of section arrays. All arrays of nan values are of dimension 1.

    k = number of non-nan sections + number of nans

    sig[indices[k][0]:indices[k][1]] == arrays[k]

    '''
    # if there are any nans then split the signal
    if np.isnan(sig).any():
        firstSplit = np.split(sig, np.nonzero(np.isnan(sig))[0])
        arrays = []
        for arr in firstSplit:
            # if the array has nans, then split it again
            if np.isnan(arr).any():
                arrays = arrays + np.split(arr, np.nonzero(np.isnan(arr))[0] + 1)
            # if it doesn't have nans, then just add it as is
            else:
                arrays.append(arr)
        # remove any empty arrays
        emptys = [i for i, arr in enumerate(arrays) if arr.shape[0] == 0]
        arrays = [arr for i, arr in enumerate(arrays) if i not in emptys]
        # build the indices list
        indices = []
        count = 0
        for i, arr in enumerate(arrays):
            count += len(arr)
            if np.isnan(arr).any():
                indices.append((count - 1, count))
            else:
                indices.append((count - len(arr), count))
    else:
        arrays, indices = [sig], [(0, len(sig))]

    return indices, arrays

def time_vector(numSamples, sampleRate):
    '''
    Returns a time vector starting at zero.

    Parameters
    ----------
    numSamples : int or float
        Total number of samples.
    sampleRate : int or float
        Sample rate.

    Returns
    -------
    time : ndarray, shape(numSamples,)
        Time vector starting at zero.

    '''
    ns = float(numSamples)
    sr = float(sampleRate)
    return np.linspace(0., (ns - 1.) / sr, num=ns)

def find_bump(accelSignal, sampleRate, speed, wheelbase, bumpLength):
    '''
    Returns the indices that surround the bump in the acceleration signal.

    Parameters
    ----------
    accelSignal : ndarray, shape(n,)
        This is an acceleration signal with a single distinctive large
        acceleration that signifies riding over the bump.
    sampleRate : float
        This is the sample rate of the signal.
    speed : float
        Speed of travel (or treadmill) in meters per second.
    wheelbase : float
        Wheelbase of the bicycle in meters.
    bumpLength : float
        Length of the bump in meters.

    Returns
    -------
    indices : tuple
        The first and last indice of the bump section.

    '''
    # get the indice of the larger of the max and min
    maxmin = (np.nanmax(accelSignal), np.nanmin(accelSignal))
    if np.abs(maxmin[0]) > np.abs(maxmin[1]):
        indice = np.nanargmax(accelSignal)
    else:
        indice = np.nanargmin(accelSignal)

    print 'Bump indice:', indice
    print 'Bump time:', indice / sampleRate

    # give a warning if the bump doesn't seem to be at the beginning of the run
    if indice > len(accelSignal) / 4.:
        print "This signal's max value is not in the first quarter of the data"
        print("It is at %f seconds out of %f seconds" %
            (indice / sampleRate, len(accelSignal) / sampleRate))

    bumpDuration = (wheelbase + bumpLength) / speed
    print "Bump duration:", bumpDuration
    bumpSamples = int(bumpDuration * sampleRate)
    # make the number divisible by four
    bumpSamples = int(bumpSamples / 4) * 4

    # get the first quarter before the tallest spike and whatever is after
    indices = (indice - bumpSamples / 4, indice, indice + 3 * bumpSamples / 4)

    if np.isnan(accelSignal[indices[0]:indices[1]]).any():
        print 'There is at least one NaN in this bump'

    return indices

def derivative(x, y, method='forward'):
    '''
    Return the derivative of y with respect to x.

    Parameters:
    -----------
    x : ndarray, shape(n,)
    y : ndarray, shape(n,)
    method : string
        'forward' : forward difference
        'central' : central difference
        'backward' : backward difference
        'combination' : forward on the first point, backward on the last and
        central on the rest

    Returns:
    --------
    dydx : ndarray, shape(n,) or shape(n-1,)
        for combination else shape(n-1,)

    The combo method doesn't work for matrices yet.

    '''
    if method == 'forward':
        return np.diff(y)/diff(x)
    elif method == 'combination':
        dxdy = np.zeros_like(y)
        for i, yi in enumerate(y[:]):
            if i == 0:
                dxdy[i] = (-3*y[0] + 4*y[1] - y[2])/2/(x[1]-x[0])
            elif i == len(y) - 1:
                dxdy[-1] = (3*y[-1] - 4*y[-2] + y[-3])/2/(x[-1] - x[-2])
            else:
                dxdy[i] = (y[i + 1] - y[i - 1])/2/(x[i] - x[i - 1])
        return dxdy
    elif method == 'backward':
        print 'There is no backward difference method defined, want to write one?'
    elif method == 'central':
        print 'There is no central difference method defined, want to write one?'
    else:
        print 'There is no %s method here! Try Again' % method

def pad_with_zeros(num, digits):
    '''
    Adds zeros to the front of a string needed to produce the number of
    digits.

    Parameters
    ----------
    num : string
        A string representation of a number (i.e. '25')
    digits : integer
        The total number of digits desired.

    If digits = 4 and num = '25' then the function returns '0025'.

    '''

    for i in range(digits - len(num)):
        num = '0' + num

    return num

def linear_calib(V, calibdata):
    '''
    Linear tranformation from raw voltage measurements (V) to calibrated
    data signals (s).

    Parameters
    ----------
    V : ndarray, shape(n, )
        Time series of voltage measurements.

    calibdata : dictionary
        Calibration data

    Output
    ----------
    s : ndarray, shape(n, )
        Calibrated signal

    '''

    p1 = calibdata['slope']
    p0 = calibdata['offset']
    s = p1*V + p0
    return s

def rollpitchyawrate(framerate_x, framerate_y, framerate_z, bikeparms):
    '''Transforms the measured body fixed rates to global rates by
    rotating them along the head angle.

    Parameters
    ----------
    omega_x, omega_y, omega_z : array
        Body fixed angular velocities

    bikeparms : Dictionary
        Bike parameters

    Output
    ----------
    yawrate, pitchrate, rollrate : array
        Calibrated signal

    '''
    lam = bikeparms['lambda']
    rollrate  =  omega_x*cos(lam) + omega_z*sin(lam)
    pitchrate =  omega_y
    yawrate   = -omega_x*sin(lam) + omega_z*cos(lam)
    return yawrate, pitchrate, rollrate

def steerrate(steergyro, framerate_z):
    lam = bikeparms['lambda']
    deltad = steergyro + framerate_z
    return deltad

def get_cell(datatable, colname, rownum):
    '''
    Returns the contents of a cell in a pytable. Apply unsize_vector correctly
    for padded vectors.

    Parameters
    ----------
    datatable : pytable table
        This is a pointer to the table.
    colname : str
        This is the name of the column in the table.
    rownum : int
        This is the rownumber of the cell.

    Return
    ------
    cell : varies
        This is the contents of the cell.

    '''
    cell = datatable[rownum][colname]
    # if it is a numpy array and the default size then unsize it
    if isinstance(cell, type(np.ones(1))) and cell.shape[0] == 12000:
        numsamp = datatable[rownum]['NINumSamples']
        cell = unsize_vector(cell, numsamp)

    return cell

def truncate_data(series, typ, fs, tau):
    '''
    Returns the truncated vectors with respect to the timeshift tau.

    Parameters
    ---------
    series : ndarray, shape(n, )
        A time series from the NIData or the VNavData.
    typ : string
        Either 'NI' or 'VN' depending on which signal you have.
    fs : int
        The sample frequency.
    tau : float
        The time shift.

    Returns
    -------
    truncated : ndarray, shape(m, )
        The truncated time series.

    '''
    n = len(series)
    t = time_vector(n, fs)

    # shift the ni data cause it is the cleaner signal
    tni = t - tau
    tvn = t

    # make the common time interval
    tcom = tvn[np.nonzero(tvn < tni[-1])]

    if typ == 'NI':
        truncated = sp.interp(tcom, tni, series)
    elif typ == 'VN':
        truncated = series[np.nonzero(tvn <= tcom[-1])]

    return truncated

def get_row_num(runid, table):
    '''
    Returns the row number for a particular run id.

    Parameters
    ----------
    runid : int or string
        The run id.
    table : pytable
        The run data table.

    Returns
    -------
    rownum : int
        The row number for runid.

    '''
    rownum = [x.nrow for x in table.iterrows()
              if x['RunID'] == int(runid)]
    return rownum[0]

def sync_error(tau, s1, s2, t):
    '''
    Returns the error between two signals.

    Parameters
    ----------
    tau : float
        Guess for the time shift
    s1 : ndarray, shape(n, )
        Vertical acceleration data of the NI accelerometer
    s2 : ndarray, shape(n, )
        Vertical acceleration data of the VN accelerometer. This signal may
        have some nan's.
    t : ndarray
        Time

    Returns
    -------
    e : float
        Error between the two signals.

    '''
    # the number of samples
    N = t.shape[0]
    # set up a shifted time vector
    t1_interp = np.linspace(np.min(t) + np.abs(tau),
                            np.max(t) - np.abs(tau),
                            N)
    t2_interp = t1_interp - tau
    s1_interp = sp.interp(t1_interp, t, s1);
    s2_interp = sp.interp(t2_interp,
                          t[~np.isnan(s2)],
                          s2[~np.isnan(s2)]);
    e  = sum((s1_interp[:round(.2*N)]-s2_interp[:round(.2*N)])**2)
    return e

def subtract_mean(sig):
    '''
    Subtracts the mean from a signal with nanmean.

    Parameters
    ----------
    sig : ndarray, shape(n,)

    Returns
    -------
    ndarray, shape(n,)
        sig minus the mean of sig

    '''
    return sig - nanmean(sig)

def normalize(sig):
    '''
    Normalizes the vector with respect to the maximum value.

    Parameters
    ----------
    sig : ndarray, shape(n,)

    Returns
    -------
    normSig : ndarray, shape(n,)
        The signal normalized with respect to the maximum value.

    '''
    return sig / np.nanmax(sig)

def find_timeshift(NIacc, VNacc, Fs, guess=None, sign=True):
    '''
    Returns the timeshift (tau) of the VectorNav (VN) data relative to the
    National Instruments (NI) data.

    Parameters
    ----------
    NIacc : ndarray, shape(n, )
        The (mostly) vertical acceleration from the NI box. This can be raw
        voltage or scaled signals. Be sure to set sign to False if the signals
        have been scaled already.
    VNacc : ndarray, shape(n, )
        The (mostly) vertical acceleration from the VectorNav. Should be the
        same length as NIacc an contain the same signal albiet time shifted.
        The VectorNav signal should be leading the NI signal.
    Fs : float or int
        Sample rate of the signals. This should be the same for each signal.
    guess : float
        A extra guess for the time shift if you have more insight.
    sign : boolean
        True means the signs of the two signals are opposite, False means they
        are the same.

    Returns
    -------
    tau : float
        The timeshift. A positive value corresponds to the NI signal lagging the
        VectorNav Signal.

    error : ndarray, shape(500,)
        Error versus tau.

    '''
    # the NIaccY is the negative of the VNaccZ
    if sign:
        NIacc = -NIacc

    # subtract the mean and normalize both signals
    niSig = normalize(subtract_mean(NIacc))
    vnSig = normalize(subtract_mean(VNacc))

    # raise an error if the signals are not the same length
    N = len(niSig)
    if N != len(vnSig):
        raise StandardError

    time = time_vector(N, float(Fs))

    # Set up the error landscape, error vs tau
    # The NI lags the VectorNav and the time shift is typically between 0 and
    # 0.5 seconds
    tauRange = np.linspace(0., .5, num=500)
    error = np.zeros_like(tauRange)
    for i, val in enumerate(tauRange):
        error[i] = sync_error(val, niSig, vnSig, time)

    # find initial condition from landscape
    tau0 = tauRange[np.argmin(error)]

    print "The minimun of the error landscape is %f and the provided guess is %f" % (tau0, guess)

    # if tau is not close to the other guess then say something
    isNone = guess == None
    isInRange = 0. < guess < 1.
    isCloseToTau = guess - .1 < tau0 < guess + .1

    if not isNone and isInRange and not isCloseToTau:
        print("This tau0 may be a bad guess, check the error function!" +
              " Using guess instead.")
        tau0 = guess

    print "Using %f as the guess for minimization." % tau0

    tau  = fmin(sync_error, tau0, args=(niSig, vnSig, time))[0]

    print "This is what came out of the minimization:", tau

    # if the minimization doesn't do a good job, just use the tau0
    if np.abs(tau - tau0) > 0.01:
        tau = tau0
        print "Bad minimizer!! Using the guess, %f, instead." % tau

    return tau, error

def butterworth(data, freq, sampRate, order=2, axis=-1):
    """
    Returns the Butterworth filtered data set.

    Parameters:
    -----------
    data : ndarray
    freq : float or int
        cutoff frequency in hertz
    sampRate : float or int
        sampling rate in hertz
    order : int
        the order of the Butterworth filter
    axis : int
        the axis to filter along

    Returns:
    --------
    filteredData : ndarray
        filtered version of data

    This does a forward and backward Butterworth filter and averages the two.

    """
    nDim = len(data.shape)
    dataSlice = '['
    for dim in range(nDim):
        if dim == axis or (np.sign(axis) == -1 and dim == nDim + axis):
            dataSlice = dataSlice + '::-1, '
        else:
            dataSlice = dataSlice + ':, '
    dataSlice = dataSlice[:-2] + '].copy()'

    b, a = butter(order, float(freq) / float(sampRate) / 2.)
    forwardFilter = lfilter(b, a, data, axis=axis)
    reverseFilter = lfilter(b, a, eval('data' + dataSlice), axis=axis)
    return(forwardFilter + eval('reverseFilter' + dataSlice)) / 2.

def unsize_vector(vector, m):
    '''Returns a vector with the nan padding removed.

    Parameters
    ----------
    vector : numpy array, shape(n, )
        A vector that may or may not have nan padding and the end of the data.
    m : int
        Number of valid values in the vector.

    Returns
    -------
    numpy array, shape(m, )
        The vector with the padding removed. m = samplenum

    '''
    # this case removes the nan padding
    if m < len(vector):
        oldvec = vector[:m]
    elif m > len(vector):
        oldvec = vector
        print('This one is actually longer, you may want to get the ' +
              'complete data, or improve this function so it does that.')
    elif m == len(vector):
        oldvec = vector
    else:
        print "Something's wrong here"
    return oldvec

def size_vector(vector, m):
    '''Returns a vector with nan's padded on to the end or a slice of the
    vector if length is less than the length of the vector.

    Parameters
    ----------
    vector : numpy array, shape(n, )
        The vector that needs sizing.
    m : int
        The desired length after the sizing.

    Returns
    -------
    newvec : numpy array, shape(m, )

    '''
    nsamp = len(vector)
    # if the desired length is larger then pad witn nan's
    if m > nsamp:
        nans = np.ones(m-nsamp)*np.nan
        newvec = np.append(vector, nans)
    elif m < nsamp:
        newvec = vector[:m]
    elif m == nsamp:
        newvec = vector
    else:
        print "This didn't work"
    return newvec

def fill_table(datafile):
    '''Adds all the data from the hdf5 files in the h5 directory to the table.

    Parameters
    ----------
    datafile : string
        path to the main hdf5 file: InstrumentedBicycleData.h5

    '''

    # load the files from the ../BicycleDAQ/data/h5 directory
    pathtoh5 = os.path.join('..', 'BicycleDAQ', 'data', 'h5')
    files = sorted(os.listdir(pathtoh5))
    # open an hdf5 file for appending
    data = tab.openFile(datafile, mode='a')
    # get the table
    datatable = data.root.data.datatable
    # get the row
    row = datatable.row
    # fill the rows with data
    for run in files:
        print 'Adding run: %s' % run
        rundata = get_run_data(os.path.join(pathtoh5, run))
        for par, val in rundata['par'].items():
            row[par] = val
        # only take the first 12000 samples for all runs
        for i, col in enumerate(rundata['NICols']):
            try: # there are no roll pot measurements
                row[col] = size_vector(rundata['NIData'][i], 12000)
            except:
                print "There is no %s measurement" % col
        for i, col in enumerate(rundata['VNavCols']):
            row[col] = size_vector(rundata['VNavData'][i], 12000)
        row.append()
    datatable.flush()
    data.close()

def create_database():
    '''Creates an HDF5 file for data collected from the instrumented bicycle'''

    # load the latest file in the ../BicycleDAQ/data/h5 directory
    pathtoh5 = os.path.join('..', 'BicycleDAQ', 'data', 'h5')
    files = sorted(os.listdir(pathtoh5))
    filteredrun = get_run_data(os.path.join(pathtoh5, files[0]))
    unfilteredrun = get_run_data(os.path.join(pathtoh5, files[-1]))
    if filteredrun['par']['ADOT'] is not 14:
        print('Run %d is not a filtered run, choose again' %
              filteredrun['par']['RunID'])
    if unfilteredrun['par']['ADOT'] is not 253:
        print('Run %d is not a unfiltered run, choose again' %
              unfilteredrun['par']['RunID'])
    # generate the table description class
    RunTable = create_run_table_class(filteredrun, unfilteredrun)
    # open a new hdf5 file for writing
    data = tab.openFile('InstrumentedBicycleData.h5', mode='w',
                        title='Instrumented Bicycle Data')
    # create a group for the raw data
    rgroup = data.createGroup('/', 'data', 'Data')
    # add the data table to this group
    rtable = data.createTable(rgroup, 'datatable', RunTable, 'Primary Data Table')
    rtable.flush()
    data.close()

def create_run_table_class(filteredrun, unfilteredrun):
    '''
    Returns a class that is used for the table description for raw data
    for each run.

    Parameters
    ----------
    rundata : dict
        Contains the python dictionary of a particular run.

    Returns
    -------
    Run : class
        Table description class for pytables with columns defined.

    '''

    # combine the VNavCols from unfiltered and filtered
    VNavCols = set(filteredrun['VNavCols'] + unfilteredrun['VNavCols'])

    # set up the table description
    class RunTable(tab.IsDescription):
        # add all of the column headings from par, NICols and VNavCols
        for i, col in enumerate(unfilteredrun['NICols']):
            exec(col + " = tab.Float32Col(shape=(12000, ), pos=i)")
        for k, col in enumerate(VNavCols):
            exec(col + " = tab.Float32Col(shape=(12000, ), pos=i+1+k)")
        for i, (key, val) in enumerate(unfilteredrun['par'].items()):
            pos = k+1+i
            if isinstance(val, type(1)):
                exec(key + " = tab.Int64Col(pos=pos)")
            elif isinstance(val, type('')):
                exec(key + " = tab.StringCol(itemsize=200, pos=pos)")
            elif isinstance(val, type(1.)):
                exec(key + " = tab.Float64Col(pos=pos)")
            elif isinstance(val, type(np.ones(1))):
                exec(key + " = tab.Float64Col(shape=(" + str(len(val)) + ", ), pos=pos)")

        # add the columns for the processed data
        processedCols = ['SteerAngle', 'SteerRate', 'RollAngle', 'RollRate',
                         'RearWheelRate', 'SteerTorque', 'YawRate',
                         'PitchRate', 'FrameAccelerationX',
                         'FrameAccelerationY', 'FrameAccelerationZ',
                         'PullForce', 'tau']
        for k, col in enumerate(processedCols):
            if col == 'tau':
                exec(col + " = tab.Float32Col(pos=i+1+k)")
            else:
                exec(col + " = tab.Float32Col(shape=(12000, ), pos=i+1+k)")

        # get rid intermediate variables so they are not stored in the class
        del(i, k, col, key, pos, val, processedCols)

    return RunTable

def parse_vnav_string(vnStr):
    '''
    Returns a list of the information in a VN-100 text string and whether the
    checksum failed.

    Parameters
    ----------
    vnStr : string
        A string from the VectorNav serial output.
    remove : int
        Specifies how many values to remove from the beginning of the output
        list. Useful for removing VNWRG, etc.

    Returns
    -------
    vnlist : list
        A list of each element in the VectorNav string.
        ['VNWRG', '26', ..., ..., ..., checksum]
    chkPass : boolean
        True if the checksum is correct and false if it isn't.
    vnrrg : boolean
        True if the str is a register reading, false if it is an async
        reading, and None if chkPass is false.

    '''
    # calculate the checksum of the raw string
    calcChkSum = vnav_checksum(vnStr)
    # get rid of the $ and the *checksum
    vnMeat = re.sub('''(?x) # verbose
                      \$ # match the dollar sign at the beginning
                      (.*) # this is the content
                      \* # match the asterisk
                      (\w*) # the checksum
                      \s* # the newline characters''', r'\1,\2', vnStr)
    # make it a list the last item should be the checksum
    vnList = vnMeat.split(',')
    # set the vnrrg flag
    if vnList[0] == 'VNRRG':
        vnrrg = True
    else:
        vnrrg = False
    # see if the checksum passes
    chkPass = calcChkSum == vnList[-1]
    if not chkPass:
        print "Checksum failed"
        print vnStr
        vnrrg = None
    # return the list and whether or not the checksum failed
    return vnList, chkPass, vnrrg

def vnav_checksum(vnStr):
    '''
    Returns the checksum in hex for the VN-100 string.

    Parameters
    ----------
    vnStr : string
        Of the form '$...*X' where X is the two digit checksum.
    Returns
    -------
    chkSum : string
        Two digit hex representation of the checksum.

    '''
    chkStr = re.sub('''(?x) # verbose
                       \$ # match the dollar sign
                       (.*) # match the stuff the checksum is calculated from
                       \* # match the asterisk
                       \w* # the checksum
                       \s* # the newline characters''', r'\1', vnStr)
    checksum = reduce(xor, map(ord, chkStr))
    return hex(checksum)[2:]

def get_run_data(pathtofile):
    '''
    Returns data from the run h5 files using pytables and formats it better
    for python.

    Parameters
    ----------
    pathtofile : string
        The path to the h5 file that contains run data.

    Returns
    -------
    rundata : dictionary
        A dictionary that looks similar to how the data was stored in Matlab.

    '''

    # open the file
    runfile = tab.openFile(pathtofile)

    # intialize a dictionary for storage
    rundata = {}

    # put the parameters into a dictionary
    rundata['par'] = {}
    for col in runfile.root.par:
        # convert them to regular python types
        try:
            if col.name == 'Speed':
                rundata['par'][col.name] = float(col.read()[0])
            else:
                rundata['par'][col.name] = int(col.read()[0])
        except:
            pstr = str(col.read()[0])
            rundata['par'][col.name] = pstr
            if pstr[0] == '$':
                parsed = parse_vnav_string(pstr)[0][2:-1]
                if len(parsed) == 1:
                    try:
                        parsed = int(parsed[0])
                    except:
                        parsed = parsed[0]
                else:
                    parsed = np.array([float(x) for x in parsed])
                rundata['par'][col.name] = parsed

    if 'Notes' not in rundata['par'].keys():
        rundata['par']['Notes'] = ''

    # get the NIData and VNavData
    rundata['NIData'] = runfile.root.NIData.read()
    rundata['VNavData'] = runfile.root.VNavData.read()

    # make the array into a list of python string and gets rid of unescaped
    # control characters
    columns = [re.sub(r'[^ -~].*', '', str(x))
               for x in runfile.root.VNavCols.read()]
    # gets rid of white space
    rundata['VNavCols'] = [x.replace(' ', '') for x in columns]
    # make a list of NI columns from the InputPair structure from matlab
    rundata['NICols'] = []
    for col in runfile.root.InputPairs:
        rundata['NICols'].append((str(col.name), int(col.read()[0])))

    rundata['NICols'].sort(key=lambda x: x[1])

    rundata['NICols'] = [x[0] for x in rundata['NICols']]

    # get the VNavDataText
    rundata['VNavDataText'] = [str(x) for x in runfile.root.VNavDataText.read()]

    # close the file
    runfile.close()

    return rundata

def get_calib_data(pathtofile):
    '''
    Returns calibration data from the run h5 files using pytables and
    formats it as a dictionairy.

    Parameters
    ----------
    pathtofile : string
        The path to the h5 file that contains the calibration data, normally:
        pathtofile = '../BicycleDAQ/data/CalibData/calibdata.h5'

    Returns
    -------
    calibdata : dictionary
        A dictionary that looks similar to how the data was stored in Matlab.

    '''

    # open the file
    calibfile = tab.openFile(pathtofile)

    # intialize a dictionary for storage
    calibdata = {}

    # Specify names to check
    namelist = ['name' + str(i) for i in range(1, 7)]
    fieldnamelist = []

    # Generate dictionary structure
    for a in calibfile.root.calibdata.__iter__():
        if a.name in namelist:
            calibdata[str(a.read()[0])] = {}
            fieldnamelist.append(str(a.read()[0]))

    # Fill dictionary structure
    for a in calibfile.root.calibdata.__iter__():
        i = int(re.sub('\D', '', a.name)) - 1
        calibdata[fieldnamelist[i]][re.sub('\d', '', a.name)] = a.read()[0]

    calibfile.close()

    return calibdata
