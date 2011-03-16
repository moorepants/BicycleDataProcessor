#!/usr/bin/env python
import os
import re
import tables as tab
import numpy as np
import scipy as sp
from scipy.optimize import fmin_bfgs
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

        # get the data table
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
            tau = find_timeshift(NIacc, VNacc, Fs)
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

def find_bump(accelSignal, sampleRate, speed, wheelbase, bumpLength):
    '''Returns the indices that surround the bump in the acceleration signal.

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
    type : string
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
        print 'There is no sure method here! Try Again'

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
    '''Linear tranformation from raw voltage measurements (V) to calibrated
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
    t = np.linspace(0., n/fs-1./fs, num=n)

    # shift the ni data cause it the cleaner signal
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

def find_timeshift(NIacc, VNacc, Fs, guess=None):
    '''
    Returns the timeshift (tau) of the VectorNav (VN) data relative to the
    National Instruments (NI) data.

    Parameters
    ----------
    NIacc : ndarray, shape(n, )
       The (mostly) vertical acceleration from the NI box. Should be scaled to
       meters per second squared.
    VNacc : ndarray, shape(n, )
        The (mostly) vertical acceleration from the VectorNav. Should be the
        same length as NIacc an contain the same signal albiet time shifted.
        The VectorNav signal should be leading the NI signal.
        Should be scaled to meters per second squared.
    Fs : float or int
        Sample rate of the signals. This should be the same for each signal.
    guess : float
        A good guess for the time shift.

    Returns
    -------
    tau : float
        The timeshift. A positive value corresponds to the NI signal lagging the
        VectorNav Signal.

    '''

    # subtract the mean
    niSig = NIacc - stats.nanmean(NIacc)
    vnSig = VNacc - stats.nanmean(VNacc)

    N = len(niSig)
    if N != len(vnSig):
        raise StandardError

    time = np.linspace(0, N/float(Fs), N)

    # Set up the error landscape, error vs tau
    # The NI lags the VectorNav and the time shift is typically between 0 and
    # 0.5 seconds
    tauRange = np.linspace(0., 0.5, num=1000)
    e = np.zeros_like(tauRange)
    for i, val in enumerate(tauRange):
        e[i] = sync_error(val, s1, s2, time)

    # Find initial condition from landscape and optimize!
    tau0 = tau_range[np.argmin(e)]
    tau  = fmin_bfgs(sync_error, tau0, args=(s1, s2, time))

    # if tau is not close to the other guess then say something
    return tau, e

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
    '''Generates a class that is used for the table description for raw data
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

def parse_vnav_string(vnstr, remove=0):
    '''Gets the good info from a VNav string

    Parameters
    ----------
    vnstr : string
        A string from the VectorNav serial output.
    remove : int
        Specifies how many values to remove from the beginning of the output
        list. Useful for removing VNWRG, etc.

    Returns
    -------
    vnlist : list
        A list of each element in the VectorNav string.
        ['VNWRG', '26', ..., ..., ...]

    '''
    # get rid of the $ and the *checksum
    vnstr = re.sub('\$(.*)\*.*', r'\1', vnstr)
    # make it a list
    vnlist = vnstr.split(',')
    # return the last values with regards to remove
    return vnlist[remove:]

def get_run_data(pathtofile):
    '''Returns data from the run h5 files using pytables and formats it better
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
                parsed = parse_vnav_string(pstr, remove=2)
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
