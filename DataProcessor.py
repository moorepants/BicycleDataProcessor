#!/usr/bin/env python

# built in imports
import os
import re
import datetime
import copy
from operator import xor
from math import pi

# dependencies
import tables as tab
import numpy as np
import scipy as sp
from scipy.stats import nanmean, nanmedian
from scipy.optimize import fmin
from scipy.signal import butter, lfilter
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from bicycleparameters import bicycleparameters as bp

class Signal(np.ndarray):
    '''A class for collecting the data for a single signal in a run.'''

    # define some basic unit converions
    conversions = {'degree->radian': pi / 180.,
                   'inch*pound->newton*meter': 25.4 / 1000. * 4.44822162,
                   'pound->newton': 4.44822162,
                   'degree/second->radian/second': pi / 180.,
                   'feet/second->meter/second': 12. * 2.54 / 100.,
                   'mile/hour->meter/second': 0.00254 * 12 / 5280. / 3600.}

    def __new__(cls, inputArray, metadata):
        '''Returns an instance of the Signal class with the addition signal
        data.'''
        if len(inputArray.shape) > 1:
            raise ValueError('Signals must be arrays of one dimension.')
        # cast the input array into my subclass of ndarray
        obj = np.asarray(inputArray).view(cls)
        # add the metadata to the object
        obj.name = metadata['name']
        obj.runid = metadata['runid']
        obj.sampleRate = metadata['sampleRate']
        obj.source = metadata['source']
        obj.units = metadata['units']
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.name = getattr(obj, 'name', None)
        self.runid = getattr(obj, 'runid', None)
        self.sampleRate = getattr(obj, 'sampleRate', None)
        self.source = getattr(obj, 'source', None)
        self.units = getattr(obj, 'units', None)

    def __array_wrap__(self, outputArray, context=None):
        outputArray.units = 'booger'
        return np.ndarray.__array_wrap__(self, outputArray, context)

    def __add__(self, otherSignal):
        '''Defines how signals add togther.'''

        try:
            if self.units != otherSignal.units:
                raise ValueError('Cannot add signals of differing units.')
            elif self.sampleRate != otherSignal.sampleRate:
                raise ValueError('Cannot add signals with different sample rates.')
            else:
                newSig = np.add(self, otherSignal)
                data = {'runid': self.runid,
                        'name': None,
                        'units': self.units,
                        'source': None,
                        'sampleRate': self.sampleRate}
                return Signal(newSig, data)
        except AttributeError:
                return np.add(self, otherSignal)

    def __mul__(self, otherSignal):
        '''Returns a new signal.'''
        try:
            if self.sampleRate != otherSignal.sampleRate:
                raise ValueError('Cannot multiply signals with different sample rates.')
            else:
                newSig = np.multiply(self, otherSignal)
                newUnits = self.units + '*' + otherSignal.units
                data = {'runid': self.runid,
                        'name': None,
                        'units': newUnits,
                        'source': None,
                        'sampleRate': self.sampleRate,
                        'signal': newSig,
                        'numberOfSamples': len(newSig)}
                return Signal(newSig, data)
        except AttributeError:
                return np.multiply(self, otherSignal)

    def __div__(self, otherSignal):
        '''Returns a new signal.'''
        try:
            if self.sampleRate != otherSignal.sampleRate:
                raise ValueError('Cannot multiply signals with different sample rates.')
            else:
                newSig = np.divide(self, otherSignal)
                newUnits = self.units + '/' + otherSignal.units
                data = {'runid': self.runid,
                        'name': None,
                        'units': newUnits,
                        'source': None,
                        'sampleRate': self.sampleRate,
                        'signal': newSig,
                        'numberOfSamples': len(newSig)}
                return Signal(newSig, data)
        except AttributeError:
                return np.divide(self, otherSignal)

    def plot(self):
        '''Plots and returns the time signal versus time.'''
        time = time_vector(len(self), self.sampleRate)
        line = plt.plot(time, self)
        plt.xlabel('Time [second]')
        plt.ylabel(self.name + ' [' + self.units + ']')
        plt.title('Signal plot during run {0}'.format(self.name))
        plt.show()
        return line

    def frequency(self):
        '''Returns the frequency content of the signal.'''
        raise NotImplementedError('This is a place holder.')

    def derivative(self):
        '''Returns the derivative of the signal.'''
        raise NotImplementedError('This is a place holder.')

    def filter(self, frequency):
        '''Returns the low pass Butterworth filtered signal at the specifited
        frequency.'''
        return butterworth(self, frequency, self.sampleRate)

    def truncate(self, tau):
        '''Returns the shifted and truncated signal based on the provided
        timeshift, tau.'''
        return truncate_data(self, self.source, self.sampleRate, tau)

    def as_dictionary(self):
        '''Returns the signal data as a dictionary.'''
        data = {'runid': self.runid,
                'name': self.name,
                'units': self.units,
                'source': self.source,
                'sampleRate': self.sampleRate,
                'numberOfSamples': self.numberOfSamples}
        return data

    def convert_units(self, units):
        '''Returns a signal with different units and proper scaling.'''
        try:
            conversion = self.units + '->' + units
            newSig = self * self.conversions[conversion]
        except KeyError:
            try:
                conversion = units + '->' + self.units
                newSig = self / self.conversions[conversion]
            except KeyError:
                raise KeyError(('Conversion from {0} to {1} is not ' +
                    'possible or not defined.').format(self.units, units))
        # make the new signal
        newSig.units = units

        return newSig

class RawSignal(Signal):
    '''A class for collecting the data for a single raw signal in a run.'''
    def __init__(self, runid, signalName, database):

        # get the tables
        dTab = database.root.data.datatable
        sTab = database.root.data.signaltable
        cTab = database.root.data.calibrationtable

        # get the row number for this particular run id
        rownum = get_row_num(runid, dTab)

        self.runid = runid
        self.timeStamp = matlab_date_to_object(get_cell(dTab, 'DateTime',
            rownum))
        self.calibrationType, self.units, self.source = [(row['calibration'],
            row['units'], row['source'])
            for row in sTab.where('signal == signalName')][0]
        self.name = signalName

        try:
            self.sensor = Sensor(self.name, cTab)
        except KeyError:
            print "There is no sensor named {0}.".format(signalName)

        # this assumes that the supply voltage for this signal is the same for
        # all sensor calibrations
        try:
            supplySource = [row['runSupplyVoltageSource']
                           for row in cTab.where('name == signalName')][0]
            if supplySource == 'na':
                self.supply = [row['runSupplyVoltage']
                               for row in cTab.where('name == signalName')][0]
            else:
                self.supply = get_cell(dTab, supplySource, rownum)
        except IndexError:
            print "{0} does not have a supply voltage.".format(signalName)
            print "-" * 79

        self.signal = get_cell(dTab, signalName, rownum)

        self.numberOfSamples = len(self.signal)

        # get the appropriate sample rate
        if self.source == 'NI':
            sampRateCol = 'NISampleRate'
        elif self.source == 'VN':
            sampRateCol = 'VNavSampleRate'
        else:
            raise ValueError('{0} is not a valid source.'.format(self.source))

        self.sampleRate = dTab[rownum][dTab.colnames.index(sampRateCol)]

    def scale(self):
        '''Returns the scaled signal based on the calibration data for the
        supplied date.

        Returns
        -------
        : ndarray (n,)
            Scaled signal.

        '''
        # these will need to be changed once we start measuring them
        doNotScale = ['LeanPotentiometer',
                      'HipPotentiometer',
                      'TwistPotentiometer']
        if self.calibrationType in ['none', 'matrix'] or self.name in doNotScale:
            print "Not scaling {0}".format(self.name)
            return self.name, self.signal, self.units
        else:
            print "Scaling {0}".format(self.name)
            # pick the largest calibration date without surpassing the run date
            runDate = self.timeStamp
            # make a list of calibration ids and time stamps
            dateIdPairs = [(k, matlab_date_to_object(v['timeStamp']))
                           for k, v in self.sensor.data.iteritems()]
            # sort the pairs with the most recent date first
            dateIdPairs.sort(key=lambda x: x[1], reverse=True)
            # go through the list and return the index at which the calibration
            # date is larger than the run date
            for i, pair in enumerate(dateIdPairs):
                if pair[1] > runDate:
                    break
            calibData = self.sensor.data[dateIdPairs[i][0]]

            slope = calibData['slope']
            bias = calibData['bias']
            intercept = calibData['offset']
            calibrationSupplyVoltage = calibData['calibrationSupplyVoltage']

            if self.calibrationType == 'interceptStar':
                calibratedSignal = (calibrationSupplyVoltage / self.supply *
                                    slope * self.signal + intercept)
            elif self.calibrationType == 'intercept':
                calibratedSignal = (calibrationSupplyVoltage / self.supply *
                                    (slope * self.signal + intercept))
            elif self.calibrationType == 'bias':
                calibratedSignal = (calibrationSupplyVoltage / self.supply *
                                    slope * (self.signal - bias))
            else:
                raise StandardError("None of the calibration equations worked.")

            return calibData['signal'], calibratedSignal, calibData['units']

    def plot_scaled(self):
        '''Plots and returns the time series versus time.'''
        time = time_vector(self.numberOfSamples, self.sampleRate)
        scaled = self.scale()
        line = plt.plot(time, scaled[1])
        plt.xlabel('Time [s]')
        plt.ylabel(scaled[2])
        plt.title('{0} signal during run {1}'.format(scaled[0],
                  str(self.runid)))
        plt.show()
        return line

class Sensor():
    '''This class is a container for calibration data for a sensor.'''

    def __init__(self, name, calibrationTable):
        '''Initializes this sensor class.

        Parameters
        ----------
        name : string
            The CamelCase name of the sensor (e.g. SteerTorqueSensor).
        calibrationTable : pyTables table object
            This is the calibration data table that contains all the data taken
            during calibrations.

        '''
        self.name = name
        self.store_calibration_data(calibrationTable)

    def store_calibration_data(self, calibrationTable):
        '''Stores a dictionary of calibration data for the sensor for all
        calibration dates.'''

        self.data = {}

        for row in calibrationTable.iterrows():
            if self.name == row['name']:
                self.data[row['calibrationID']] = {}
                for col in calibrationTable.colnames:
                    self.data[row['calibrationID']][col] = row[col]
        if self.data == {}:
            raise KeyError(('{0} is not a valid sensor ' +
                           'name').format(self.name))

class Run():
    '''The fundamental class for a run.'''

    def __init__(self, runid, database, forceRecalc=False):
        '''Loads all parameters if available otherwise it generates them.

        Parameters
        ----------
        runid : int or string
            The run id should be an integer, e.g. 5, or a five digit string with
            leading zeros, e.g. '00005'.
        database : pytable object of an hdf5 file
            This file must contain the run data table and the calibration data
            table.
        forceRecalc : boolean
            If true then it will force a recalculation of all the the non raw
            data.

        '''

        # get the tables
        dataTable = database.root.data.datatable
        signalTable = database.root.data.signaltable

        # get the row number for this particular run id
        rownum = get_row_num(runid, dataTable)

        # make some dictionaries to store all the data
        self.metadata = {}
        self.rawSignals = {}
        self.calibratedSignals = {}
        self.truncatedSignals = {}
        self.computedSignals ={}

        # make lists of the input and output signals
        rawDataCols = [x['signal'] for x in
                       signalTable.where("isRaw == True")]
        computedCols = [x['signal'] for x in
                        signalTable.where("isRaw == False")]

        # store the current data for this run
        for col in dataTable.colnames:
            if col not in (rawDataCols + computedCols):
                self.metadata[col] = get_cell(dataTable, col, rownum)

        for col in rawDataCols:
            self.rawSignals[col] = RawSignal(runid, col, database)

        # tell the user about the run
        print self

        # load the parameters for the bicycle
        # this code will not work for other bicycle/rider combinations. it will
        # need to be updated
        bicycles = {'Rigid Rider': 'Rigid'}
        pathToBicycles = '/media/Data/Documents/School/UC Davis/Bicycle Mechanics/BicycleParameters/data/bicycles'
        rigid = bp.Bicycle(bicycles[self.metadata['Bicycle']], pathToBicycles=pathToBicycles)
        pathToRider = '/media/Data/Documents/School/UC Davis/Bicycle Mechanics/BicycleParameters/data/riders'
        rigid.add_rider(pathToRider=pathToRider + '/Jason/JasonRigidBenchmark.txt')
        self.bikeParameters = rigid.parameters['Benchmark']

        if forceRecalc == True:
            # calibrate the signals for the run
            for sig in self.rawSignals.values():
                sD = sig.as_dictionary()
                sD['name'], sD['signal'], sD['units'] = sig.scale()
                self.calibratedSignals[sD['name']] = Signal(sD)

            # calculate tau for this run
            self.tau = find_timeshift(
                self.calibratedSignals['AccelerometerAccelerationY'].signal,
                self.calibratedSignals['AccelerationZ'].signal,
                self.metadata['NISampleRate'],
                self.metadata['Speed'])

            # truncate all the raw data signals
            for name, sig in self.calibratedSignals.items():
                trData = sig.as_dictionary()
                trData['signal'] = sig.truncate(self.tau)
                self.truncatedSignals[name] = Signal(trData)

            # compute the final output signals
            self.computedSignals['PullForce'] = self.truncatedSignals['PullForce'].signal
            self.computedSignals['RearWheelRate'] = self.truncatedSignals['RearWheelRate'].signal
            self.computedSignals['RollAngle'] = self.truncatedSignals['RollAngle'].signal
            self.computedSignals['SteerAngle'] = self.truncatedSignals['SteerAngle'].signal
            self.computedSignals['PullForce'] = self.truncatedSignals['PullForce'].signal
            self.computedSignals['ForwardSpeed'] = (
                self.bikeParameters['rR'].nominal_value *
                self.truncatedSignals['RearWheelRate'].signal)
            # these are both in degrees
            self.computedSignals['SteerRate'] = steer_rate(
                self.truncatedSignals['ForkRate'].signal,
                self.truncatedSignals['AngularRateZ'].signal)
            yr, rr, pr = yaw_roll_pitch_rate(
                    self.truncatedSignals['AngularRateX'].signal,
                    self.truncatedSignals['AngularRateY'].signal,
                    self.truncatedSignals['AngularRateZ'].signal,
                    self.bikeParameters['lam'].nominal_value,
                    rollAngle=self.truncatedSignals['RollAngle'].signal)
            self.computedSignals['YawRate'] = yr
            self.computedSignals['RollRate'] = rr
            self.computedSignals['PitchRate'] = pr
        else:
            for col in computedCols:
                self.computedSignals[col] = RawSignal(runid, col, datafile)

    def plot(self, *args):
        '''
        Plots the time series of various signals.

        Parameters
        ----------
        args* : string
            These should be strings that correspond to processed data
            columns.
        truncated : boolean
            If true, then the plots will show the data that has been shifted
            and truncated.

        '''
        sampleRate = self.metadata['NISampleRate']

        for i, arg in enumerate(args):
            time = time_vector(len(self.computedSignals[arg]), sampleRate)
            plt.plot(time, self.computedSignals[arg])

        plt.legend(args)

        plt.title('Rider: ' + self.metadata['Rider'] +
                  ', Speed: ' + str(self.metadata['Speed']) + 'm/s\n' +
                  'Maneuver: ' + self.metadata['Maneuver'] +
                  ', Environment: ' + self.metadata['Environment'] + '\n' +
                  'Notes: ' + self.metadata['Notes'])
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

    def __str__(self):
        '''Prints basic run information to the screen.'''

        line = "=" * 79
        info = '''Run # {0}
Environment: {1}
Rider: {2}
Bicycle: {3}
Speed: {4}
Maneuver: {5}
Notes: {6}'''.format(
        self.metadata['RunID'],
        self.metadata['Environment'],
        self.metadata['Rider'],
        self.metadata['Bicycle'],
        self.metadata['Speed'],
        self.metadata['Maneuver'],
        self.metadata['Notes'])

        return line + '\n' + info + '\n' + line


def matlab_date_to_object(matDate):
    '''Returns a date time object based on a Matlab datestr() output.

    Parameters
    ----------
    matDate : string
        String in the form '21-Mar-2011 14:45:54'.

    Returns
    -------
    python datetime object

    '''
    return datetime.datetime.strptime(matDate, '%d-%b-%Y %H:%M:%S')

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
    '''Returns a time vector starting at zero.

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
    if indice > len(accelSignal) / 3.:
        print "This signal's max value is not in the first third of the data"
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

def yaw_roll_pitch_rate(angularRateX, angularRateY, angularRateZ,
                        lam, rollAngle=0.):
    '''Returns the bicycle frame yaw, roll and pitch rates based on the body
    fixed rate data taken with the VN-100 and optionally the roll angle
    measurement.

    Parameters
    ----------
    angularRateX : ndarray, shape(n,)
        The body fixed rate perpendicular to the headtube and pointing forward.
    angularRateY : ndarray, shape(n,)
        The body fixed rate perpendicular to the headtube and pointing to the
        right of the bicycle.
    angularRateZ : ndarray, shape(n,)
        The body fixed rate aligned with the headtube and pointing downward.
    lam : float
        The steer axis tilt.
    rollAngle : ndarray, shape(n,), optional
        The roll angle of the bicycle frame.

    Returns
    -------
    yawRate : ndarray, shape(n,)
        The yaw rate of the bicycle frame.
    rollRate : ndarray, shape(n,)
        The roll rate of the bicycle frame.
    pitchRate : ndarray, shape(n,)
        The pitch rate of the bicycle frame.

    '''
    yawRate = -(angularRateX*np.sin(lam) -
                angularRateZ * np.cos(lam)) / np.cos(rollAngle)
    rollRate = angularRateX * np.cos(lam) + angularRateZ * np.sin(lam)
    pitchRate = (angularRateY + angularRateX * np.sin(lam) * np.tan(rollAngle) -
                 angularRateZ * np.cos(lam) * np.tan(rollAngle))

    return yawRate, rollRate, pitchRate

def steer_rate(forkRate, angularRateZ):
    '''Returns the steer rate.'''
    return forkRate - angularRateZ

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

def truncate_data(signal, source, sampleRate, tau):
    '''
    Returns the truncated vectors with respect to the timeshift tau.

    Parameters
    ---------
    signal : ndarray, shape(n, )
        A time signal from the NIData or the VNavData.
    source : string
        Either 'NI' or 'VN' depending on which signal you have.
    sampleRate : int
        The sample frequency.
    tau : float
        The time shift.

    Returns
    -------
    truncated : ndarray, shape(m, )
        The truncated time signal.

    '''
    t = time_vector(len(signal), sampleRate)

    # shift the ni data cause it is the cleaner signal
    tni = t - tau
    tvn = t

    # make the common time interval
    tcom = tvn[np.nonzero(tvn < tni[-1])]

    if source == 'NI':
        truncated = sp.interp(tcom, tni, signal)
    elif source == 'VN':
        truncated = signal[np.nonzero(tvn <= tcom[-1])]

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
    # the row number should be the same as the run id but there is a
    # possibility that it isn't
    rownum = table[int(runid)]['RunID']
    if rownum != int(runid):
        rownum = [x.nrow for x in table.iterrows()
                  if x['RunID'] == int(runid)][0]
        print "The row numbers in the database do not match the run ids!"
    return rownum

def sync_error(tau, signal1, signal2, time):
    '''Returns the error between two signal time histories.

    Parameters
    ----------
    tau : float
        The time shift.
    signal1 : ndarray, shape(n, )
        The signal that will be interpolated. This signal is
        typically "cleaner" that signal2 and/or has a higher sample rate.
    signal2 : ndarray, shape(n, )
        The signal that will be shifted to syncronize with signal 1.
    time : ndarray
        Time

    Returns
    -------
    error : float
        Error between the two signals for the given tau.

    '''
    # make sure tau isn't too large
    if np.abs(tau) >= time[-1]:
        raise ValueError(('abs(tau), {0}, must be less than or equal to ' +
                         '{1}').format(str(np.abs(tau)), str(time[-1])))

    # this is the time for the second signal which is assumed to lag the first
    # signal
    shiftedTime = time + tau

    # create time vector where the two signals overlap
    if tau > 0:
        intervalTime = shiftedTime[np.nonzero(shiftedTime < time[-1])]
    else:
        intervalTime = shiftedTime[np.nonzero(shiftedTime > time[0])]

    # interpolate between signal 1 samples to find points that correspond in
    # time to signal 2 on the shifted time
    sig1OnInterval = sp.interp(intervalTime, time, signal1);

    # truncate signal 2 to the time interval
    if tau > 0:
        sig2OnInterval = signal2[np.nonzero(shiftedTime <= intervalTime[-1])]
    else:
        sig2OnInterval = signal2[np.nonzero(shiftedTime >= intervalTime[0])]

    # calculate the error between the two signals
    error = np.linalg.norm(sig1OnInterval - sig2OnInterval)

    return error

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

def find_timeshift(niAcc, vnAcc, sampleRate, speed):
    '''Returns the timeshift, tau, of the VectorNav [VN] data relative to the
    National Instruments [NI] data.

    Parameters
    ----------
    NIacc : ndarray, shape(n, )
        The acceleration of the NI accelerometer in its local Y direction. 
    VNacc : ndarray, shape(n, )
        The acceleration of the VN-100 in its local Z direction. Should be the
        same length as NIacc and contains the same signal albiet time shifted.
        The VectorNav signal should be leading the NI signal.
    sampleRate : integer
        Sample rate of the signals. This should be the same for each signal.
    speed : float
        The approximate forward speed of the bicycle.

    Returns
    -------
    tau : float
        The timeshift.

    '''
    # raise an error if the signals are not the same length
    N = len(niAcc)
    if N != len(vnAcc):
        raise StandardError('Signals are not the same length!')

    # make a time vector
    time = time_vector(N, sampleRate)

    # scale the NI signal from volts to m/s**2, and switch the sign
    niSig = -niAcc
    vnSig = vnAcc

    # some constants for find_bump
    wheelbase = 1.02
    bumpLength = 1.
    cutoff = 50.
    # filter the NI Signal
    filNiSig = butterworth(niSig, cutoff, sampleRate)
    # find the bump in the filtered NI signal
    niBump =  find_bump(filNiSig, sampleRate, speed, wheelbase, bumpLength)

    # remove the nan's in the VN signal and the time
    v = vnSig[np.nonzero(np.isnan(vnSig) == False)]
    t = time[np.nonzero(np.isnan(vnSig) == False)]
    # fit a spline through the data
    vn_spline = UnivariateSpline(t, v, k=3, s=0)
    # and filter it
    filVnSig = butterworth(vn_spline(time), cutoff, sampleRate)
    # and find the bump in the filtered VN signal
    vnBump = find_bump(filVnSig, sampleRate, speed, wheelbase, bumpLength)

    # get an initial guess for the time shift based on the bump indice
    guess = (niBump[1] - vnBump[1]) / float(sampleRate)

    # find the section that the bump belongs to
    indices, arrays = split_around_nan(vnSig)
    for pair in indices:
        if pair[0] <= vnBump[1] < pair[1]:
            bSec = pair

    # subtract the mean and normalize both signals
    niSig = normalize(subtract_mean(niSig))
    vnSig = normalize(subtract_mean(vnSig))

    niBumpSec = niSig[bSec[0]:bSec[1]]
    vnBumpSec = vnSig[bSec[0]:bSec[1]]
    timeBumpSec = time[bSec[0]:bSec[1]]

    if len(niBumpSec) < 200:
        raise Warning('The bump section is mighty small.')

    # set up the error landscape, error vs tau
    # The NI lags the VectorNav and the time shift is typically between 0 and
    # 0.5 seconds
    tauRange = np.linspace(0., .5, num=500)
    error = np.zeros_like(tauRange)
    for i, val in enumerate(tauRange):
        error[i] = sync_error(val, niBumpSec, vnBumpSec, timeBumpSec)

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

    tau  = fmin(sync_error, tau0, args=(niBumpSec, vnBumpSec, timeBumpSec))[0]

    print "This is what came out of the minimization:", tau

    # if the minimization doesn't do a good job, just use the tau0
    if np.abs(tau - tau0) > 0.01:
        tau = tau0
        print "Bad minimizer!! Using the guess, %f, instead." % tau

    return tau

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
        raise StandardError("Something's wrong with the unsizing")
    return oldvec

def size_array(arr, desiredShape):
    '''Returns a one or two dimensional array that has either been padded with
    nans or reduced in shape.

    Parameters
    ----------
    arr : ndarray, shape(n,) or shape(n,m)
    desiredShape : integer or tuple
        If arr is one dimensinal, then desired shape can be a positive integer,
        else it needs to be a tuple of two positive integers.

    '''

    # this only works for arrays up to dimension 2
    message = "size_array only works with arrays of dimension 1 or 2."
    if len(arr.shape) > 2:
        raise ValueError(message)

    try:
        desiredShape[1]
    except TypeError:
        desiredShape = (desiredShape, 1)

    try:
        arr.shape[1]
        arrShape = arr.shape
    except IndexError:
        arrShape = (arr.shape[0], 1)

    print desiredShape

    # first adjust the rows
    if desiredShape[0] > arrShape[0]:
        try:
            adjRows = np.ones((desiredShape[0], arrShape[1])) * np.nan
        except IndexError:
            adjRows = np.ones(desiredShape[0]) * np.nan
        adjRows[:arrShape[0]] = arr
    else:
        adjRows = arr[:desiredShape[0]]

    newArr = adjRows

    if desiredShape[1] > 1:
        # now adjust the columns
        if desiredShape[1] > arrShape[1]:
            newArr = np.ones((adjRows.shape[0], desiredShape[1])) * np.nan
            newArr[:, :arrShape[1]] = adjRows
        else:
            newArr = adjRows[:, :desiredShape[1]]

    return newArr

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
        nans = np.ones(m - nsamp) * np.nan
        newvec = np.append(vector, nans)
    elif m < nsamp:
        newvec = vector[:m]
    elif m == nsamp:
        newvec = vector
    else:
        raise StandardError("Vector sizing didn't work")
    return newvec

def fill_tables(datafile='InstrumentedBicycleData.h5',
                pathToData='../BicycleDAQ/data'):
    '''Adds all the data from the hdf5 files in the h5 directory to the tables.

    Parameters
    ----------
    datafile : string
        path to the main hdf5 file: InstrumentedBicycleData.h5

    '''

    # open the hdf5 file for appending
    data = tab.openFile(datafile, mode='a')

    print "Loading run data."
    # get the table
    datatable = data.root.data.datatable
    # get the row
    row = datatable.row
    # load the files from the h5 directory
    pathToRunH5 = os.path.join(pathToData, 'h5')
    files = sorted(os.listdir(pathToRunH5))

    # fill the rows with data
    for run in files:
        print 'Adding run: %s' % run
        rundata = get_run_data(os.path.join(pathToRunH5, run))
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

    print "Loading signal data."
    # fill in the signal table
    signaltable = data.root.data.signaltable
    row = signaltable.row

    # these are data signals that will be created from the raw data
    processedCols = ['FrameAccelerationX',
                     'FrameAccelerationY',
                     'FrameAccelerationZ',
                     'PitchRate',
                     'PullForce',
                     'RearWheelRate',
                     'RollAngle',
                     'RollRate',
                     'ForwardSpeed',
                     'SteerAngle',
                     'SteerRate',
                     'SteerTorque',
                     'tau',
                     'YawRate']

    # get two example runs
    filteredRun, unfilteredRun = get_two_runs(pathToRunH5)

    niCols = filteredRun['NICols']
    # combine the VNavCols from unfiltered and filtered
    vnCols = set(filteredRun['VNavCols'] + unfilteredRun['VNavCols'])

    for sig in set(niCols + list(vnCols) + processedCols):
        row['signal'] = sig

        if sig in niCols:
            row['source'] = 'NI'
            row['isRaw'] = True
            row['units'] = 'volts'
            if sig.startswith('FrameAccel') or sig == 'SteerRateGyro':
                row['calibration'] = 'bias'
            elif sig.endswith('Potentiometer'):
                row['calibration'] = 'interceptStar'
            elif sig in ['WheelSpeedMotor', 'SteerTorqueSensor',
                         'PullForceBridge']:
                row['calibration'] = 'intercept'
            elif sig[:-1].endswith('Bridge'):
                row['calibration'] = 'matrix'
            else:
                row['calibration'] = 'none'
        elif sig in vnCols:
            row['source'] = 'VN'
            row['isRaw'] = True
            row['calibration'] = 'none'
        elif sig in processedCols:
            row['source'] = 'NA'
            row['isRaw'] = False
            row['calibration'] = 'na'
        else:
            raise KeyError('{0} is not raw or processed'.format(sig))

        row.append()

    signaltable.flush()

    print "Loading calibration data."
    # fill in the calibration table
    calibrationtable = data.root.data.calibrationtable
    row = calibrationtable.row

    # load the files from the h5 directory
    pathToCalibH5 = os.path.join(pathToData, 'CalibData', 'h5')
    files = sorted(os.listdir(pathToCalibH5))

    for f in files:
        print "Calibration file:", f
        calibDict = get_calib_data(os.path.join(pathToCalibH5, f))
        for k, v in calibDict.items():
            if k in ['x', 'y', 'v']:
                row[k] = size_vector(v, 50)
            else:
                row[k] = v
        row.append()

    calibrationtable.flush()

    data.close()

    print datafile, "ready for action!"

def get_two_runs(pathToH5):
    '''Gets the data from both a filtered and unfiltered run.'''

    # load in the data files
    files = sorted(os.listdir(pathToH5))

    # get an example filtered and unfiltered run (wrt to the VN-100 data)
    filteredRun = get_run_data(os.path.join(pathToH5, files[0]))
    if filteredRun['par']['ADOT'] is not 14:
        raise ValueError('Run %d is not a filtered run, choose again' %
              filteredRun['par']['RunID'])

    unfilteredRun = get_run_data(os.path.join(pathToH5, files[-1]))
    if unfilteredRun['par']['ADOT'] is not 253:
        raise ValueError('Run %d is not a unfiltered run, choose again' %
              unfilteredRun['par']['RunID'])

    return filteredRun, unfilteredRun

def create_database(filename='InstrumentedBicycleData.h5',
                    pathToH5='../BicycleDAQ/data/h5'):
    '''Creates an HDF5 file for data collected from the instrumented
    bicycle.'''

    # open a new hdf5 file for writing
    data = tab.openFile(filename, mode='w',
                        title='Instrumented Bicycle Data')
    # create a group for the data
    rgroup = data.createGroup('/', 'data', 'Data')

    # generate the signal table description class
    SignalTable = create_signal_table_class()
    # add the signal table to the group
    sTable = data.createTable(rgroup, 'signaltable',
                              SignalTable, 'Signal Information')
    sTable.flush()

    # generate the calibration table description class
    CalibrationTable = create_calibration_table_class()
    # add the calibration table to the group
    cTable = data.createTable(rgroup, 'calibrationtable',
                              CalibrationTable, 'Calibration Information')
    cTable.flush()

    # get two example runs
    filteredRun, unfilteredRun = get_two_runs(pathToH5)
    # generate the table description class
    RunTable = create_run_table_class(filteredRun, unfilteredRun)
    # setup up a compression filter
    compression = tab.Filters(complevel=1, complib='zlib')
    # add the data table to this group
    rtable = data.createTable(rgroup, 'datatable',
                              RunTable, 'Run Data',
                              filters=compression)
    rtable.flush()

    data.close()

def create_signal_table_class():
    '''Creates a class that is used to describe the table containing
    information about the signals.'''

    class SignalTable(tab.IsDescription):
        calibration = tab.StringCol(20)
        isRaw = tab.BoolCol()
        sensor = tab.StringCol(20)
        signal = tab.StringCol(20)
        source = tab.StringCol(2)
        units = tab.StringCol(20)

    return SignalTable

def create_calibration_table_class():
    '''Creates a class that is used to describe the table containing the
    calibration data.'''

    class CalibrationTable(tab.IsDescription):
        accuracy = tab.StringCol(10)
        bias = tab.Float32Col(dflt=np.nan)
        calibrationID = tab.StringCol(5)
        calibrationSupplyVoltage = tab.Float32Col(dflt=np.nan)
        name = tab.StringCol(20)
        notes = tab.StringCol(500)
        offset = tab.Float32Col(dflt=np.nan)
        runSupplyVoltage = tab.Float32Col(dflt=np.nan)
        runSupplyVoltageSource = tab.StringCol(10)
        rsq = tab.Float32Col(dflt=np.nan)
        sensorType = tab.StringCol(20)
        signal = tab.StringCol(26)
        slope = tab.Float32Col(dflt=np.nan)
        timeStamp = tab.StringCol(21)
        units = tab.StringCol(20)
        v = tab.Float32Col(shape=(50,))
        x = tab.Float32Col(shape=(50,))
        y = tab.Float32Col(shape=(50,))

    return CalibrationTable

def create_run_table_class(filteredrun, unfilteredrun):
    '''Returns a class that is used for the table description for raw data
    for each run.

    Parameters
    ----------
    filteredrun : dict
        Contains the python dictionary of a run with filtered VN-100 data.
    unfilteredrun : dict
        Contains the python dictionary of a run with unfiltered VN-100 data.

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
            pos = k + 1 + i
            if isinstance(val, type(1)):
                exec(key + " = tab.Int64Col(pos=pos)")
            elif isinstance(val, type('')):
                exec(key + " = tab.StringCol(itemsize=200, pos=pos)")
            elif isinstance(val, type(1.)):
                exec(key + " = tab.Float64Col(pos=pos)")
            elif isinstance(val, type(np.ones(1))):
                exec(key + " = tab.Float64Col(shape=(" + str(len(val)) + ", ), pos=pos)")

        # add the columns for the processed data
        processedCols = ['FrameAccelerationX',
                         'FrameAccelerationY',
                         'FrameAccelerationZ',
                         'PitchRate',
                         'PullForce',
                         'RearWheelRate',
                         'RollAngle',
                         'RollRate',
                         'ForwardSpeed',
                         'SteerAngle',
                         'SteerRate',
                         'SteerTorque',
                         'tau',
                         'YawRate']

        for k, col in enumerate(processedCols):
            if col == 'tau':
                exec(col + " = tab.Float32Col(pos=i + 1 + k)")
            else:
                exec(col + " = tab.Float32Col(shape=(12000, ), pos=i + 1 + k)")

        # get rid intermediate variables so they are not stored in the class
        del(i, k, col, key, pos, val, processedCols)

    return RunTable

def replace_corrupt_strings_with_nan(vnOutput, vnCols):
    '''Returns a numpy matrix with the VN-100 output that has the corrupt
    values replaced by nan values.

    Parameters
    ----------
    vnOutput : list
        The list of output strings from an asyncronous reading from the VN-100.
    vnCols : list
        A list of the column names for this particular async output.

    Returns
    -------
    vnData : array (m, n)
        An array containing the corrected data values. n is the number of
        samples and m is the number of signals.

    '''

    vnData = []

    nanRow = [np.nan for i in range(len(vnCols))]

    # go through each sample in the vn-100 output
    for i, vnStr in enumerate(vnOutput):
        # parse the string
        vnList, chkPass, vnrrg = parse_vnav_string(vnStr)
        # if the checksum passed, then append the data unless vnList is not the
        # correct length, (this is because run139 sample 4681 seems to calculate the correct
        # checksum for an incorrect value)
        if chkPass and len(vnList[1:-1]) == len(vnCols):
            vnData.append([float(x) for x in vnList[1:-1]])
        # if not append some nan values
        else:
            if i == 0:
                vnData.append(nanRow)
            # there are typically at least two corrupted samples combine into
            # one
            else:
                vnData.append(nanRow)
                vnData.append(nanRow)

    # remove extra values so that the number of samples equals the number of
    # samples of the VN-100 output
    vnData = vnData[:len(vnOutput)]

    return np.transpose(np.array(vnData))

def parse_vnav_string(vnStr):
    '''Returns a list of the information in a VN-100 text string and whether
    the checksum failed.

    Parameters
    ----------
    vnStr : string
        A string from the VectorNav serial output (UART mode).

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
    #print('Checksum for the raw string is %s' % calcChkSum)

    # get rid of the $ and the newline characters
    vnMeat = re.sub('''(?x) # verbose
                       \$ # match the dollar sign at the beginning
                       (.*) # this is the content
                       \* # match the asterisk
                       (\w*) # the checksum
                       \s* # the newline characters''', r'\1,\2', vnStr)

    # make it a list with the last item being the checksum
    vnList = vnMeat.split(',')

    # set the vnrrg flag
    if vnList[0] == 'VNRRG':
        vnrrg = True
    else:
        vnrrg = False

    #print("Provided checksum is %s" % vnList[-1])
    # see if the checksum passes
    chkPass = calcChkSum == vnList[-1]
    if not chkPass:
        #print "Checksum failed"
        #print vnStr
        vnrrg = None

    # return the list, whether or not the checksum failed and whether or not it
    # is a VNRRG
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
        Two character hex representation of the checksum. The letters are
        capitalized and single digits have a leading zero.

    '''
    chkStr = re.sub('''(?x) # verbose
                       \$ # match the dollar sign
                       (.*) # match the stuff the checksum is calculated from
                       \* # match the asterisk
                       \w* # the checksum
                       \s* # the newline characters''', r'\1', vnStr)

    checksum = reduce(xor, map(ord, chkStr))
    # remove the first '0x'
    hexVal = hex(checksum)[2:]

    # if the hexVal is only a single digit, it needs a leading zero to match
    # the VN-100's output
    if len(hexVal) == 1:
        hexVal = '0' + hexVal

    # the letter's need to be capitalized to match too
    return hexVal.upper()

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

    # get the NIData
    rundata['NIData'] = runfile.root.NIData.read()

    # get the VN-100 data column names
    # make the array into a list of python string and gets rid of unescaped
    # control characters
    columns = [re.sub(r'[^ -~].*', '', str(x))
               for x in runfile.root.VNavCols.read()]
    # gets rid of white space
    rundata['VNavCols'] = [x.replace(' ', '') for x in columns]

    # get the NI column names
    # make a list of NI columns from the InputPair structure from matlab
    rundata['NICols'] = []
    for col in runfile.root.InputPairs:
        rundata['NICols'].append((str(col.name), int(col.read()[0])))

    rundata['NICols'].sort(key=lambda x: x[1])

    rundata['NICols'] = [x[0] for x in rundata['NICols']]

    # get the VNavDataText
    rundata['VNavDataText'] = [re.sub(r'[^ -~].*', '', str(x))
                               for x in runfile.root.VNavDataText.read()]

    # redefine the NIData using parsing that accounts for the corrupt values
    # better
    rundata['VNavData'] = replace_corrupt_strings_with_nan(
                           rundata['VNavDataText'],
                           rundata['VNavCols'])

    # close the file
    runfile.close()

    return rundata

def get_calib_data(pathToFile):
    '''
    Returns calibration data from the run h5 files using pytables and
    formats it as a dictionairy.

    Parameters
    ----------
    pathToFile : string
        The path to the h5 file that contains the calibration data, normally:
        pathtofile = '../BicycleDAQ/data/CalibData/h5/00000.h5'

    Returns
    -------
    calibData : dictionary
        A dictionary that looks similar to how the data was stored in Matlab.

    '''

    calibFile = tab.openFile(pathToFile)

    calibData = {}

    for thing in calibFile.root.data:
        if len(thing.read().flatten()) == 1:
            calibData[thing.name] = thing.read()[0]
        else:
            if thing.name in ['x', 'v']:
                try:
                    calibData[thing.name] = np.mean(thing.read(), 1)
                except ValueError:
                    calibData[thing.name] = thing.read()
            else:
                calibData[thing.name] = thing.read()

    calibFile.close()

    return calibData
