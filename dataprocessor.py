#!/usr/bin/env python

# built in imports
import os
import re
import datetime
from math import pi

# dependencies
from scipy.interpolate import UnivariateSpline
from scipy import io
import matplotlib.pyplot as plt

# local dependencies
from signalprocessing import *
from database import *
#from dtk.process import *
from bicycleparameters import bicycleparameters as bp

class Signal(np.ndarray):
    """
    A subclass of ndarray for collecting the data for a single signal in a run.

    Attributes
    ----------
    conversions : dictionary
        A mapping for unit conversions.
    name : str
        The name of the signal. Should be CamelCase.
    runid : str
        A five digit identification number associated with the
        trial this signal was collected from (e.g. '00104').
    sampleRate : float
        The sample rate in hertz of the signal.
    source : str
        The source of the data. This should be 'NI' for the
        National Instruments USB-6218 and 'VN' for the VN-100 IMU.
    units : str
        The physcial units of the signal. These should be specified
        as lowercase complete words using only multiplication and
        division symbols (e.g. 'meter/second/second').
        Signal.conversions will show the avialable options.

    Methods
    -------
    plot()
        Plot's the signal versus time and returns the line.
    frequency()
        Returns the frequency spectrum of the signal.
    time_derivative()
        Returns the time derivative of the signal.
    filter(frequency)
        Returns the low passed filter of the signal.
    truncate(tau)
        Interpolates and truncates the signal the based on the time shift,
        `tau`, and the signal source.
    as_dictionary
        Returns a dictionary of the metadata of the signal.
    convert_units(units)
        Returns a signal with different units. `conversions` specifies the
        available options.

    """

    # define some basic unit converions
    conversions = {'degree->radian': pi / 180.,
                   'inch*pound->newton*meter': 25.4 / 1000. * 4.44822162,
                   'pound->newton': 4.44822162,
                   'degree/second->radian/second': pi / 180.,
                   'feet/second->meter/second': 12. * 2.54 / 100.,
                   'mile/hour->meter/second': 0.00254 * 12. / 5280. / 3600.,
                   'degree/second/second->radian/second/second': pi / 180.}

    def __new__(cls, inputArray, metadata):
        """
        Returns an instance of the Signal class with the addition signal
        data.

        Parameters
        ----------
        inputArray : ndarray, shape(n,)
            A one dimension array representing a single variable's time
            history.
        metadata : dictionary
            This dictionary contains the metadata for the signal.
                name : str
                    The name of the signal. Should be CamelCase.
                runid : str
                    A five digit identification number associated with the
                    trial this experiment was collected at (e.g. '00104').
                sampleRate : float
                    The sample rate in hertz of the signal.
                source : str
                    The source of the data. This should be 'NI' for the
                    National Instruments USB-6218 and 'VN' for the VN-100 IMU.
                units : str
                    The physcial units of the signal. These should be specified
                    as lowercase complete words using only multiplication and
                    division symbols (e.g. 'meter/second/second').
                    Signal.conversions will show the avialable options.

        Raises
        ------
        ValueError
            If `inputArray` is not a vector.

        """
        if len(inputArray.shape) > 1:
            raise ValueError('Signals must be arrays of one dimension.')
        # cast the input array into the Signal class
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
        # doesn't support these things in basic ufunc calls...maybe one day
        # That means anytime you add, subtract, multiply, divide, etc, the
        # following are not retained.
        outputArray.name = None
        outputArray.source = None
        outputArray.units = None
        return np.ndarray.__array_wrap__(self, outputArray, context)

    def plot(self, show=True):
        '''Plots and returns the signal versus time.'''
        time = self.time()
        line = plt.plot(time, self)
        plt.xlabel('Time [second]')
        plt.ylabel('{0} [{1}]').format(self.name, self.units)
        plt.title('Signal plot during run {0}'.format(self.runid))
        if show:
            plt.show()
        return line

    def frequency(self):
        '''Returns the frequency content of the signal.'''
        return freq_spectrum(self, self.sampleRate)

    def time_derivative(self):
        '''Returns the time derivative of the signal.'''
        time = self.time()
        # caluculate the numerical time derivative
        dsdt = derivative(time, self, method='combination')
        # map the metadeta from self onto the derivative
        dsdt = Signal(dsdt, self.as_dictionary())
        dsdt.name = dsdt.name + 'Dot'
        dsdt.units = dsdt.units + '/second'
        return dsdt

    def time(self):
        """Returns the time vector of the signal."""
        return time_vector(len(self), self.sampleRate)

    def filter(self, frequency):
        '''Returns the signal filter by a low pass Butterworth at the given
        frequency.'''

        # If the signal has nans, then we need to spline fit the data before
        # filtering it. A masked numpy array may be a better option.
        if np.isnan(self).any():
            time = self.time()
            # remove the nan's in the signal and the time
            t = time[np.nonzero(np.isnan(self) == False)]
            v = self[np.nonzero(np.isnan(self) == False)]
            # fit a spline through the data
            splineObject = UnivariateSpline(t, v, k=3, s=0)
            # get the data from the spline function for time
            splined = splineObject(time)
        else:
            splined = self

        filteredArray = butterworth(splined, frequency, self.sampleRate)

        return Signal(filteredArray, self.as_dictionary())

    def truncate(self, tau):
        '''Returns the shifted and truncated signal based on the provided
        timeshift, tau.'''
        return truncate_data(self, tau)

    def as_dictionary(self):
        '''Returns the signal metadata as a dictionary.'''
        data = {'runid': self.runid,
                'name': self.name,
                'units': self.units,
                'source': self.source,
                'sampleRate': self.sampleRate}
        return data

    def convert_units(self, units):
        """
        Returns a signal with the specified units.

        Parameters
        ----------
        units : str
            The units to convert the signal to. The mapping must be in the
            attribute `conversions`.

        Returns
        -------
        newSig : Signal
            The signal with the desired units.

        """
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
    """
    A subclass of Signal for collecting the data for a single raw signal in
    a run.

    Attributes
    ----------
    sensor : Sensor
        Each raw signal has a sensor associated with it. Most sensors contain
        calibration data for that sensor/signal.
    calibrationType :

    Notes
    -----
    This is a class for the signals that are the raw measurement outputs
    collected by the BicycleDAQ software and are already stored in the pytables
    database file.

    """

    def __new__(cls, runid, signalName, database):
        """
        Returns an instance of the RawSignal class with the additional signal
        metadata.

        Parameters
        ----------
        runid : str
            A five digit
        signalName : str
            A CamelCase signal name that corresponds to the raw signals output
            by BicycleDAQ_.
        database : pytables object
            The hdf5 database for the instrumented bicycle.

        .. _BicycleDAQ: https://github.com/moorepants/BicycleDAQ

        """

        # get the tables
        dTab = database.root.data.datatable
        sTab = database.root.data.signaltable
        cTab = database.root.data.calibrationtable

        # get the row number for this particular run id
        rownum = get_row_num(runid, dTab)
        signal = get_cell(dTab, signalName, rownum)

        # cast the input array into my subclass of ndarray
        obj = np.asarray(signal).view(cls)

        obj.runid = runid
        obj.timeStamp = matlab_date_to_object(get_cell(dTab, 'DateTime',
            rownum))
        obj.calibrationType, obj.units, obj.source = [(row['calibration'],
            row['units'], row['source'])
            for row in sTab.where('signal == signalName')][0]
        obj.name = signalName

        try:
            obj.sensor = Sensor(obj.name, cTab)
        except KeyError:
            print "There is no sensor named {0}.".format(signalName)

        # this assumes that the supply voltage for this signal is the same for
        # all sensor calibrations
        try:
            supplySource = [row['runSupplyVoltageSource']
                           for row in cTab.where('name == signalName')][0]
            if supplySource == 'na':
                obj.supply = [row['runSupplyVoltage']
                               for row in cTab.where('name == signalName')][0]
            else:
                obj.supply = get_cell(dTab, supplySource, rownum)
        except IndexError:
            print "{0} does not have a supply voltage.".format(signalName)
            print "-" * 79

        # get the appropriate sample rate
        if obj.source == 'NI':
            sampRateCol = 'NISampleRate'
        elif obj.source == 'VN':
            sampRateCol = 'VNavSampleRate'
        else:
            raise ValueError('{0} is not a valid source.'.format(obj.source))

        obj.sampleRate = dTab[rownum][dTab.colnames.index(sampRateCol)]

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.calibrationType = getattr(obj, 'calibrationType', None)
        self.name = getattr(obj, 'name', None)
        self.runid = getattr(obj, 'runid', None)
        self.sampleRate = getattr(obj, 'sampleRate', None)
        self.sensor = getattr(obj, 'sensor', None)
        self.source = getattr(obj, 'source', None)
        self.units = getattr(obj, 'units', None)
        self.timeStamp = getattr(obj, 'timeStamp', None)

    def __array_wrap__(self, outputArray, context=None):
        # doesn't support these things in basic ufunc calls...maybe one day
        outputArray.calibrationType = None
        outputArray.name = None
        outputArray.sensor = None
        outputArray.source = None
        outputArray.units = None
        return np.ndarray.__array_wrap__(self, outputArray, context)

    def scale(self):
        """
        Returns the scaled signal based on the calibration data for the
        supplied date.

        Returns
        -------
        : ndarray (n,)
            Scaled signal.

        """
        try:
            self.calibrationType
        except AttributeError:
            raise AttributeError("Can't scale without the calibration type")

        # these will need to be changed once we start measuring them
        doNotScale = ['LeanPotentiometer',
                      'HipPotentiometer',
                      'TwistPotentiometer']
        if self.calibrationType in ['none', 'matrix'] or self.name in doNotScale:
            print "Not scaling {0}".format(self.name)
            return self
        else:
            print "Scaling {0}".format(self.name)

            # pick the largest calibration date without surpassing the run date
            calibData = self.sensor.get_data_for_date(self.timeStamp)

            slope = calibData['slope']
            bias = calibData['bias']
            intercept = calibData['offset']
            calibrationSupplyVoltage = calibData['calibrationSupplyVoltage']

            print "slope {0}, bias {1}, intercept {2}".format(slope, bias,
                    intercept)

            if self.calibrationType == 'interceptStar':
                # this is for potentiometers, where the slope is ratiometric
                # and zero degrees is always zero volts
                calibratedSignal = (calibrationSupplyVoltage / self.supply *
                                    slope * self + intercept)
            elif self.calibrationType == 'intercept':
                # this is the typical calibration that I use for all the
                # sensors that I calibrate myself
                calibratedSignal = (calibrationSupplyVoltage / self.supply *
                                    (slope * self + intercept))
            elif self.calibrationType == 'bias':
                # this is for the accelerometers and rate gyros that are
                # "ratiometric", but I'm still not sure this is correct
                calibratedSignal = (slope * (self - self.supply /
                                    calibrationSupplyVoltage * bias))
            else:
                raise StandardError("None of the calibration equations worked.")
            calibratedSignal.name = calibData['signal']
            calibratedSignal.units = calibData['units']
            calibratedSignal.source = self.source

            return calibratedSignal.view(Signal)

    def plot_scaled(self, show=True):
        '''Plots and returns the scaled signal versus time.'''
        time = self.time()
        scaled = self.scale()
        line = plt.plot(time, scaled[1])
        plt.xlabel('Time [s]')
        plt.ylabel(scaled[2])
        plt.title('{0} signal during run {1}'.format(scaled[0],
                  str(self.runid)))
        if show:
            plt.show()
        return line

class Sensor():
    """This class is a container for calibration data for a sensor."""

    def __init__(self, name, calibrationTable):
        """
        Initializes this sensor class.

        Parameters
        ----------
        name : string
            The CamelCase name of the sensor (e.g. SteerTorqueSensor).
        calibrationTable : pyTables table object
            This is the calibration data table that contains all the data taken
            during calibrations.

        """
        self.name = name
        self.store_calibration_data(calibrationTable)

    def store_calibration_data(self, calibrationTable):
        """
        Stores a dictionary of calibration data for the sensor for all
        calibration dates in the object.

        Parameters
        ----------
        calibrationTable : pyTables table object
            This is the calibration data table that contains all the data taken
            during calibrations.

        """
        self.data = {}

        for row in calibrationTable.iterrows():
            if self.name == row['name']:
                self.data[row['calibrationID']] = {}
                for col in calibrationTable.colnames:
                    self.data[row['calibrationID']][col] = row[col]

        if self.data == {}:
            raise KeyError(('{0} is not a valid sensor ' +
                           'name').format(self.name))

    def get_data_for_date(self, runDate):
        """
        Returns the calibration data for the sensor for the most recent
        calibration relative to `runDate`.

        Parameters
        ----------
        runDate : datetime object
            This is the date of the run that the calibration data is needed
            for.

        Returns
        -------
        calibData : dictionary
            A dictionary containing the sensor calibration data for the
            calibration closest to but not past `runDate`.

        Notes
        -----
        This method will select the calibration data for the date closest to
        but not past `runDate`. **All calibrations must be taken before the
        runs.**

        """
        # make a list of calibration ids and time stamps
        dateIdPairs = [(k, matlab_date_to_object(v['timeStamp']))
                       for k, v in self.data.iteritems()]
        # sort the pairs with the most recent date first
        dateIdPairs.sort(key=lambda x: x[1], reverse=True)
        # go through the list and return the index at which the calibration
        # date is larger than the run date
        for i, pair in enumerate(dateIdPairs):
            if runDate >= pair[1]:
                break
        return self.data[dateIdPairs[i][0]]

class Run():
    """The fundamental class for a run."""

    def __init__(self, runid, database, forceRecalc=False):
        """
        Loads all the data for a run if available otherwise it generates the
        data from the raw data.

        Parameters
        ----------
        runid : int or str
            The run id should be an integer, e.g. 5, or a five digit string with
            leading zeros, e.g. '00005'.
        database : pytable object of an hdf5 file
            This file must contain the run data table and the calibration data
            table.
        forceRecalc : boolean
            If true then it will force a recalculation of all the the non raw
            data.

        """

        print "Initializing the run object."
        # get the tables
        dataTable = database.root.data.datatable
        signalTable = database.root.data.signaltable

        # get the row number for this particular run id
        rownum = get_row_num(runid, dataTable)

        # make some dictionaries to store all the data
        self.metadata = {}
        self.rawSignals = {}
        self.computedSignals ={}

        # make lists of the input and output signals
        rawDataCols = [x['signal'] for x in
                       signalTable.where("isRaw == True")]
        computedCols = [x['signal'] for x in
                        signalTable.where("isRaw == False")]

        # store the current data for this run
        print "Loading metadata from the database."
        for col in dataTable.colnames:
            if col not in (rawDataCols + computedCols):
                self.metadata[col] = get_cell(dataTable, col, rownum)

        # tell the user about the run
        print self

        print "Loading the raw signals from the database."
        for col in rawDataCols:
            self.rawSignals[col] = RawSignal(runid, col, database)

        print "Loading the bicycle and rider data."
        # load the parameters for the bicycle
        # this code will not work for other bicycle/rider combinations. it will
        # need to be updated
        bicycles = {'Rigid Rider': 'Rigid'}
        pathToBicycles = '/media/Data/Documents/School/UC Davis/Bicycle Mechanics/BicycleParameters/data/bicycles'
        rigid = bp.Bicycle(bicycles[self.metadata['Bicycle']], pathToBicycles=pathToBicycles)
        pathToRider = '/media/Data/Documents/School/UC Davis/Bicycle Mechanics/BicycleParameters/data/riders'
        rigid.add_rider(pathToRider=pathToRider + '/Jason/JasonRigidBenchmark.txt')
        self.bikeParameters = bp.remove_uncertainties(rigid.parameters['Benchmark'])

        if forceRecalc == True:
            print "Computing signals from raw data."

            self.calibratedSignals = {}
            self.truncatedSignals = {}

            # calibrate the signals for the run
            for sig in self.rawSignals.values():
                calibSig = sig.scale()
                self.calibratedSignals[calibSig.name] = calibSig

            # calculate tau for this run
            self.tau = find_timeshift(
                self.calibratedSignals['AccelerometerAccelerationY'],
                self.calibratedSignals['AccelerationZ'],
                self.metadata['NISampleRate'],
                self.metadata['Speed'])

            # truncate all the raw data signals
            for name, sig in self.calibratedSignals.items():
                self.truncatedSignals[name] = sig.truncate(self.tau)

            # compute the final output signals
            noChange = ['FiveVolts',
                        'PushButton',
                        'RearWheelRate',
                        'RollAngle',
                        'SteerAngle',
                        'ThreeVolts']
            for sig in noChange:
                if sig in ['RollAngle', 'SteerAngle']:
                    self.computedSignals[sig] =\
                    self.truncatedSignals[sig].convert_units('radian')
                else:
                    self.computedSignals[sig] = self.truncatedSignals[sig]

            # the pull force was always from the left side, so far...
            pullForce = -self.truncatedSignals['PullForce']
            pullForce.name = self.truncatedSignals['PullForce'].name
            pullForce.units = self.truncatedSignals['PullForce'].units
            self.computedSignals['PullForce'] =\
            pullForce.convert_units('newton')

            self.computedSignals['ForwardSpeed'] =\
                (self.bikeParameters['rR'] *
                self.truncatedSignals['RearWheelRate'])
            self.computedSignals['ForwardSpeed'].units = 'meter/second'
            self.computedSignals['ForwardSpeed'].name = 'ForwardSpeed'

            self.computedSignals['SteerRate'] =\
                steer_rate(self.truncatedSignals['ForkRate'],
                self.truncatedSignals['AngularRateZ'].\
                        convert_units('radian/second'))
            self.computedSignals['SteerRate'].units = 'radian/second'
            self.computedSignals['SteerRate'].name = 'SteerRate'

            yr, rr, pr = yaw_roll_pitch_rate(
                    self.truncatedSignals['AngularRateX'].convert_units('radian/second'),
                    self.truncatedSignals['AngularRateY'].convert_units('radian/second'),
                    self.truncatedSignals['AngularRateZ'].convert_units('radian/second'),
                    self.bikeParameters['lam'],
                    rollAngle=self.truncatedSignals['RollAngle'].convert_units('radian'))
            yr.units = 'radian/second'
            yr.name = 'YawRate'
            rr.units = 'radian/second'
            rr.name = 'RollRate'
            pr.units = 'radian/second'
            pr.name = 'PitchRate'

            self.computedSignals['YawRate'] = yr
            self.computedSignals['RollRate'] = rr
            self.computedSignals['PitchRate'] = pr

            steerTorque = steer_torque(
                self.computedSignals['SteerRate'],
                self.computedSignals['SteerRate'].time_derivative(),
                self.truncatedSignals['SteerTubeTorque'].convert_units('newton*meter'),
                rigid.steer_assembly_moment_of_inertia(fork=False,
                    wheel=False).nominal_value,
                0.3475, 0.0861)
            steerTorque.units = 'newton*meter'
            steerTorque.name = 'SteerTorque'
            self.computedSignals['SteerTorque'] = steerTorque
        else:
            # else just get the values stored in the database
            print "Loading computed signals from database."
            for col in computedCols:
                self.computedSignals[col] = RawSignal(runid, col, datafile)

    def export(self, filetype):
        """
        Exports the computed signals to a file.

        Parameters
        ----------
        filetype : str
            The type of file to export the data to. Options are 'mat', 'csv',
            and 'pickle'.

        """

        if filetype == 'mat':
            exportData = {}
            exportData.update(self.metadata)
            exportData.update(self.computedSignals)
            exportData.update(self.bikeParameters)
            filename = pad_with_zeros(str(self.metadata['RunID']), 5) + '.mat'
            io.savemat(filename, exportData)
        else:
            raise NotImplementedError(('{0} method is not available' +
                                      ' yet.').format(filetype))

    def plot(self, *args, **kwargs):
        '''
        Plots the time series of various signals.

        Parameters
        ----------
        signalName : string
            These should be strings that correspond to processed data
            columns.
        signalType : string, optional
            This allows you to plot from the various signal types. Options are
            'computed', 'truncated', 'calibrated', 'raw'.

        '''
        if not kwargs:
            kwargs = {'signalType': 'computed'}

        # this currently only works if the sample rates from both sources is
        # the same
        sampleRate = self.metadata['NISampleRate']

        mapping = {'computed': self.computedSignals,
                   'truncated': self.truncatedSignals,
                   'calibrated': self.calibratedSignals,
                   'raw': self.rawSignals}

        for i, arg in enumerate(args):
            signal = mapping[kwargs['signalType']][arg]
            time = time_vector(len(signal), sampleRate)
            plt.plot(time, signal)

        plt.legend([arg + ' [' + mapping[kwargs['signalType']][arg].units + ']' for arg in args])

        plt.title('Rider: ' + self.metadata['Rider'] +
                  ', Speed: ' + str(self.metadata['Speed']) + 'm/s' +
                  ', Maneuver: ' + self.metadata['Maneuver'] +
                  ', Environment: ' + self.metadata['Environment'] + '\n' +
                  'Notes: ' + self.metadata['Notes'])

        plt.xlabel('Time [second]')

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
