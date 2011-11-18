#!/usr/bin/env python

# built in imports
import os
import datetime
from math import pi

# dependencies
import numpy as np
from scipy import io
from scipy.integrate import cumtrapz
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# local dependencies
from database import *
from signalprocessing import *
from dtk.process import spline_over_nan, derivative, butterworth, freq_spectrum
import bicycleparameters as bp

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
                   'degree/second->radian/second': pi / 180.,
                   'degree/second/second->radian/second/second': pi / 180.,
                   'inch*pound->newton*meter': 25.4 / 1000. * 4.44822162,
                   'pound->newton': 4.44822162,
                   'feet/second->meter/second': 12. * 2.54 / 100.,
                   'mile/hour->meter/second': 0.00254 * 12. / 5280. / 3600.}

    def __new__(cls, inputArray, metadata):
        """
        Returns an instance of the Signal class with the additional signal
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
        if units == self.units:
            return self
        else:
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
            newSig = Signal(newSig, self.as_dictionary())
            newSig.units = units

            return newSig

    def filter(self, frequency):
        """Returns the signal filtered by a low pass Butterworth at the given
        frequency."""
        filteredArray = butterworth(self.spline(), frequency, self.sampleRate)
        return Signal(filteredArray, self.as_dictionary())

    def frequency(self):
        """Returns the frequency content of the signal."""
        return freq_spectrum(self.spline(), self.sampleRate)

    def integrate(self, initialCondition=0., subtractMean=False):
        """Integrates the signal using the trapezoidal rule."""
        time = self.time()
        # integrate using trapz and adjust with the initial condition
        grated = np.hstack((0., cumtrapz(self, x=time))) + initialCondition
        # this tries to characterize the drift in the integrated signal. It
        # works well for signals from straight line tracking but not
        # necessarily for lange change.
        if subtractMean:
            def line(x, a, b, c):
                return a * x**2 + b * x + c
            popt, pcov = curve_fit(line, time, grated)
            grated = grated - line(time, popt[0], popt[1], popt[2])
        grated = Signal(grated, self.as_dictionary())
        grated.units = self.units + '*second'
        grated.name = self.name + 'Int'
        return grated

    def plot(self, show=True):
        """Plots and returns the signal versus time."""
        time = self.time()
        line = plt.plot(time, self)
        if show:
            plt.xlabel('Time [second]')
            plt.ylabel('{0} [{1}]'.format(self.name, self.units))
            plt.title('Signal plot during run {0}'.format(self.runid))
            plt.show()
        return line

    def spline(self):
        """Returns the signal with nans replaced by the resultes of a cubic
        spline."""
        splined = spline_over_nan(self.time(), self)
        return Signal(splined, self.as_dictionary())

    def time(self):
        """Returns the time vector of the signal."""
        return time_vector(len(self), self.sampleRate)

    def time_derivative(self):
        """Returns the time derivative of the signal."""
        # caluculate the numerical time derivative
        dsdt = derivative(self.time(), self, method='combination')
        # map the metadata from self onto the derivative
        dsdt = Signal(dsdt, self.as_dictionary())
        dsdt.name = dsdt.name + 'Dot'
        dsdt.units = dsdt.units + '/second'
        return dsdt

    def truncate(self, tau):
        '''Returns the shifted and truncated signal based on the provided
        timeshift, tau.'''
        # this is now an ndarray instead of a Signal
        return Signal(truncate_data(self, tau), self.as_dictionary())

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
        self._store_calibration_data(calibrationTable)

    def _store_calibration_data(self, calibrationTable):
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

    def __init__(self, runid, database, pathToParameterData, forceRecalc=True, filterSigs=False):
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
        pathToParameterData : string
            The is the path to the data directory for the BicycleParameters
            package.
        forceRecalc : boolean, optional
            If true then it will force a recalculation of all the the non-raw
            data.
        filterSigs : boolean, optional
            If true the computed signals will be low pass filtered.

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

        # store the metadata for this run
        print "Loading metadata from the database."
        for col in dataTable.colnames:
            if col not in (rawDataCols + computedCols):
                self.metadata[col] = get_cell(dataTable, col, rownum)

        # tell the user about the run
        print self

        print "Loading the raw signals from the database."
        for col in rawDataCols:
            self.rawSignals[col] = RawSignal(runid, col, database)

        print("Loading the bicycle and rider data for " +
              "{} on {}".format(self.metadata['Rider'],
              self.metadata['Bicycle']))
        self.load_rider(pathToParameterData)

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
                self.metadata['Speed'], plotError=False)

            # truncate and spline all of the calibrated signals
            for name, sig in self.calibratedSignals.items():
                self.truncatedSignals[name] = sig.truncate(self.tau).spline()

            # transfer some of the signals to computed
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

            # compute the quantities that we are interested in
            self.compute_pull_force()
            self.compute_forward_speed()
            self.compute_steer_rate()
            self.compute_yaw_roll_pitch_rates()
            self.compute_steer_torque()
            self.compute_yaw_angle()
            self.compute_wheel_contact_rates()
            self.compute_wheel_contact_points()

            if filterSigs:
                # filter all the computed signals
                for k, v in self.computedSignals.items():
                    self.computedSignals[k] = v.filter(30.)
        else:
            # else just get the values stored in the database
            print "Loading computed signals from database."
            for col in computedCols:
                self.computedSignals[col] = RawSignal(runid, col, datafile)

    def compute_wheel_contact_points(self):
        """Computes the location of the wheel contact points in the ground
        plane."""

        # get the rates
        try:
            latRate = self.computedSignals['LateralRearContactRate']
            lonRate = self.computedSignals['LongitudinalRearContactRate']
        except AttributeError:
            print('At least one of the rates are not available. ' +
                  'The YawAngle was not computed.')
        else:
            # convert to radians per second
            latRate = latRate.convert_units('meter/second')
            lonRate = lonRate.convert_units('meter/second')
            # integrate and try to account for the drift
            lat = latRate.integrate(subtractMean=True)
            lon = lonRate.integrate()
            # set the new name and units
            lat.name = 'LateralRearContact'
            lat.units = 'meter'
            lon.name = 'LongitudinalRearContact'
            lon.units = 'meter'
            # store in computed signals
            self.computedSignals[lat.name] = lat
            self.computedSignals[lon.name] = lon

    def compute_wheel_contact_rates(self):
        """Calculates the rates of the wheel contact points in the ground
        plane."""

        try:
            yawAngle = self.computedSignals['YawAngle']
            rearWheelRate = self.truncatedSignals['RearWheelRate']
            rR = self.bicycleRiderParameters['rR'] # this should be in meters
        except AttributeError:
            print('Either the yaw angle, rear wheel rate or ' +
                  'front wheel radius is not available. The ' +
                  'contact rates were not computed.')
        else:
            yawAngle = yawAngle.convert_units('radian')
            rearWheelRate = rearWheelRate.convert_units('radian/second')

            lon, lat = rear_wheel_contact_rate(rR, rearWheelRate, yawAngle)

            lon.name = 'LongitudinalRearContactRate'
            lon.units = 'meter/second'
            self.computedSignals[lon.name] = lon

            lat.name = 'LateralRearContactRate'
            lat.units = 'meter/second'
            self.computedSignals[lat.name] = lat

    def compute_yaw_angle(self):
        """Computes the yaw angle by integrating the yaw rate."""

        # get the yaw rate
        try:
            yawRate = self.computedSignals['YawRate']
        except AttributeError:
            print('YawRate is not available. The YawAngle was not computed.')
        else:
            # convert to to radians per second
            yawRate = yawRate.convert_units('radian/second')
            # integrate and try to account for the drift
            yawAngle = yawRate.integrate(subtractMean=True)
            # set the new name and units
            yawAngle.name = 'YawAngle'
            yawAngle.units = 'radian'
            # store in computed signals
            self.computedSignals['YawAngle'] = yawAngle

    def compute_steer_torque(self, plotComponents=False):
        # steer torque
        try:
            frameAngRate = np.vstack((
                self.truncatedSignals['AngularRateX'],
                self.truncatedSignals['AngularRateY'],
                self.truncatedSignals['AngularRateZ']))
            frameAngAccel = np.vstack((
                self.truncatedSignals['AngularRateX'].time_derivative(),
                self.truncatedSignals['AngularRateY'].time_derivative(),
                self.truncatedSignals['AngularRateZ'].time_derivative()))
            frameAccel = np.vstack((
                self.truncatedSignals['AccelerationX'],
                self.truncatedSignals['AccelerationY'],
                self.truncatedSignals['AccelerationZ']))
            handlebarAngRate = self.truncatedSignals['ForkRate']
            handlebarAngAccel = self.truncatedSignals['ForkRate'].time_derivative()
            steerAngle = self.truncatedSignals['SteerAngle']
            steerColumnTorque =\
                self.truncatedSignals['SteerTubeTorque'].convert_units('newton*meter')
            handlebarMass = self.bicycleRiderParameters['mG']
            handlebarInertia =\
                self.bicycle.steer_assembly_moment_of_inertia(fork=False,
                    wheel=False, nominal=True)
            # this is the distance from the handlebar center of mass to the
            # steer axis
            w = self.bicycleRiderParameters['w']
            c = self.bicycleRiderParameters['c']
            lam = self.bicycleRiderParameters['lam']
            xG = self.bicycleRiderParameters['xG']
            zG = self.bicycleRiderParameters['zG']
            handlebarCoM = np.array([xG, 0., zG])
            d = bp.geometry.distance_to_steer_axis(w, c, lam,
                    handlebarCoM)
            # these are the distances from the point on the steer axis which is
            # aligned with the handlebar center of mass to the accelerometer on
            # the frame
            ds1 = self.bicycle.parameters['Measured']['ds1']
            ds3 = self.bicycle.parameters['Measured']['ds3']
            ds = np.array([ds1, 0., ds3]) # i measured these
            # damping and friction values come from Peter's work, I need to verify
            # them still
            damping = 0.3475
            friction = 0.0861
        except AttributeError:
            print('All signals were not available. SteerTorque was not ' +
                  'computed.')
        else:
            steerTorque = steer_torque(
                              frameAngRate,
                              frameAngAccel,
                              frameAccel,
                              handlebarAngRate,
                              handlebarAngAccel,
                              steerAngle,
                              steerColumnTorque,
                              handlebarMass,
                              handlebarInertia,
                              damping,
                              friction,
                              d,
                              ds,
                              plot=plotComponents)
            stDict = {'units':'newton*meter',
                      'name':'SteerTorque',
                      'runid':self.metadata['RunID'],
                      'sampleRate':steerAngle.sampleRate,
                      'source':'NA'}
            self.computedSignals['SteerTorque'] = Signal(steerTorque, stDict)

    def compute_yaw_roll_pitch_rates(self):
        """Computes the yaw, roll and pitch rates of the bicycle frame."""

        try:
            omegaX = self.truncatedSignals['AngularRateX']
            omegaY = self.truncatedSignals['AngularRateY']
            omegaZ = self.truncatedSignals['AngularRateZ']
            rollAngle = self.truncatedSignals['RollAngle']
            lam = self.bicycleRiderParameters['lam']
        except AttributeError:
            print('All needed signals are not available. ' +
                  'Yaw, roll and pitch rates were not computed.')
        else:
            omegaX = omegaX.convert_units('radian/second')
            omegaY = omegaY.convert_units('radian/second')
            omegaZ = omegaZ.convert_units('radian/second')
            rollAngle = rollAngle.convert_units('radian')

            yr, rr, pr = yaw_roll_pitch_rate(omegaX, omegaY, omegaZ, lam,
                                             rollAngle=rollAngle)
            yr.units = 'radian/second'
            yr.name = 'YawRate'
            rr.units = 'radian/second'
            rr.name = 'RollRate'
            pr.units = 'radian/second'
            pr.name = 'PitchRate'

            self.computedSignals['YawRate'] = yr
            self.computedSignals['RollRate'] = rr
            self.computedSignals['PitchRate'] = pr

    def compute_steer_rate(self):
        """Calculate the steer rate from the frame and fork rates."""
        try:
            forkRate = self.truncatedSignals['ForkRate']
            omegaZ = self.truncatedSignals['AngularRateZ']
        except AttributeError:
            print('ForkRate or AngularRateZ is not available. ' +
                  'SteerRate was not computed.')
        else:
            forkRate = forkRate.convert_units('radian/second')
            omegaZ = omegaZ.convert_units('radian/second')

            steerRate = steer_rate(forkRate, omegaZ)
            steerRate.units = 'radian/second'
            steerRate.name = 'SteerRate'
            self.computedSignals['SteerRate'] = steerRate

    def compute_forward_speed(self):
        """Calculates the magnitude of the main component of velocity of the
        center of the rear wheel."""

        try:
            rR = self.bicycleRiderParameters['rR']
            rearWheelRate = self.truncatedSignals['RearWheelRate']
        except AttributeError:
            print('rR or RearWheelRate is not availabe. ' +
                  'ForwardSpeed was not computed.')
        else:
            rearWheelRate = rearWheelRate.convert_units('radian/second')

            self.computedSignals['ForwardSpeed'] = rR * rearWheelRate
            self.computedSignals['ForwardSpeed'].units = 'meter/second'
            self.computedSignals['ForwardSpeed'].name = 'ForwardSpeed'

    def compute_pull_force(self):
        """
        Computes the pull force from the truncated pull force signal.

        """
        try:
            pullForce = self.truncatedSignals['PullForce']
        except AttributeError:
            print 'PullForce was not available. PullForce was not computed.'
        else:
            pullForce = pullForce.convert_units('newton')
            pullForce.name = 'PullForce'
            pullForce.units = 'newton'
            self.computedSignals[pullForce.name] = pullForce

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

    def export(self, filetype, directory='exports'):
        """
        Exports the computed signals to a file.

        Parameters
        ----------
        filetype : str
            The type of file to export the data to. Options are 'mat', 'csv',
            and 'pickle'.

        """

        if filetype == 'mat':
            fullDir = os.path.join(directory, filetype)
            if not os.path.exists(fullDir):
                print "Creating {0}".format(fullDir)
                os.makedirs(fullDir)
            exportData = {}
            exportData.update(self.metadata)
            exportData.update(self.computedSignals)
            exportData.update(self.bicycleRiderParameters)
            filename = pad_with_zeros(str(self.metadata['RunID']), 5) + '.mat'
            io.savemat(os.path.join(fullDir, filename), exportData)
        else:
            raise NotImplementedError(('{0} method is not available' +
                                      ' yet.').format(filetype))

    def load_rider(self, pathToParameterData):
        """Creates a bicycle/rider attribute which contains the physical
        parameters for the bicycle and rider for this run."""

        # currently this isn't very generic, it only assumes that there was
        # Luke, Jason, and Charlie riding on the instrumented bicycle.
        rider = self.metadata['Rider']
        if rider == 'Charlie' or rider == 'Luke':
            # Charlie and Luke rode the bike in the same configuration
            bicycle = 'Rigidcl'
        elif rider == 'Jason' :
            bicycle = 'Rigid'
        else:
            raise StandardError('There are no bicycle parameters ' +
                    'for {}'.format(rider))

        # force a recalculation (but not the period calcs, they take too long)
        self.bicycle = bp.Bicycle(bicycle, pathToData=pathToParameterData,
                forceRawCalc=True)
        # force a recalculation of the human parameters
        self.bicycle.add_rider(rider, reCalc=True)
        self.bicycleRiderParameters =\
            bp.io.remove_uncertainties(self.bicycle.parameters['Benchmark'])

    def plot(self, *args, **kwargs):
        '''
        Plots the time series of various signals.

        Parameters
        ----------
        signalName : string
            These should be strings that correspond to the signals available in
            the computed data. If the first character of the string is `-` then
            the negative signal will be plotted.
        signalType : string, optional
            This allows you to plot from the other signal types. Options are
            'computed', 'truncated', 'calibrated', 'raw'.

        '''
        if not kwargs:
            kwargs = {'signalType': 'computed'}

        # this currently only works if the sample rates from both sources is
        # the same

        mapping = {'computed': self.computedSignals,
                   'truncated': self.truncatedSignals,
                   'calibrated': self.calibratedSignals,
                   'raw': self.rawSignals}

        leg = []
        for i, arg in enumerate(args):
            legName = arg
            sign = 1.
            if arg[0] == '-':
                arg = arg[1:]
                sign = -1.
            signal = sign * mapping[kwargs['signalType']][arg]
            plt.plot(signal.time(), signal)
            leg.append(legName + ' [' + mapping[kwargs['signalType']][arg].units + ']')

        plt.legend(leg)

        plt.title('Rider: ' + self.metadata['Rider'] +
                  ', Speed: ' + str(self.metadata['Speed']) + 'm/s' +
                  ', Maneuver: ' + self.metadata['Maneuver'] +
                  ', Environment: ' + self.metadata['Environment'] + '\n' +
                  'Notes: ' + self.metadata['Notes'])

        plt.xlabel('Time [second]')

        plt.grid()

        plt.show()

    def verify_time_sync(self):
        self.plot('-AccelerometerAccelerationY', 'AccelerationZ', signalType="truncated")

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


def matlab_date_to_object(matDate):
    '''Returns a date time object based on a Matlab `datestr()` output.

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

    If `digits` = 4 and `num` = '25' then the function returns '0025'.

    Parameters
    ----------
    num : string
        A string representation of a number (i.e. '25')
    digits : integer
        The total number of digits desired.

    Returns
    -------
    num : string

    '''

    for i in range(digits - len(num)):
        num = '0' + num

    return num

def sync_data(directory='exports/'):
    """Sync's data to the biosport website."""
    user = 'biosport'
    host = 'mae.ucdavis.edu'
    remoteDir = '/home/grads/biosport/public_html/InstrumentedBicycleData/ProcessedData/'
    os.system("rsync -avz " + directory + ' -e ssh ' + user + '@' + host + ':' + remoteDir)

def create_html_tables(database, directory='docs/tables'):
    """Creates a table of all the basic info for the runs."""

    # create the directory if it isn't already there
    if not os.path.exists(directory):
        print "Creating {0}".format(directory)
        os.makedirs(directory)

    # make a run table
    dTab = database.root.data.datatable

    # only write these columns
    cols = ['DateTime',
            'RunID',
            'Rider',
            'Bicycle',
            'Maneuver',
            'Environment',
            'Speed',
            'Notes']

    lines = ['<table border="1">\n<tr>\n']

    for col in cols:
        lines.append("<th>" + col + "</th>\n")

    lines.append("</tr>\n")

    for row in dTab.iterrows():
        lines.append("<tr>\n")
        for cell in cols:
            lines.append("<td>" + str(row[cell]) + "</td>\n")
        lines.append("</tr>\n")

    lines.append("</table>")

    f = open(os.path.join(directory, 'RunTable.html'), 'w')
    f.writelines(lines)
    f.close()

    sTab = database.root.data.signaltable
    lines = ['<table border="1">\n<tr>\n']
    for col in sTab.colnames:
        lines.append("<th>" + col + "</th>\n")

    lines.append("</tr>\n")

    for row in sTab.iterrows():
        lines.append("<tr>\n")
        for cell in sTab.colnames:
            lines.append("<td>" + str(row[cell]) + "</td>\n")
        lines.append("</tr>\n")

    lines.append("</table>")

    f = open(os.path.join(directory, 'SignalTable.html'), 'w')
    f.writelines(lines)
    f.close()

    cTab = database.root.data.calibrationtable
    lines = ['<table border="1">\n<tr>\n']
    for col in cTab.colnames:
        if col not in ['v', 'x', 'y']:
            lines.append("<th>" + col + "</th>\n")

    lines.append("</tr>\n")

    for row in cTab.iterrows():
        lines.append("<tr>\n")
        for cell in cTab.colnames:
            if cell not in ['v', 'x', 'y']:
                lines.append("<td>" + str(row[cell]) + "</td>\n")
        lines.append("</tr>\n")

    lines.append("</table>")

    f = open(os.path.join(directory, 'CalibrationTable.html'), 'w')
    f.writelines(lines)
    f.close()

