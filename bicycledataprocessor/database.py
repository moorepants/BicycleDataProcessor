#!/usr/bin/env python

# built in imports
import os
import re
from operator import xor

# dependencies
import numpy as np
import tables

class DataSet(object):

    def __init__(self, fileName='InstrumentedBicycleData.h5', pathToH5='../BicycleDAQ/data/h5',
            pathToCorruption='data-corruption.csv'):
        """Creates the object and sets the filename.

        Parameters
        ----------
        fileName : string
            Path to the database file.
        pathToH5 : string
            Path to the directory containing the run h5 files.
        pathToCorruption : string
            The path to the data corruption csv file.

        """

        self.fileName = fileName
        self.pathToH5 = pathToH5
        self.pathToCalibH5 = os.path.join(self.pathToH5, '..', 'CalibData', 'h5')
        self.pathToCorruption = pathToCorruption

        # these columns may be used in the future but for now there is no reason to
        # introduce them into the database
        self.ignoredNICols = (['SeatpostBridge' + str(x) for x in range(1, 7)] +
            [x + 'Potentiometer' for x in ['Hip', 'Lean', 'Twist']] +
            [x + 'FootBridge' + y for x, y in zip(2 * ['Right',
            'Left'], ['1', '2', '2', '1'])])

        # these are data signals that will be created from the raw data
        self.processedCols = ['FrameAccelerationX',
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

    def open(self, **kwargs):
        """Opens the HDF5 database. This accepts any keyword arguments that
        tables.openFile uses."""

        self.database = tables.openFile(self.fileName, **kwargs)

    def close(self):
        """Closes the currently open HDF5 database."""

        try:
            self.database.close()
        except AttributeError:
            print('The database is not open.')
        else:
            del self.database

    def create_run_table(self):
        """Creates an empty run information table.

        Notes
        -----
        The database must be open with write permission.

        """

        numRuns = len(sorted(os.listdir(self.pathToH5)))

        # get two example runs
        filteredRun, unfilteredRun = get_two_runs(self.pathToH5)

        # generate the table description class
        RunTable = self.run_table_class(unfilteredRun)

        # add the data table to the root group
        self.create_table('/', 'runTable',
            RunTable, 'Run Information', expectedrows=(numRuns + 50))

    def create_table(self, *args, **kwargs):
        """Creates an empty table at the root.

        Notes
        -----
        This excepts the same arguments as createTable. The database must be
        open with write permission.

        """
        where = args[0]
        name = args[1]
        # add the signal table to the root group
        try:
            table = self.database.createTable(*args, **kwargs)
        except tables.NodeError:
            response = raw_input('{} already exists.\n'.format(name) +
                'Do you want to overwrite it? (y or n)\n')
            if response == 'y':
                print("{} will be overwritten.".format(name))
                self.database.removeNode(where, name)
                table = self.database.createTable(*args, **kwargs)
                table.flush()
            else:
                print("Aborted, {} was not overwritten.".format(name))
        except AttributeError:
            print('The database is not open for writing.')
        else:
            table.flush()

    def create_signal_table(self):
        """Creates an empty signal information table.

        Notes
        -----
        The database must be open with write permission.

        """

        # generate the signal table description class
        SignalTable = self.signal_table_class()

        self.create_table('/', 'signalTable',
                SignalTable, 'Signal Information', expectedrows=50)

    def create_calibration_table(self):
        """Creates an empty calibration table.

        Notes
        -----
        The database must be open with write permission.

        """

        numCalibs = len(sorted(os.listdir(self.pathToCalibH5)))

        # generate the calibration table description class
        calibrationTable = self.calibration_table_class()

        # add the calibration table to the root group
        self.create_table('/', 'calibrationTable',
            calibrationTable, 'Calibration Information',
            expectedrows=(numCalibs + 10))

    def create_database(self, compression=False):
        """Creates an HDF5 file for data collected from the instrumented
        bicycle.

        Parameters
        ----------
        compression : boolean, optional
            Basic compression will be used in the creation of the objects in
            the database.

        """

        if os.path.exists(self.fileName):
            response = raw_input(('{0} already exists.\n' +
                'Do you want to overwrite it? (y or n)\n').format(self.fileName))
            if response == 'y':
                print("{0} will be overwritten".format(self.fileName))
            else:
                print("{0} was not overwritten.".format(self.fileName))
                return

        print("Creating database...")

        self.compression = compression

        # create a new hdf5 file ready for writing
        self.open(mode='w', title='Instrumented Bicycle Data')

        # initialize all of the tables
        self.create_run_table()
        self.create_signal_table()
        self.create_calibration_table()

        self.close()

        print "{0} successfully created.".format(self.fileName)

    def signal_table_class(self):
        """Creates a class that is used to describe the table containing
        information about the signals.

        Returns
        -------
        CalibrationTable : class
            Table description class for pytables with columns defined.

        """

        class SignalTable(tables.IsDescription):
            calibration = tables.StringCol(20)
            isRaw = tables.BoolCol()
            sensor = tables.StringCol(20)
            signal = tables.StringCol(20)
            source = tables.StringCol(2)
            units = tables.StringCol(20)

        return SignalTable

    def calibration_table_class(self):
        """Creates a class that is used to describe the table containing the
        calibration data.

        Returns
        -------
        CalibrationTable : class
            Table description class for pytables with columns defined.

        """

        class CalibrationTable(tables.IsDescription):
            accuracy = tables.StringCol(10)
            bias = tables.Float32Col(dflt=np.nan)
            calibrationID = tables.StringCol(5)
            calibrationSupplyVoltage = tables.Float32Col(dflt=np.nan)
            name = tables.StringCol(20)
            notes = tables.StringCol(500)
            offset = tables.Float32Col(dflt=np.nan)
            runSupplyVoltage = tables.Float32Col(dflt=np.nan)
            runSupplyVoltageSource = tables.StringCol(10)
            rsq = tables.Float32Col(dflt=np.nan)
            sensorType = tables.StringCol(20)
            signal = tables.StringCol(26)
            slope = tables.Float32Col(dflt=np.nan)
            timeStamp = tables.StringCol(21)
            units = tables.StringCol(20)
            v = tables.Float32Col(shape=(50,))
            x = tables.Float32Col(shape=(50,))
            y = tables.Float32Col(shape=(50,))

        return CalibrationTable

    def run_table_class(self, run):
        '''Returns a class that is used for the table description for raw data
        for each run.

        Parameters
        ----------
        run : dict
            Contains the python dictionary of a run.

        Returns
        -------
        RunTable : class
            Table description class for pytables with columns defined.

        '''

        # set up the table description
        class RunTable(tables.IsDescription):
            for i, (key, val) in enumerate(run['par'].items()):
                if isinstance(val, type(1)):
                    exec(key + " = tables.Int32Col(pos=i)")
                elif isinstance(val, type('')):
                    exec(key + " = tables.StringCol(itemsize=300, pos=i)")
                elif isinstance(val, type(1.)):
                    exec(key + " = tables.Float32Col(pos=i)")
                elif isinstance(val, type(np.ones(1))):
                    exec(key + " = tables.Float32Col(shape=(" + str(len(val)) +
                            ", ), pos=i)")
                # a marker that declares the data corrupt or unusable
                corrupt = tables.BoolCol()
                # a market that declares the data quiestionable
                warning = tables.BoolCol()
                # mark a disturbance in the run that hand either a knee come
                # off, the handlebar touch the treadmill sides or the trailer
                # hit the side of the treadmill. There should never be any more
                # that 15 disturbances per run
                knee = tables.BoolCol(shape=(15))
                handlebar = tables.BoolCol(shape=(15))
                trailer = tables.BoolCol(shape=(15))

            # get rid intermediate variables so they are not stored in the class
            del(i, key, val)

        return RunTable

    def fill_signal_table(self):
        """Writes data to the signal information table."""

        # this needs a check to see if there is data already in the table with
        # the option to overwrite it

        print "Loading signal data."

        self.open(mode='a')

        # fill in the signal table
        signalTable = self.database.root.signalTable
        row = signalTable.row

        # get two example runs
        filteredRun, unfilteredRun = get_two_runs(self.pathToH5)

        # remove the bridge signals (I haven't used them yet!)
        niCols = filteredRun['NICols']
        for col in self.ignoredNICols:
            niCols.remove(col)

        # combine the VNavCols from unfiltered and filtered
        vnCols = set(filteredRun['VNavCols'] + unfilteredRun['VNavCols'])

        vnUnitMap = {'MagX': 'unitless',
                     'MagY': 'unitless',
                     'MagZ': 'unitless',
                     'AccelerationX': 'meter/second/second',
                     'AccelerationY': 'meter/second/second',
                     'AccelerationZ': 'meter/second/second',
                     'AngularRateX': 'radian/second',
                     'AngularRateY': 'radian/second',
                     'AngularRateZ': 'radian/second',
                     'AngularRotationX': 'degree',
                     'AngularRotationY': 'degree',
                     'AngularRotationZ': 'degree',
                     'Temperature': 'kelvin'}

        for sig in set(niCols + list(vnCols) + self.processedCols):
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
                row['units'] = vnUnitMap[sig]
            elif sig in self.processedCols:
                row['source'] = 'NA'
                row['isRaw'] = False
                row['calibration'] = 'na'
            else:
                raise KeyError('{0} is not raw or processed'.format(sig))

            row.append()

        signalTable.flush()

        self.close()

    def fill_calibration_table(self):
        """Writes the calibration data to the calibration table."""

        print "Loading calibration data."

        self.open(mode='a')

        # fill in the calibration table
        calibrationTable = self.database.root.calibrationTable
        row = calibrationTable.row

        # load the files from the h5 directory
        files = sorted(os.listdir(self.pathToCalibH5))

        for f in files:
            print "Calibration file:", f
            calibDict = get_calib_data(os.path.join(self.pathToCalibH5, f))
            for k, v in calibDict.items():
                if k in ['x', 'y', 'v']:
                    row[k] = size_vector(v, 50)
                else:
                    row[k] = v
            row.append()

        calibrationTable.flush()

        self.close()

    def fill_run_table(self, runs=None, overwrite=False):
        """Adds all the data from the hdf5 files in the h5 directory to the run
        information table and stores the time series data in arrays.

        Parameters
        ----------
        runs : string or list of strings, optional
            If `run` is `all`, the entire directory of individual run files
            will be added to the database. If `run` is a list of run ids, e.g.
            ['00345', '00346'], then those files will be added to the database.
            If run is the default `None`, then only the new files in the
            directory will be added.
        overwrite : boolean, optional
            If `True` any runs that are already in the database will be
            overwritten. Otherwise, if the runs already exist in the database,
            the user will be prompted to overwrite the data.

        """

        import warnings
        warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

        # open the hdf5 file for appending
        self.open(mode='a')

        # create a group to store the time series data, if the group is already
        # there, the leave it be
        try:
            self.database.createGroup('/', 'rawData')
        except tables.NodeError:
            pass

        # get a list of run ids that are already in the database
        runTable = self.database.root.runTable
        databaseRuns = [pad_with_zeros(str(x), 5) for x in
                runTable.col('RunID')]

        # load the list of files from the h5 directory
        files = sorted(os.listdir(self.pathToH5))
        # remove the extensions
        directoryRuns = [os.path.splitext(x)[0] for x in files]

        # find which runs that need to be added/modified
        if runs is not None:
            try:
                # try to sort the list if given
                runs.sort()
            except AttributeError:
                # otherwise it should be 'all'
                if runs != 'all':
                    raise ValueError("Please supply a list of runs or 'all' ")
                else:
                    runs = directoryRuns
        else:
            # if None then all the new runs in the directory should be added
            runs = list(set(directoryRuns) - set(databaseRuns))

        # load the corruption data
        corruption = self.load_corruption_data()

        # now find the runs that need to be modified and the runs that need to
        # be added
        runsToUpdate = sorted(list(set(runs) & set(databaseRuns)))
        runsToAppend = sorted(list(set(runs) - set(runsToUpdate)))

        def fill_row(row, runData):
            """Fills out all the columns in a row given the runData dictionary
            and the corruption dictionary."""

            runNum = runData['par']['RunID']

            # stick all the metadata in the run info table
            for par, val in runData['par'].items():
                row[par] = val

            # add the data corruption information
            if runNum in corruption['runid']:
                index = corruption['runid'].index(runNum)

                row['corrupt'] = corruption['corrupt'][index]
                row['warning'] = corruption['warning'][index]

                knee = np.zeros(15, dtype=np.bool)
                knee[corruption['knee'][index]] = True
                row['knee'] = knee

                handlebar = np.zeros(15, dtype=np.bool)
                handlebar[corruption['handlebar'][index]] = True
                row['handlebar'] = handlebar

                trailer = np.zeros(15, dtype=np.bool)
                trailer[corruption['trailer'][index]] = True
                row['trailer'] = trailer

        for runID in runsToUpdate:
            if overwrite is True:
                yesOrNo = 'yes'
            else:
                yesOrNo = None

            while yesOrNo != 'yes' and yesOrNo != 'no':
                q = 'Do you want to overwrite run {}?'.format(runID)
                yesOrNo = raw_input(q + ' (yes or no)\n')

            if yesOrNo == 'yes':
                for row in runTable.where('RunID == {}'.format(str(int(runID)))):
                    runData = get_run_data(os.path.join(self.pathToH5, runID + '.h5'))
                    fill_row(row, runData)
                    row.update()
                # overwrite the arrays
                runGroup = self.database.root.rawData._f_getChild(runID)
                for i, col in enumerate(runData['NICols']):
                    if col not in self.ignoredNICols:
                        timeSeries = runGroup._f_getChild(col)
                        timeSeries = runData['NIData'][i]
                for i, col in enumerate(runData['VNavCols']):
                    timeSeries = runGroup._f_getChild(col)
                    timeSeries = runData['VNavData'][i]
                print('Overwrote run {}.'.format(runID))
            else:
                print('Did not overwrite run {}.'.format(runID))

        runTable.flush()

        # now add any new runs
        row = runTable.row
        for runID in runsToAppend:
            print('Appending run: {}'.format(runID))

            runData = get_run_data(os.path.join(self.pathToH5, runID + '.h5'))

            fill_row(row, runData)
            row.append()

            # store the time series data in arrays
            runGroup = self.database.createGroup(self.database.root.rawData,
                    runID)
            for i, col in enumerate(runData['NICols']):
                if col not in self.ignoredNICols:
                    try:
                        self.database.createArray(runGroup, col,
                            runData['NIData'][i])
                    except IndexError:
                        print("{} not measured in this run.".format(col))

            for i, col in enumerate(runData['VNavCols']):
                self.database.createArray(runGroup, col,
                        runData['VNavData'][i])

        runTable.flush()

        self.close()

    def fill_all_tables(self):
        """Writes data to all of the tables."""

        self.fill_signal_table()
        self.fill_calibration_table()
        self.fill_run_table()

        print("{} is ready for action!".format(self.fileName))

    def load_corruption_data(self):
        """Returns a dictionary containing the contents of the provided data
        corruption file.

        Returns
        -------
        corruption : dictionary
            There is a keyword for each column in the file and a list of the
            values.

        """

        with open(self.pathToCorruption, 'r') as f:
            for i, line in enumerate(f):
                values = line.split(',')
                if i == 0:
                    corruption = {v.strip().lower(): list() for v in values}
                else:
                    corruption['runid'].append(int(values[0]))
                    corruption['corrupt'].append(values[1] == 'TRUE')
                    corruption['warning'].append(values[2] == 'TRUE')
                    corruption['knee'].append([int(x) for x in
                        values[3].split(';') if x])
                    corruption['handlebar'].append([int(x) for x in
                        values[4].split(';') if x])
                    corruption['trailer'].append([int(x) for x in
                        values[5].split(';') if x])
                    if corruption['reason'] != 'na':
                        corruption['reason'].append(values[6].strip())
                    else:
                        corruption['reason'].append('')

        return corruption

    def add_runs(self):
        self.open()

        # get list of runs in database

        # get list of runs in h5 folder
        files = sorted(os.listdir(self.pathToH5))

        # make list of runs to add

        # add runs

        self.close()

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
    if isinstance(cell, type(np.ones(1))) and cell.shape[0] == 18000:
        numsamp = datatable[rownum]['NINumSamples']
        cell = unsize_vector(cell, numsamp)

    return cell

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
    runfile = tables.openFile(pathtofile)

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

    calibFile = tables.openFile(pathToFile)

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
