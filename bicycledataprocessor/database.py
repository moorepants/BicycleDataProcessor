#!/usr/bin/env python

# built in imports
import os
import re
from operator import xor
from ConfigParser import SafeConfigParser

# I use this for debugging in IPython if available.
try:
    from IPython.core.debugger import Tracer
except ImportError:
    pass
else:
    set_trace = Tracer()

# dependencies
import numpy as np
import tables
import warnings
from scipy.io import loadmat

# I name my array nodes with a string of numbers which causes PyTables natural
# naming scheme not to work. This ignores those errors.
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

config = SafeConfigParser()
config.read(os.path.join(os.path.dirname(__file__), '..', 'defaults.cfg'))

class DataSet(object):

    def __init__(self, **kwargs):
        """Creates the object and sets the filename.

        Parameters
        ----------
        pathToDatabase : string, optional
            Path to the database file.
        pathToRunMat : string, optional
            Path to the directory which contains the raw run mat files.
        pathToCalibMat : string, optional
            Path to the directory which contains the raw calibration mat files.
        pathToRunH5 : string, optional
            Path to the directory containing the the raw run h5 files.
        pathToCalibH5 : string, optional
            Path to the directory containing the the raw calibration h5 files.
        pathToCorruption : string, optional
            The path to the data corruption csv file.

        Notes
        -----
        If a keyword argument is not specified the values present in the
        defaults.cfg data section will be used. The mat file directories will
        always be used if present, only if it isn't will the H5 directories be
        used.

        """

        acceptedKeywords = ['pathToDatabase', 'pathToRunMat', 'pathToCalibMat',
            'pathToRunH5', 'pathToCalibH5', 'pathToCorruption']

        for k in acceptedKeywords:
            if k in kwargs.keys():
                setattr(self, k, kwargs[k])
            else:
                setattr(self, k, config.get('data', k))

        # This class has the ability to load data from mat files or h5 files.
        # The preference is mat files.
        if self.pathToRunMat is not None:
            self.pathToRun = self.pathToRunMat
            self.runExt = '.mat'
        else:
            self.pathToRun = self.pathToRunH5
            self.runExt = '.h5'

        if self.pathToCalibMat is not None:
            self.pathToCalib = self.pathToCalibMat
            self.calibExt = '.mat'
        else:
            self.pathToCalib = self.pathToCalibH5
            self.calibExt = '.h5'

        # these columns may be used in the future but for now there is no reason to
        # introduce them into the database
        self.ignoredNICols = (['SeatpostBridge' + str(x) for x in range(1, 7)] +
            [x + 'Potentiometer' for x in ['Hip', 'Lean', 'Twist']] +
            [x + 'FootBridge' + y for x, y in zip(2 * ['Right',
            'Left'], ['1', '2', '2', '1'])])

        # TODO : these signal names don't match what I'm actually computing.
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
                              'tau', # why tau?
                              'YawRate']

    def _task_table_class(self):
        """Creates a class that is used to describe the table containing meta
        data for the processed task signals.

        Returns
        -------
        TaskTable : tables.IsDescription

        """

        class TaskTable(tables.IsDescription):
            Duration = tables.Float32Col(dflt=0.)
            FilterFrequency = tables.Float32Col(dflt=0.)
            MeanSpeed = tables.Float32Col(dflt=0.)
            RunID = tables.Int32Col(dflt=0)
            StdSpeed = tables.Float32Col(dflt=0.)
            Tau = tables.Float32Col(dflt=0.)

        return TaskTable

    def _calibration_table_class(self):
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

    def _run_table_class(self, run):
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

    def _signal_table_class(self):
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

    def create_database(self, compression=False):
        """Creates an HDF5 file for data collected from the instrumented
        bicycle.

        Parameters
        ----------
        compression : boolean, optional
            Basic compression will be used in the creation of the objects in
            the database.

        """

        if os.path.exists(self.pathToDatabase):
            response = raw_input(('{0} already exists.\n' +
                'Do you want to overwrite it? (y or n)\n').format(self.pathToDatabase))
            if response == 'y':
                print("{0} will be overwritten".format(self.pathToDatabase))
            else:
                print("{0} was not overwritten.".format(self.pathToDatabase))
                return

        print("Creating database...")

        self.compression = compression

        # create a new hdf5 file ready for writing
        self.open(mode='w', title='Instrumented Bicycle Data')
        self.close()

        # initialize all of the tables
        self.create_run_table()
        self.create_signal_table()
        self.create_calibration_table()
        self.create_task_table()

        print "{0} successfully created.".format(self.pathToDatabase)

    def open(self, **kwargs):
        """Opens the HDF5 database. This accepts any keyword arguments that
        tables.openFile uses."""

        self.database = tables.openFile(self.pathToDatabase, **kwargs)

    def close(self):
        """Closes the currently open HDF5 database."""

        self.database.close()

    def create_table(self, *args, **kwargs):
        """Creates an empty table at the root.

        Notes
        -----
        This is a wrappre to tables.createTable and excepts the same arguments as tables.createTable.

        """
        self.close()
        self.open(mode='a')

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
        else:
            table.flush()

        self.close()

    def create_run_table(self):
        """Creates an empty run information table."""

        files = list_files_in_dir(self.pathToRun)

        numRuns = len(files)

        # get two example runs
        filteredRun, unfilteredRun = get_two_runs(self.pathToRun)

        # generate the table description class
        RunTable = self._run_table_class(unfilteredRun)

        # add the data table to the root group
        self.create_table('/', 'runTable',
            RunTable, 'Run Information', expectedrows=(numRuns + 100))

    def create_signal_table(self):
        """Creates an empty signal information table."""

        # generate the signal table description class
        SignalTable = self._signal_table_class()

        self.create_table('/', 'signalTable',
                SignalTable, 'Signal Information', expectedrows=50)

    def create_calibration_table(self):
        """Creates an empty calibration table."""

        files = list_files_in_dir(self.pathToCalib)

        numCalibs = len(files)

        # generate the calibration table description class
        calibrationTable = self._calibration_table_class()

        # add the calibration table to the root group
        self.create_table('/', 'calibrationTable',
            calibrationTable, 'Calibration Information',
            expectedrows=(numCalibs + 10))

    def create_task_table(self):
        """Creates an empty task table."""


        taskTable = self._task_table_class()
        self.create_table('/', 'taskTable', taskTable,
            'Processed task signal meta data', expectedrows=1000)

        # delete any arrays that may be there too
        self.close()
        self.open(mode='a')
        try:
            self.database.root.taskData._f_remove(recursive=True)
        except tables.NoSuchNodeError:
            pass
        self.close()

    def sync_data(self, directory='exports/'):
        """Synchronizes data to the biosport website."""
        user = 'biosport'
        host = 'mae.ucdavis.edu'
        remoteDir = '/home/grads/biosport/public_html/InstrumentedBicycleData/ProcessedData/'
        os.system("rsync -avz " + directory + ' -e ssh ' + user + '@' + host + ':' + remoteDir)

    def create_html_tables(self, directory='docs/tables'):
        """Creates a table of all the basic info for the runs."""

        # create the directory if it isn't already there
        if not os.path.exists(directory):
            print "Creating {0}".format(directory)
            os.makedirs(directory)

        self.open()

        # make a run table
        dTab = self.database.root.runTable

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

        sTab = self.database.root.signalTable
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

        cTab = self.database.root.calibrationTable
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

        self.close()

    def fill_all_tables(self):
        """Writes data to all of the tables."""

        self.fill_signal_table()
        self.fill_calibration_table()
        self.fill_run_table()

        print("{} is ready for action!".format(self.pathToDatabase))

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
        filteredRun, unfilteredRun = get_two_runs(self.pathToRun)

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
        files = list_files_in_dir(self.pathToCalib)

        for f in files:
            print "Calibration file:", f
            calibDict = get_calib_data(os.path.join(self.pathToCalib, f))
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
        databaseRuns = [run_id_string(x) for x in runTable.col('RunID')]

        # load the list of files from the h5 directory
        files = list_files_in_dir(self.pathToRun)
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
                # make sure they are all run id strings
                runs = [run_id_string(x) for x in runs]
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
                    runData = get_run_data(os.path.join(self.pathToRun, runID +
                        self.runExt))
                    fill_row(row, runData)
                    row.update()
                # overwrite the arrays
                runGroup = self.database.root.rawData._f_getChild(runID)
                for i, col in enumerate(runData['NICols']):
                    if col not in self.ignoredNICols:
                        timeSeries = runGroup._f_getChild(col)
                        timeSeries[:] = runData['NIData'][i]
                for i, col in enumerate(runData['VNavCols']):
                    timeSeries = runGroup._f_getChild(col)
                    timeSeries[:] = runData['VNavData'][i]
                print('Overwrote run {}.'.format(runID))
            else:
                print('Did not overwrite run {}.'.format(runID))

        runTable.flush()

        # now add any new runs
        row = runTable.row
        for runID in runsToAppend:
            print('Appending run: {}'.format(runID))

            try:
                runData = get_run_data(os.path.join(self.pathToRun, runID +
                    self.runExt))
            except ValueError:
                # I'm getting a scipy.io.loadmat issue "total size of new array
                # must be unchanged"
                pass
            else:
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

    def add_task_signals(self, taskSignals, meta):
        """Writes processed task signals to the data base.

        Parameters
        ----------
        taskSignals : dictionary
            A dictionary of Signal objects.
        meta : dictionary
            The should contain the RunID, Tau, Duration, MeanSpeed, StdSpeed.

        """
        self.close()
        self.open(mode='a')

        taskTable = self.database.root.taskTable

        try:
            taskData = self.database.root.taskData
        except tables.NoSuchNodeError:
            taskData = self.database.createGroup('/', 'taskData')

        # if the run isn't in the table, then append it, if it is then overwite
        # it
        if meta['RunID'] in taskTable.cols.RunID:
            for row in taskTable.where('RunID == {}'.format(str(int(meta['RunID'])))):
                for k, v in meta.items():
                    row[k] = v
                row.update()
            runGroup = taskData._f_getChild(run_id_string(meta['RunID']))
            for name, sig in taskSignals.items():
                timeSeries = runGroup._f_getChild(name)
                timeSeries[:] = sig
                for attr in ['units', 'name', 'runid', 'sampleRate', 'source']:
                    timeSeries._f_setAttr(attr, getattr(sig, attr))
        else:
            for k, v in meta.items():
                taskTable.row[k] = v
            taskTable.row.append()

            # store all of the task signals as arrays
            taskGroup = self.database.createGroup(self.database.root.taskData,
                    run_id_string(meta['RunID']))
            for name, sig in taskSignals.items():
                arr = self.database.createArray(taskGroup, name, sig)
                for attr in ['units', 'name', 'runid', 'sampleRate', 'source']:
                    arr._f_setAttr(attr, getattr(sig, attr))
            taskTable.flush()

        self.close()

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

    def update_corrupt(self):
        """Updates the run table to reflect the latest values in the corruption
        file."""

        # load the corruption data
        corruption = self.load_corruption_data()

        # make sure the database is open for appending
        self.close()
        self.open(mode='a')

        for row in self.database.root.runTable.iterrows():
            if row['RunID'] in corruption['runid']:
                index = corruption['runid'].index(row['RunID'])

                for col in ['corrupt', 'warning']:
                    row[col] = corruption[col][index]

                for col in ['knee', 'handlebar', 'trailer']:
                    default = np.zeros(15, dtype=np.bool)
                    default[corruption[col][index]] = True
                    row[col] = default

                row.update()
                print('Updated the corruption data for run ' + row['RunID'])
            else:
                # set everything to default
                for col in ['corrupt', 'warning']:
                    row[col] = False

                for col in ['knee', 'handlebar', 'trailer']:
                    row[col] = np.zeros(15, dtype=np.bool)

                row.update()
                print('Corruption data for ' + row['RunID'] +
                        ' set to default.')

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

def run_id_string(runID):
    """Returns the run id in the five digit string format.

    Parameters
    ----------
    runID : str or int
        The run id either as an integer or string with leading zeros.

    Returns
    -------
    runID : str
        The five digit run id string.

    """

    return pad_with_zeros(str(runID), 5)

def get_row_num(runid, table):
    '''
    Returns the row number for a particular run id.

    Parameters
    ----------
    runid : int or string
        The run id.
    table : pytable
        A table which has a `RunID` column with run id integers.

    Returns
    -------
    rownum : int
        The row number for runid.

    '''
    # if the row number happens to correspond to the RunID, then try the quick
    # calculation, otherwise search for it
    try:
        rownum = table[int(runid)]['RunID']
    except IndexError:
        rownum = None

    if rownum != int(runid):
        rownum = [x.nrow for x in table.iterrows()
                  if x['RunID'] == int(runid)][0]
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

def get_two_runs(pathToRun):
    '''Gets the data from both a filtered and unfiltered run.'''

    # load in the data files
    files = list_files_in_dir(pathToRun)

    # get an example filtered and unfiltered run (wrt to the VN-100 data)
    filteredRun = get_run_data(os.path.join(pathToRun, files[0]))
    if filteredRun['par']['ADOT'] is not 14:
        raise ValueError('Run %d is not a filtered run, choose again' %
              filteredRun['par']['RunID'])

    unfilteredRun = get_run_data(os.path.join(pathToRun, files[-1]))
    if unfilteredRun['par']['ADOT'] is not 253:
        raise ValueError('Run %d is not a unfiltered run, choose again' %
              unfilteredRun['par']['RunID'])

    return filteredRun, unfilteredRun

def get_run_data(pathToFile):
    '''
    Returns data from the raw run files.

    Parameters
    ----------
    pathtofile : string
        The path to the mat or h5 file that contains run data.

    Returns
    -------
    rundata : dictionary
        A dictionary that looks similar to how the data was stored in Matlab.

    '''

    def parse_par(runData, key, val):
        """Cleans up the par dictionary."""
        try:
            if key == 'Speed':
                runData['par'][key] = float(val)
            else:
                runData['par'][key] = int(val)
        except:
            pstr = str(val)
            runData['par'][key] = pstr
            if pstr[0] == '$':
                parsed = parse_vnav_string(pstr)[0][2:-1]
                if len(parsed) == 1:
                    try:
                        parsed = int(parsed[0])
                    except:
                        parsed = parsed[0]
                else:
                    parsed = np.array([float(x) for x in parsed])
                runData['par'][key] = parsed

    ext = os.path.splitext(pathToFile)[1]

    # intialize a dictionary for storage
    runData = {}
    # put the parameters into a dictionary
    runData['par'] = {}

    if ext == '.mat':
        mat = loadmat(pathToFile, squeeze_me=True)

        for key, val in zip(mat['par'].dtype.names, mat['par'][()]):
            parse_par(runData, key, val)

        runData['NIData'] = mat['NIData'].T
        runData['VNavCols'] = [str(x).replace(' ', '') for x in mat['VNavCols']]
        inputPairs = [(x, int(y)) for x, y in zip(mat['InputPairs'].dtype.names, mat['InputPairs'][()])]
        runData['NICols'] = list(mat['InputPairs'].dtype.names)
        runData['VNavDataText'] = [str(x).strip() for x in mat['VNavDataText']]

    elif ext == '.h5':
        # open the file
        runfile = tables.openFile(pathToFile)

        # get the NIData
        runData['NIData'] = runfile.root.NIData.read()

        # get the VN-100 data column names
        # make the array into a list of python string and gets rid of unescaped
        # control characters
        columns = [re.sub(r'[^ -~].*', '', str(x))
                   for x in runfile.root.VNavCols.read()]
        # gets rid of white space
        runData['VNavCols'] = [x.replace(' ', '') for x in columns]

        for col in runfile.root.par:
            key = col.name
            val = col.read()[0]
            parse_par(runData, key, val)

        # get the NI column names
        # make a list of NI columns from the InputPair structure from matlab
        inputPairs = []
        for col in runfile.root.InputPairs:
            inputPairs.append((str(col.name), int(col.read()[0])))

        # get the VNavDataText
        runData['VNavDataText'] = [re.sub(r'[^ -~].*', '', str(x))
                                   for x in runfile.root.VNavDataText.read()]

        # close the file
        runfile.close()

    inputPairs.sort(key=lambda x: x[1])
    runData['NICols'] = [x[0] for x in inputPairs]

    # redefine the VNData using parsing that accounts for the corrupt values
    # better
    runData['VNavData'] = replace_corrupt_strings_with_nan(
                           runData['VNavDataText'],
                           runData['VNavCols'])

    if 'Notes' not in runData['par'].keys():
        runData['par']['Notes'] = ''

    if runData['par']['Notes'] == '[]':
        runData['par']['Notes'] = ''

    return runData

def get_calib_data(pathToFile):
    """Returns calibration data from the run h5 files using pytables and
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

    """

    calibData = {}

    ext = os.path.splitext(pathToFile)[1]

    if ext == '.mat':

        mat = loadmat(pathToFile)

        for k, v in zip(mat['data'].dtype.names, mat['data'][0, 0]):
            if len(v.flatten()) == 1:
                if isinstance(v[0], type(u'')):
                    calibData[k] = str(v[0])
                else:
                    calibData[k] = float(v[0])
            else:
                if k in ['x', 'v']:
                    if len(v.squeeze().shape) > 1:
                        calibData[k] = v.mean(axis=0)
                    else:
                        calibData[k] = v.squeeze()
                else:
                    calibData[k] = v.squeeze()

    elif ext == '.h5':

        calibFile = tables.openFile(pathToFile)

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

def list_files_in_dir(path):
    """Creates a list of mat or h5 files in a directory and sorts them by
    name."""

    inDir = os.listdir(path)
    files = []
    for thing in inDir:
        if thing.endswith('.mat') or thing.endswith('.h5'):
            files.append(thing)
    files.sort()

    return files
