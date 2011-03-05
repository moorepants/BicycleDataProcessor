import tables as tab
import numpy as np
import os
import re

def pad_vector(vector, length):
    '''Returns a vector with nan's padded on to the end.

    Parameters
    ----------
    vector : numpy array, shape(n, )
        The vector that needs some padding.
    length : int
        The desired length after the padding.
    '''
    nans = np.ones(length-len(vector))*np.nan
    return np.append(vector, nans)

def fill_table(datafile):
    '''Adds all the data to the table'''

    # load the files from the ../BicycleDAQ/data/h5 directory
    pathtoh5 = os.path.join('..', 'BicycleDAQ', 'data', 'h5')
    files = sorted(os.listdir(pathtoh5))
    # open an hdf5 file for appending
    data = tab.openFile(datafile, mode='a')
    # get the table
    rawtable = data.root.rawdata.rawdatatable
    # get the row
    row = rawtable.row
    # fill the rows with data
    for run in files:
        print run
        rundata = get_run_data(os.path.join(pathtoh5, run))
        for par, val in rundata['par'].items():
            row[par] = val
        for i, col in enumerate(rundata['NICols']):
            #print row[col].shape, rundata['NIData'][i].shape
            row[col] = pad_vector(rundata['NIData'][i], 12000)
        for i, col in enumerate(rundata['VNavCols']):
            row[col] = pad_vector(rundata['VNavData'][i], 12000)
        row.append()
    rawtable.flush()
    data.close()

def create_database():
    '''Creates an HDF5 file for data collected from the instrumented bicycle'''

    # load the latest file in the ../BicycleDAQ/data/h5 directory
    pathtoh5 = os.path.join('..', 'BicycleDAQ', 'data', 'h5')
    files = sorted(os.listdir(pathtoh5))
    filteredrun = get_run_data(os.path.join(pathtoh5, files[0]))
    unfilteredrun = get_run_data(os.path.join(pathtoh5, files[-1]))
    if filteredrun['par']['ADOT'] is not 14:
        print 'Run %d is not a filtered run, choose again' %
        filteredrun['par']['RunID']
    if unfilteredrun['par']['ADOT'] is not 253:
        print 'Run %d is not a unfiltered run, choose again' %
        unfilteredrun['par']['RunID']
    # generate the table description class
    RawRun = create_raw_run_class(filteredrun, unfilteredrun)
    # open a new hdf5 file for writing
    data = tab.openFile('InstrumentedBicycleData.h5', mode='w',
                               title='Instrumented Bicycle Data')
    # create a group for the raw data
    rgroup = data.createGroup('/', 'rawdata', 'Raw Data')
    # add the data table to this group
    rtable = data.createTable(rgroup, 'rawdatatable', RawRun, 'Primary Data Table')
    rtable.flush()
    data.close()

def create_raw_run_class(filteredrun, unfilteredrun):
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
    

    # set up the table description
    class RawRun(tab.IsDescription):
        # add all of the column headings from par, NICols and VNavCols
        for i, col in enumerate(unfilteredrun['NICols']):
            exec(col + " = tab.Float32Col(shape=(12000, ), pos=i)")
        for k, col in enumerate(unfilteredrun['VNavCols']):
            exec(col + " = tab.Float32Col(shape=(12000, ), pos=i+1+k)")
        for i, (key, val) in enumerate(unfilteredrun['par'].items()):
            pos = k+1+i
            if isinstance(val, type(1)):
                exec(key + " = tab.Int64Col(pos=pos)")
            elif isinstance(val, type('')):
                exec(key + " = tab.StringCol(itemsize=50, pos=pos)")
            elif isinstance(val, type(1.)):
                exec(key + " = tab.Float64Col(pos=pos)")
            elif isinstance(val, type(np.ones(1))):
                exec(key + " = tab.Float64Col(shape=(" + str(len(val)) + ", ), pos=pos)")

        # get rid of these intermediate variables
        del(i, k, col, key, pos, val)

    return RawRun

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

    # get the NIData and VNavData
    rundata['NIData'] = runfile.root.NIData.read()
    rundata['VNavData'] = runfile.root.VNavData.read()

    # make the array into a list of python strings
    columns = [str(x) for x in runfile.root.VNavCols.read()]
    # gets rid of unescaped control characters
    columns = [re.sub(r'[^ -~].*', '', x) for x in columns]
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
