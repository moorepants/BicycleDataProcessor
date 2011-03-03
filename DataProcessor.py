import tables as tab
import numpy as np
import os
import re

def create_database():
    '''Creates an HDF5 file for data collected from the instrumented bicycle'''

    # load the latest file in the data/h5 directory
    pathtoh5 = os.path.join('..', 'BicycleDAQ', 'data', 'h5')
    files = os.listdir(pathtoh5)
    files.sort()
    rundata = get_run_data(os.path.join(pathtoh5, files[-1]))
    # set up the table description
    class Run(tab.IsDescription):
        # add all of the column headings from par, NICols and VNavCols
        a = 1.

def parse_vnav_string(vnstr):
    '''Gets the good info from a VNav string'''
    vnstr = re.sub('$(.*)\*', '\1', vnstr)
    vnlist = vnstr.split(',')
    return vnlist

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

    # first let's get the NIData and VNavData
    rundata['NIData'] = runfile.root.NIData.read()
    rundata['VNavData'] = runfile.root.VNavData.read()

    # now create two lists that give the column headings for the two data sets
    rundata['VNavCols'] = [str(x) for x in runfile.root.VNavCols.read()]
    rundata['NICols'] = []
    for col in runfile.root.InputPairs:
        rundata['NICols'].append((str(col.name), int(col.read()[0])))

    rundata['NICols'].sort(key=lambda x: x[1])

    rundata['NICols'] = [x[0] for x in rundata['NICols']]

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
            rundata['par'][col.name] = str(col.read()[0])

    # get the VNavDataText
    rundata['VNavDataText'] = [str(x) for x in runfile.root.VNavDataText.read()]

    # close the file
    runfile.close()

    return rundata
