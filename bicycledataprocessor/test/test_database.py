#!/usr/bin/env python

import os

from numpy.random import randint
from numpy import ones
import numpy.testing as npt

from bicycledataprocessor import database


def test_create_signal_table():
    db = database.DataSet(fileName='sigtest.h5')
    # this should not work because the file isn't open
    db.create_signal_table()
    # open a new file and create the table
    db.open(mode='w')
    db.create_signal_table()
    db.close()
    # now try to create the table if it is already there
    db.open(mode='a')
    db.create_signal_table()
    db.close()
    os.remove('sigtest.h5')


def test_run_id_string():
    ids = [105, '000105', '00105', '0105', '105']

    for run in ids:
        assert database.run_id_string(run) == '00105'


def test_get_calib_data():
    """This just tests whether the .mat files return the same values as the .h5
    files."""

    runID = database.run_id_string(randint(0, 20))

    pathToData = '/media/Data/Documents/School/UC Davis/Bicycle Mechanics/BicycleDAQ/data/CalibData'

    matFile = os.path.join(pathToData, runID + '.mat')
    h5File =  os.path.join(pathToData, 'h5', runID + '.h5')

    matDat = database.get_calib_data(matFile)
    h5Dat = database.get_calib_data(h5File)

    for k, v in h5Dat.items():
        if isinstance(v, type('')):
            assert v == matDat[k]
        else:
            npt.assert_allclose(v, matDat[k])


def test_get_run_data():
    """This just tests whether the .mat files return the same values as the .h5
    files."""

    runID = database.run_id_string(randint(0, 700))

    pathToData = '/media/Data/Documents/School/UC Davis/Bicycle Mechanics/BicycleDAQ/data'

    matFile = os.path.join(pathToData, runID + '.mat')
    h5File =  os.path.join(pathToData, 'h5', runID + '.h5')

    matDat = database.get_run_data(matFile)
    h5Dat = database.get_run_data(h5File)

    for k, v in h5Dat.items():
        if k == 'NICols' or k == 'VNavCols' or k == 'VNavDataText' :
            assert v == matDat[k]
        elif k == 'NIData' or k == 'VNavData':
            npt.assert_allclose(v, matDat[k])
        elif k == 'par':
            for subKey, subVal in v.items():
                if isinstance(subVal, type(ones(1))) or subKey == 'Speed':
                    npt.assert_allclose(subVal, matDat[k][subKey])
                else:
                    assert subVal == matDat[k][subKey]
                #print('{} matches'.format(subKey))
        else:
            assert v == matDat[k]

        #print('{} in {} matches'.format(k, runID))
