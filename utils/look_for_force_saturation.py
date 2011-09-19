#!/usr/bin/env python

# this code tries to count how many of the lateral force disturbances were too
# strong and saturated the sensor, making the signal unusable

import sys
sys.path.append('..')

import dataprocessor.dataprocessor as dp
import numpy as np

database = dp.load_database(filename='../InstrumentedBicycleData.h5')

runTable = database.root.data.datatable

withSaturationPull = []
withSaturationPushPull = []

for row in runTable.iterrows():
    isDisturbance = row['Maneuver'].endswith('Disturbance')
    isPull = int(row['RunID']) < 227 # only pull lateral force
    isPushPull = not isPull # push and pull lateral force
    isLow = (row['PullForceBridge'] < 0.05).any()
    isPullSat = isDisturbance and isPull and isLow
    isPushPullSat = isDisturbance and isPushPull and (isLow or (row['PullForceBridge'] > 5.8).any())
    if isPullSat:
        withSaturationPull.append(row['RunID'])
    if isPushPullSat:
        withSaturationPushPull.append(row['RunID'])

print withSaturationPull
print withSaturationPushPull

disturbanceDict = {}

for run in withSaturationPull:
    lateralForceVolts = dp.get_cell(runTable, 'PullForceBridge', run)
    runid = dp.get_cell(runTable, 'RunID', run)
    disturbanceDict[runid] = {}
    disturbanceDict[runid]['numOfDisturbances'] = 0
    disturbanceDict[runid]['numOfSaturated'] = 0
    # count the lateral disturbances
    zeroedRectified = np.abs(lateralForceVolts - lateralForceVolts[0])
    disturbanceIndices = np.nonzero(zeroedRectified > 1.5)[0]
    if len(disturbanceIndices != 0):
        disturbanceDict[runid]['numOfDisturbances'] += 1
    for i, ind in enumerate(disturbanceIndices[1:]):
        if ind - disturbanceIndices[i] > 100:
            disturbanceDict[runid]['numOfDisturbances'] += 1
    saturatedIndices = np.nonzero(zeroedRectified > 5.4444)[0]
    if len(saturatedIndices != 0):
        disturbanceDict[runid]['numOfSaturated'] += 1
    for i, ind in enumerate(saturatedIndices[1:]):
        if ind - saturatedIndices[i] > 1:
            disturbanceDict[runid]['numOfSaturated'] += 1


numOfDisturbances = 0
numOfSaturated = 0
for run, numbers in disturbanceDict.items():
    for k, v in numbers.items():
        if k == 'numOfDisturbances':
            numOfDisturbances += v
        elif k == 'numOfSaturated':
            numOfSaturated += v

print "{} pull disturbances in {} runs with {} saturated disturbances".format(numOfDisturbances, len(withSaturationPull), numOfSaturated)
print disturbanceDict

disturbanceDict = {}

for run in withSaturationPushPull:
    lateralForceVolts = dp.get_cell(runTable, 'PullForceBridge', run)
    runid = dp.get_cell(runTable, 'RunID', run)
    disturbanceDict[runid] = {}
    disturbanceDict[runid]['numOfDisturbances'] = 0
    disturbanceDict[runid]['numOfSaturated'] = 0
    # count the lateral disturbances
    zeroedRectified = np.abs(lateralForceVolts - lateralForceVolts[0])
    disturbanceIndices = np.nonzero(zeroedRectified > 1.5)[0]
    if len(disturbanceIndices != 0):
        disturbanceDict[runid]['numOfDisturbances'] += 1
    for i, ind in enumerate(disturbanceIndices[1:]):
        if ind - disturbanceIndices[i] > 100:
            disturbanceDict[runid]['numOfDisturbances'] += 1
    saturatedIndices = np.nonzero(zeroedRectified > 2.85)[0]
    if len(saturatedIndices != 0):
        disturbanceDict[runid]['numOfSaturated'] += 1
    for i, ind in enumerate(saturatedIndices[1:]):
        if ind - saturatedIndices[i] > 1:
            disturbanceDict[runid]['numOfSaturated'] += 1

numOfDisturbances = 0
numOfSaturated = 0
for run, numbers in disturbanceDict.items():
    for k, v in numbers.items():
        if k == 'numOfDisturbances':
            numOfDisturbances += v
        elif k == 'numOfSaturated':
            numOfSaturated += v

for k, v in disturbanceDict.items():
    print k, v

print "{} pull disturbances in {} runs with {} saturated disturbances".format(numOfDisturbances, len(withSaturationPushPull), numOfSaturated)

database.close()

