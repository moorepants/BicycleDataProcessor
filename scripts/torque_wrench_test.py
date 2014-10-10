import re
import numpy as np
import matplotlib.pyplot as plt
import bicycledataprocessor.main as bdp

dataset = bdp.DataSet(fileName='../InstrumentedBicycleData.h5')
dataset.open()

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for trialNum in ['00' + str(x) for x in range(707, 712)]:
    trial = bdp.Run(trialNum, dataset.database, '../BicycleParameters/data/')
    torque = trial.calibratedSignals['SteerTubeTorque'].convert_units('newton*meter')
    ax.plot(torque.time(), torque + 0.21)

ax.set_yticks(np.linspace(-10, 10, num=21))
ax.set_xlabel('Time [s]')
ax.set_ylabel('Steer Tube Torque Sensor [Nm]')
ax.grid()
fig.savefig('torque-step.png')
fig.show()

meanTorques = []
stdTorques = []
trueTorques = []
for trialNum in ['00' + str(x) for x in range(719, 739)]:
    trial = bdp.Run(trialNum, dataset.database, '../BicycleParameters/data/')
    torque = trial.calibratedSignals['SteerTubeTorque'].convert_units('newton*meter')
    meanTorques.append(torque.mean())
    stdTorques.append(torque.std())
    notes = trial.metadata['Notes']
    if 'positive' in notes:
        trueTorques.append(float(re.search(r'\d', notes).group(0)))
    elif 'negative' in notes:
        trueTorques.append(-float(re.search(r'\d', notes).group(0)))
    else:
        raise StandardError

# torque wrench error, this is an estimate of the 3 sigma error of the values
# reported by the torque wrench based on the calibration data that came with
# the wrench
dialReadings = np.array([[15., 45., 75., -15., -45., -75.]] * 3)
calibReadings = np.array([[14.91, 14.78, 14.87],
                         [44.68, 44.75, 44.87],
                         [75.58, 75.72, 75.78],
                         [-14.74, -14.79, -14.85],
                         [-44.10, -44.24, -44.37],
                         [-74.38, -74.48, -74.64]])
diffInReadings = abs(dialReadings.T - calibReadings).max(axis=1)
percentDiff = diffInReadings / dialReadings[0]

inlb2nm = 1. / 8.85074579

fig2 = plt.figure()
ax = fig2.add_subplot(1, 1, 1)
ax.plot(trueTorques, meanTorques, 'o')
dial = inlb2nm * dialReadings[0]
pd = percentDiff[dial.argsort()]
dial.sort()
# I add 0.1 N-m based on the 3 sigma error in actually reading the dial
# correctly
ax.plot(dial + dial * pd + 0.1, dial, 'k--')
ax.plot(dial - dial * pd - 0.1, dial, 'k--')
ax.plot(trueTorques, trueTorques, 'k-')
ax.set_yticks(np.linspace(-10, 10, num=21))
ax.grid()
ax.set_ylabel('Steer Tube Torque Sensor [Nm]')
ax.set_xlabel('Torque Wrench [Nm]')
ax.set_xlabel('Torque Wrench [Nm]')
fig2.savefig('torque-wrench-compare.png')
fig2.show()
