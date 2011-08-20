import dataprocessor.dataprocessor as dp
import matplotlib.pyplot as plt

database = dp.load_database()

run = dp.Run('00124', database, forceRecalc=True, filterSigs=True)

steerAngle = run.computedSignals['YawAngle']
steerRate = run.computedSignals['YawRate']

steerAngFromInt = steerRate.integrate(initialCondition=steerAngle[0])

time = steerAngle.time()

plt.figure()
plt.plot(time, steerAngle)
plt.plot(time, steerAngFromInt)
plt.legend(('Steer Angle', 'Integrated Steer Rate'))

steerAngFromInt2 = steerRate.integrate(initialCondition=steerAngle[0],
                                       subtractMean=True)

plt.figure()
plt.plot(time, steerAngle)
plt.plot(time, steerAngFromInt2)
plt.legend(('Steer Angle', 'Integrated Steer Rate (drift fixed)'))

plt.show()
