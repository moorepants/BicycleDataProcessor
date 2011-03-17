#!/usr/bin/env python
import tables as tab
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import DataProcessor as dp

runid = 145
print "RunID:", runid
wheelbase = 1.02
bumpLength = 1.

# open the data file
datafile = tab.openFile('InstrumentedBicycleData.h5')

datatable = datafile.root.data.datatable

# get some data
niSig = dp.get_cell(datatable, 'FrameAccelY', runid)
vnSig = dp.get_cell(datatable, 'AccelerationZ', runid)
sampleRate = dp.get_cell(datatable, 'NISampleRate', runid)
numSamples = dp.get_cell(datatable, 'NINumSamples', runid)
speed = dp.get_cell(datatable, 'Speed', runid)
threeVolts = dp.get_cell(datatable, 'ThreeVolts', runid)

datafile.close()

# scale the NI signal from volts to m/s**2
niSig = -(niSig - threeVolts / 2.) / (300. / 1000.) * 9.81

# find the bump in both signals
print 'NI Signal'
nibump =  dp.find_bump(niSig, sampleRate, speed, wheelbase, bumpLength)
print nibump

print 'VNav Signal'
vnbump =  dp.find_bump(vnSig, sampleRate, speed, wheelbase, bumpLength)
print vnbump

# plot the two raw signals and a shaded area for the bump
plt.figure()
fillx = [vnbump[0], vnbump[0], nibump[2], nibump[2]]
filly = [-30, 30, 30, -30]
plt.plot(niSig, 'g')
plt.plot(vnSig, 'b')
plt.fill(fillx, filly, 'y', edgecolor='k', alpha=0.4)
plt.ylim((np.nanmax(niSig) + .1, np.nanmin(niSig) - .1))
plt.legend(['NI', 'VN'])
plt.title('Before truncation')

# plot only the bump
niBumpSig = niSig[vnbump[0]:nibump[2]]
vnBumpSig = vnSig[vnbump[0]:nibump[2]]

plt.figure()
plt.plot(niBumpSig)
plt.plot(vnBumpSig)
plt.title('This the bump')

# get an initial guess for the time shift based on the bump indice
guess = (nibump[1] - vnbump[1]) / float(sampleRate)
print 'This is the guess:', guess

tau, error = dp.find_timeshift(niBumpSig, vnBumpSig, sampleRate, guess=guess)
print 'Tau:', tau

# plot the error landscape
plt.figure()
plt.plot(np.linspace(0., .5, num=len(error)), error)

# truncate the signals based on the calculated tau
niSigTr = dp.truncate_data(niSig, 'NI', sampleRate, tau)
vnSigTr = dp.truncate_data(vnSig, 'VN', sampleRate, tau)

# plot the truncated data
plt.figure()
plt.plot(niSigTr)
plt.plot(vnSigTr)
plt.legend(['NI', 'VN'])
plt.title('After truncation')

#### plot the difference to see if you can see the point at which the nan's
#### potentially shift the data
###plt.figure()
###plt.plot(niSigTr-vnSigTr)
###plt.title('Difference in the two signals')
###
###t = np.linspace(0., numSamples/sampleRate, num=numSamples)
###dNdt = dp.derivative(t, niSig, method='combination')
###
#### plot the derivative of the ni sig
###plt.figure()
###plt.plot(t, dNdt)
###plt.title('Derivative of the NI Signal')

plt.show()
