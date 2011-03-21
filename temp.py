#!/usr/bin/env python
import tables as tab
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline, interp1d
import DataProcessor as dp

runid = 108
print "RunID:", runid
wheelbase = 1.02
bumpLength = 1.

# open the data file
datafile = tab.openFile('InstrumentedBicycleData.h5')

datatable = datafile.root.data.datatable

# get some data
niAcc = dp.get_cell(datatable, 'FrameAccelY', runid)
vnAcc = dp.get_cell(datatable, 'AccelerationZ', runid)
sampleRate = dp.get_cell(datatable, 'NISampleRate', runid)
numSamples = dp.get_cell(datatable, 'NINumSamples', runid)
speed = dp.get_cell(datatable, 'Speed', runid)
threeVolts = dp.get_cell(datatable, 'ThreeVolts', runid)

# close the file
datafile.close()

time = dp.time_vector(numSamples, sampleRate)

# scale the NI signal from volts to m/s**2
niSig = -(niAcc - threeVolts / 2.) / (300. / 1000.) * 9.81
vnSig = vnAcc

filNiSig = dp.butterworth(niSig, 50., float(sampleRate))
# find the bump
nibump =  dp.find_bump(filNiSig, sampleRate, speed, wheelbase, bumpLength)
print 'NI Signal'
print nibump

# remove the nan's
v = vnSig[np.nonzero(np.isnan(vnSig)==False)]
t = time[np.nonzero(np.isnan(vnSig)==False)]
vn_spline = interp1d(t, v, kind='cubic')
plt.figure()
plt.plot(time, vn_spline(time))
plt.show()
filVnSig = dp.butterworth(vn_spline(time), 50., float(sampleRate))
vnbump =  dp.find_bump(filVnSig, sampleRate, speed, wheelbase, bumpLength)
print 'VNav Signal'
print vnbump

# plot the two raw signals and a shaded area for the bump
plt.figure()
fillx = [vnbump[0], vnbump[0], nibump[2], nibump[2]]
filly = [-30, 30, 30, -30]
plt.plot(filNiSig)
plt.plot(vnSig)
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

tau, error = dp.find_timeshift(niAcc, vnAcc, sampleRate, guess=guess)
print 'Tau:', tau

# plot the error landscape
plt.figure()
plt.plot(np.linspace(0., .5, num=len(error)), error)
plt.ylabel('Error')
plt.xlabel('Tau')

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
