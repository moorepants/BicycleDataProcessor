#!/usr/bin/env python

import tables as tab
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import UnivariateSpline
import DataProcessor as dp

# pick a run number
runid = 156
print "RunID:", runid

# open the data file
datafile = tab.openFile('InstrumentedBicycleData.h5')

datatable = datafile.root.data.datatable

# get the raw data
niAcc = dp.get_cell(datatable, 'FrameAccelY', runid)
vnAcc = dp.get_cell(datatable, 'AccelerationZ', runid)
sampleRate = dp.get_cell(datatable, 'NISampleRate', runid)
numSamples = dp.get_cell(datatable, 'NINumSamples', runid)
speed = dp.get_cell(datatable, 'Speed', runid)
threeVolts = dp.get_cell(datatable, 'ThreeVolts', runid)

# close the file
datafile.close()

# make a nice time vector
time = dp.time_vector(numSamples, sampleRate)

# scale the NI signal from volts to m/s**2, and switch the sign
niSig = -(niAcc - threeVolts / 2.) / (300. / 1000.) * 9.81
vnSig = vnAcc

# see if there are nan's in the vnSig
if np.isnan(vnAcc).any():
    nanI = np.nonzero(np.isnan(vnAcc))[0]
    print "This run has %d nan's at indices:" % len(nanI), nanI

# some constants for find_bump
wheelbase = 1.02
bumpLength = 1.
cutoff = 50.
# filter the NI Signal
filNiSig = dp.butterworth(niSig, cutoff, sampleRate)
# find the bump of the filter NI signal
nibump =  dp.find_bump(filNiSig, sampleRate, speed, wheelbase, bumpLength)
print 'NI Signal'
print nibump

# remove the nan's
v = vnSig[np.nonzero(np.isnan(vnSig) == False)]
t = time[np.nonzero(np.isnan(vnSig) == False)]
vn_spline = UnivariateSpline(t, v, k=3, s=0)
filVnSig = dp.butterworth(vn_spline(time), cutoff, sampleRate)
vnbump =  dp.find_bump(filVnSig, sampleRate, speed, wheelbase, bumpLength)
print 'VNav Signal'
print vnbump

# plot the filtered signals
plt.figure()
plt.plot(time, filNiSig, '-') #, time, niSig, '.')
plt.plot(time, filVnSig, '-') #, time, vnSig, '.')
plt.xlabel('Time [sec]')
plt.ylabel('Acceleration') # [$\frac{m}{s^2}$]')
plt.title('Filtered Signals')
plt.legend(('NI Filterd', 'VN Filtered'))
#plt.legend(('NI Filterd', 'NI', 'VN Filtered', 'VN'))

# plot the two raw signals and a shaded area for the bump
plt.figure()
fillx = [time[vnbump[0]], time[vnbump[0]],
         time[nibump[2]], time[nibump[2]]]
filly = [-30, 30, 30, -30]
plt.plot(time, niSig)
plt.plot(time, vnSig)
plt.fill(fillx, filly, 'y', edgecolor='k', alpha=0.4)
plt.ylim((np.nanmax(niSig) + .1, np.nanmin(niSig) - .1))
plt.legend(['NI', 'VN'])
plt.title('Before truncation')

# plot only the bump
niBumpSig = niSig[vnbump[0]:nibump[2]]
vnBumpSig = vnSig[vnbump[0]:nibump[2]]
timeBump = time[vnbump[0]:nibump[2]]

plt.figure()
plt.plot(timeBump, niBumpSig)
plt.plot(timeBump, vnBumpSig)
plt.title('This the bump')

# get an initial guess for the time shift based on the bump indice
guess = (nibump[1] - vnbump[1]) / float(sampleRate)

# find the section that the bump belongs to
indices, arrays = dp.split_around_nan(vnAcc)
for pair in indices:
    if pair[0] <= vnbump[1] < pair[1]:
        bSec = pair

# plot the bump section
niBumpSec = niSig[bSec[0]:bSec[1]]
vnBumpSec = vnSig[bSec[0]:bSec[1]]
timeBumpSec = time[bSec[0]:bSec[1]]

plt.figure()
plt.plot(timeBumpSec, niBumpSec)
plt.plot(timeBumpSec, vnBumpSec)
plt.title('This the bump section')

tau, error = dp.find_timeshift(niAcc[bSec[0]:bSec[1]], vnAcc[bSec[0]:bSec[1]],
                               sampleRate, guess=guess)

# plot the error landscape
plt.figure()
plt.plot(np.linspace(0., .5, num=len(error)), error)
plt.ylabel('Error')
plt.xlabel('Tau')

# truncate the signals based on the calculated tau
niSigTr = dp.truncate_data(niSig, 'NI', sampleRate, tau)
vnSigTr = dp.truncate_data(vnSig, 'VN', sampleRate, tau)

timeTr = dp.time_vector(len(niSigTr), sampleRate)

# plot the truncated data
plt.figure()
plt.plot(timeTr, niSigTr)
plt.plot(timeTr, vnSigTr)
plt.legend(['NI', 'VN'])
plt.title('After truncation')

plt.show()
