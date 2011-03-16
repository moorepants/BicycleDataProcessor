#!/usr/bin/env python
import tables as tab
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import DataProcessor as dp

runid = 195
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
threevolts = dp.get_cell(datatable, 'ThreeVolts', runid)

datafile.close()

# find the bump in both signals
print 'NI Signal'
nibump =  dp.find_bump(niSig, float(sampleRate), speed, wheelbase, bumpLength)
print nibump
print 'VNav Signal'
vnbump =  dp.find_bump(vnSig, float(sampleRate), speed, wheelbase, bumpLength)
print vnbump

niBumpSig = niSig[vnbump[0]:nibump[1]]
vnBumpSig = vnSig[vnbump[0]:nibump[1]]

tau, e = dp.find_timeshift(niBumpSig, vnBumpSig, sampleRate)
print 'Tau:', tau

plt.figure()
plt.plot(e)

# scale 
niSig = -(niSig - threevolts/2.) / (300. / 1000.) * 9.81

# subtract the mean
niSig = niSig - stats.nanmean(niSig)
vnSig = vnSig - stats.nanmean(vnSig)

niBumpSig = niSig[vnbump[0]:nibump[1]]
vnBumpSig = vnSig[vnbump[0]:nibump[1]]

plt.figure()
plt.plot(niBumpSig)
plt.plot(vnBumpSig)
plt.title('This the bump')

plt.figure()
plt.plot(niSig)
plt.plot(vnSig)
plt.legend(['NI', 'VN'])
plt.title('Before truncation')

niSigTr = dp.truncate_data(niSig, 'NI', sampleRate, tau)
vnSigTr = dp.truncate_data(vnSig, 'VN', sampleRate, tau)

# plot the truncated data
plt.figure()
plt.plot(niSigTr)
plt.plot(vnSigTr)
plt.legend(['NI', 'VN'])
plt.title('After truncation')

# plot the difference to see if you can see the point at which the nan's
# potentially shift the data
plt.figure()
plt.plot(niSigTr-vnSigTr)
plt.title('Difference in the two signals')

t = np.linspace(0., numSamples/sampleRate, num=numSamples)
dNdt = dp.derivative(t, niSig, method='combination')

# plot the derivative of the ni sig
plt.figure()
plt.plot(t, dNdt)
plt.title('Derivative of the NI Signal')

plt.show()
