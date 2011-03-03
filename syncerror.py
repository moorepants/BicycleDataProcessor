# -*- coding: utf-8 -*-
"""
Created on Wed Mar 02 10:02:55 2011

@author: Peter
"""

import numpy as np
import scipy as sp
import DataProcessor as dp
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

Fs = 200. # Samplerate [Hz]

run = dp.importmat('00157.h5')

s1_raw = -(run.root.NIData[8]-1.5)/(300./1000.)*9.81
s2_raw =  run.root.VNavData[5]

s1 = s1_raw - np.mean(s1_raw)
s2 = s2_raw - np.mean(s2_raw[~np.isnan(s2_raw)])

t  = np.arange(0,s1.shape[0])/Fs
print t


#===============================================================================
# # Test Signal: Parabola
# tshift = 1
# t  = np.linspace(0, 10, 11)
# t1 = t;
# t2 = t + tshift
# tau = 1
# s1 = t1**2.
# s2 = t2**2.
#===============================================================================

print 'test'
# Error function
def syncerror(tau,s1,s2,t): 
    t1_interp = np.linspace(np.min(t)+np.abs(tau),np.max(t)-np.abs(tau),t.shape[0])
    t2_interp = t1_interp - tau
    s1_interp = sp.interp(t1_interp, t, s1);
    s2_interp = sp.interp(t2_interp, t[~np.isnan(s2)], s2[~np.isnan(s2)]);
    e  = sum((s1_interp-s2_interp)**2)
    return e

tau0 = -0.01

tau = fmin_bfgs(syncerror,tau0,args=(s1,s2,t))
print tau


plt.plot(t, s1,'k') # plot x and y using default line style and color
#plt.plot(t, s2) # plot x and y using default line style and color
#plt.plot(t+tau0, s2) # plot x and y using default line style and color
plt.plot(t+tau , s2,'k:') # plot x and y using default line style and color
plt.show()


run.close()

