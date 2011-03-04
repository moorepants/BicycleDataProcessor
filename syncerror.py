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

data = dp.get_run_data('../BicycleDAQ/data/h5/00170.h5')

s1_raw = -(data['NIData'][8]-1.5)/(300./1000.)*9.81
s2_raw = data['VNavData'][5]
s1 = s1_raw - np.mean(s1_raw)
s2 = s2_raw - np.mean(s2_raw[~np.isnan(s2_raw)])

N = s1.shape[0]
t = np.arange(0,N)/Fs

# # Test Signal: Parabola
# tshift = 1
# t  = np.linspace(0, 10, 11)
# t1 = t;
# t2 = t + tshift
# tau = 1
# s1 = t1**2.
# s2 = t2**2.

def find_timeshift(NI_acc,VN_acc):
    '''
    Returns the timeshift (tau) of the VectorNav (VN) data relative to the
    National Instruments (NI) data based on the first 20% of the data.

    Parameters
    ----------
    NI_acc : array
        Lateral acceleration data of the NI accelerometer
    VN_acc : array
        Lateral acceleration data of the VN accelerometer
        
    Returns
    -------
    tau : array
        Timeshift relative to the NI signals
    '''
    
    s1 = NI_acc
    S2 = VN_acc

    # Error function
    def sync_error(tau,s1,s2,t):
        N = t.shape[0]
        t1_interp = np.linspace(np.min(t)+np.abs(tau),np.max(t)-np.abs(tau),N)
        t2_interp = t1_interp - tau
        s1_interp = sp.interp(t1_interp, t, s1);
        s2_interp = sp.interp(t2_interp, t[~np.isnan(s2)], s2[~np.isnan(s2)]);
        e  = sum((s1_interp[0:round(0.2*N)]-s2_interp[0:round(0.2*N)])**2)
        return e
        
    # Error Landscape
    tau_range = np.linspace(-1,1,201)
    e = np.zeros(tau_range.shape)
    for i in range(len(tau_range)):
        e[i] = sync_error(tau_range[i],s1,s2,t)

    # Find initial condition from landscape and optimize!
    tau0 = tau_range[np.argmin(e)]
    tau  = fmin_bfgs(sync_error,tau0,args=(s1,s2,t))
    
    return tau
    
    
tau = find_timeshift(s1,s2)


## # Check roll angle
## s1roll_raw = -(data['NIData'][7]-1.5)/(300./1000.)*9.81
## s2roll_raw = data['VNavData'][4]
## s1roll = s1roll_raw - np.mean(s1roll_raw)
## s2roll = s2roll_raw - np.mean(s2roll_raw[~np.isnan(s2roll_raw)])

###########################
## # Plotting
###########################

## plt.close('all')

## # Plotting
## plt.figure()
## plt.plot(tau_range,e,'k') 
## plt.show()

## # Plotting
## plt.figure()
## plt.plot(t,s1roll)
## plt.plot(t+tau,s2roll)
## plt.show

## # Plotting
## plt.figure()
## plt.plot(t, s1,'k') 
## plt.plot(t+tau , s2,'k--')
## plt.show()
