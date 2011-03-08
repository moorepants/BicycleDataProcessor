import numpy as np
import scipy as sp

def linearcalib(V,calibdata):
    '''Linear tranformation from raw voltage measurements (V) to calibrated
    data signals (s).

    Parameters
    ----------
    V : array
        Measurements

    calibdata : Dictionary
        Calibration data

    Output
    ----------
    s : array
        Calibrated signal

    '''

    p1 = calibdata['Slope']
    p0 = calibdata['Offset']
    s = p1*V + p0
    return s

def rollpitchyawrate(framerate_x,framerate_y,framerate_z,bikeparms):
    '''Transforms the measured body fixed rates to global rates by
    rotating them along the head angle.

    Parameters
    ----------
    omega_x, omega_y, omega_z : array
        Body fixed angular velocities

    bikeparms : Dictionary
        Bike parameters

    Output
    ----------
    yawrate, pitchrate, rollrate : array
        Calibrated signal

    '''
    lam = bikeparms['lambda']
    rollrate  =  omega_x*cos(lam) + omega_z*sin(lam)
    pitchrate =  omega_y
    yawrate   = -omega_x*sin(lam) + omega_z*cos(lam)
    return yawrate, pitchrate, rollrate

def steerrate(steergyro,framerate_z):
    lam = bikeparms['lambda']
    deltad = steergyro + framerate_z
    return deltad



