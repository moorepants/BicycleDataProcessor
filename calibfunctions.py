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

def bodyfixedgyro(omega_x,omega_y,omega_z,bikeparms):
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
    s : array
        Calibrated signal

    '''
    lam = bikeparms['lambda']
    omega_xg =  omega_x*cos(lam) +omega_z*sin(lam)
    omega_yg =  omega_y
    omega_zg = -omega_x*sin(lam) +omega_z*sin(lam)
    return omega_xg

def steerrate(steergyro,omega_xg, omega_zg, bikeparms):
    lam = bikeparms['lambda']
    deltad = steergyro + omega_xg*np.sin(lam) - omega_zg*cos(lam)
    return deltad


