#!/usr/bin/env python

#try:
    #from IPython.core.debugger import Tracer
#except ImportError:
    #pass
#else:
    #set_trace = Tracer()

from warnings import warn

# dependencies
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fmin
import matplotlib.pyplot as plt # only for testing

from dtk.process import time_vector, butterworth, normalize, subtract_mean

# local dependencies
from bdpexceptions import TimeShiftError
#from signalprocessing import *

def find_bump(accelSignal, sampleRate, speed, wheelbase, bumpLength):
    '''Returns the indices that surround the bump in the acceleration signal.

    Parameters
    ----------
    accelSignal : ndarray, shape(n,)
        An acceleration signal with a single distinctive large acceleration
        that signifies riding over the bump.
    sampleRate : float
        The sample rate of the signal.
    speed : float
        Speed of travel (or treadmill) in meters per second.
    wheelbase : float
        Wheelbase of the bicycle in meters.
    bumpLength : float
        Length of the bump in meters.

    Returns
    -------
    indices : tuple
        The first and last indice of the bump section.

    '''

    # Find the indice to the maximum absolute acceleration in the provided
    # signal. This is mostly likely where the bump is. Skip the first few
    # points in case there are some endpoint problems with filtered data.
    n = len(accelSignal)
    nSkip = 5
    rectAccel = abs(subtract_mean(accelSignal[nSkip:n / 2.]))
    indice = np.nanargmax(rectAccel) + nSkip

    # This calculates how many time samples it takes for the bicycle to roll
    # over the bump given the speed of the bicycle, it's wheelbase, the bump
    # length and the sample rate.
    bumpDuration = (wheelbase + bumpLength) / speed
    bumpSamples = int(bumpDuration * sampleRate)
    # make the number divisible by four
    bumpSamples = int(bumpSamples / 4) * 4

    # These indices try to capture the length of the bump based on the max
    # acceleration indice.
    indices = (indice - bumpSamples / 4, indice, indice + 3 * bumpSamples / 4)

    # If the maximum acceleration is not greater than 0.5 m/s**2, then there was
    # probably was no bump collected in the acceleration data.
    maxChange = rectAccel[indice - nSkip]
    if maxChange < 0.5:
        warn('This run does not have a bump that is easily detectable. ' +
                'The bump only gave a {:1.2f} m/s^2 change in nominal accerelation.\n'\
                .format(maxChange) +
                'The bump indice is {} and the bump time is {:1.2f} seconds.'\
                    .format(str(indice), indice / float(sampleRate)))
        return None
    else:
        # If the bump isn't at the beginning of the run, give a warning.
        if indice > n / 3.:
            warn("This signal's max value is not in the first third of the data\n"
                    + "It is at %1.2f seconds out of %1.2f seconds" %
                 (indice / float(sampleRate), n / float(sampleRate)))

        # If there is a nan in the bump this maybe an issue down the line as the
        # it is prefferable for the bump to be in the data when the fitting occurs,
        # to get a better fit.
        if np.isnan(accelSignal[indices[0]:indices[1]]).any():
            warn('There is at least one NaN in this bump')

        return indices

def split_around_nan(sig):
    '''
    Returns the sections of an array not polluted with nans.

    Parameters
    ----------
    sig : ndarray, shape(n,)
        A one dimensional array that may or may not contain m nan values where
        0 <= m <= n.

    Returns
    -------
    indices : list, len(indices) = k
        List of tuples containing the indices for the sections of the array.
    arrays : list, len(indices) = k
        List of section arrays. All arrays of nan values are of dimension 1.

    k = number of non-nan sections + number of nans

    sig[indices[k][0]:indices[k][1]] == arrays[k]

    '''
    # if there are any nans then split the signal
    if np.isnan(sig).any():
        firstSplit = np.split(sig, np.nonzero(np.isnan(sig))[0])
        arrays = []
        for arr in firstSplit:
            # if the array has nans, then split it again
            if np.isnan(arr).any():
                arrays = arrays + np.split(arr, np.nonzero(np.isnan(arr))[0] + 1)
            # if it doesn't have nans, then just add it as is
            else:
                arrays.append(arr)
        # remove any empty arrays
        emptys = [i for i, arr in enumerate(arrays) if arr.shape[0] == 0]
        arrays = [arr for i, arr in enumerate(arrays) if i not in emptys]
        # build the indices list
        indices = []
        count = 0
        for i, arr in enumerate(arrays):
            count += len(arr)
            if np.isnan(arr).any():
                indices.append((count - 1, count))
            else:
                indices.append((count - len(arr), count))
    else:
        arrays, indices = [sig], [(0, len(sig))]

    return indices, arrays

def steer_rate(forkRate, angularRateZ):
    '''Returns the steer rate.

    Parameters
    ----------
    forkRate : ndarray, size(n,)
        The rate of the fork about the steer axis relative to the Newtonian
        reference frame.
    angularRateZ : ndarray, size(n,)
        The rate of the bicycle frame about the steer axis in the Newtonian
        reference frame.

    Returns
    -------
    steerRate : ndarray, size(n,)
        The rate of the fork about the steer axis relative to the bicycle
        frame.

    Notes
    -----
    The rates are defined such that a positive rate is pointing downward along
    the steer axis.

    '''
    return forkRate - angularRateZ

def steer_torque_components(frameAngRate, frameAngAccel, frameAccel,
        handlebarAngRate, handlebarAngAccel, steerAngle, steerColumnTorque,
        handlebarMass, handlebarInertia, damping, friction, d, ds):
    """Returns the components of the steer torque applied by the rider.

    Parameters
    ----------
    frameAngRate : ndarray, shape(3,n)
        The angular velocity of the bicycle frame in the Newtonian frame
        expressed in body fixed coordinates.
    frameAngAccel : ndarray, shape(3,n)
        The angular acceleration of the bicycle frame in the Newtonian frame
        expressed in body fixed coordinates.
    frameAccel : ndarray, shape(3,n)
        The linear acceleration of the frame accelerometer in the Newtonian
        frame expressed in body fixed coordinates.
    handlebarAngRate : ndarray, shape(n,)
        The component of angular rate of the handlebar in the Newtonian frame
        expressed in body fixed coordinates about the steer axis.
    handlebarAngAccel : ndarray, shape(n,)
        The component of angular acceleration of the handlebar in the Newtonian
        frame expressed in body fixed coordinates about the steer axis.
    steerAngle : ndarray, shape(n,)
        The angle of the steer column relative to the frame about the steer
        axis.
    steerColumnTorque : ndarray, shape(n,)
        The torque measured on the steer column between the handlebars and the
        fork and between the upper and lower bearings.
    handlebarMass : float
        The mass of the handlebar.
    handlebarInertia : ndarray, shape(3,3)
        The inertia tensor of the handlebars. Includes everything above and
        including the steer tube torque sensor. This is relative to a reference
        frame aligned with the steer axis and is about the center of mass.
    damping : float
        The damping coefficient associated with the total (upper and lower)
        bearing friction.
    friction : float
        The columb friction associated with the total (upper and lower) bearing
        friction.
    d : float
        The distance from the handlebar center of mass to the steer axis. The
        point on the steer axis is at the projection of the mass center onto
        the axis.
    ds : ndarray, shape(3,)
        The distance from the accelerometer to the point on the steer axis.
    plot : boolean, optional
        If true a plot of the components of the steer torque will be shown.

    Returns
    -------
    components : dictionary
        A dictionary containing the ten components of the rider applied steer
        torque.

    Notes
    -----
    The friction torque from the upper and lower bearings is halved because
    only one bearing is associated with the handlebar rigid body.

    """
    wb1 = frameAngRate[0]
    wb2 = frameAngRate[1]
    wb3 = frameAngRate[2]
    wb1p = frameAngAccel[0]
    wb2p = frameAngAccel[1]
    wb3p = frameAngAccel[2]
    av1 = frameAccel[0]
    av2 = frameAccel[1]
    av3 = frameAccel[2]
    wh3 = handlebarAngRate
    wh3p = handlebarAngAccel
    IH = handlebarInertia
    mH = handlebarMass
    delta = steerAngle

    components = {}

    components['Hdot1'] = -((IH[0, 0] * (wb1 * np.cos(delta) +
                                  wb2 * np.sin(delta)) +
                      IH[2, 0] * wh3) *
                      (-wb1 * np.sin(delta) + wb2 * np.cos(delta)))

    components['Hdot2'] = (IH[1, 1] *
            (-wb1 * np.sin(delta) + wb2 * np.cos(delta)) *
            (wb1 * np.cos(delta) + wb2 * np.sin(delta)))

    components['Hdot3'] = IH[2, 2] * wh3p

    components['Hdot4'] = IH[2, 0] * (-(-wb3 + wh3) * wb1 * np.sin(delta) +
                                (-wb3 + wh3) * wb2 * np.cos(delta) +
                                np.sin(delta) * wb2p + np.cos(delta) * wb1p)

    components['cross1'] = d * mH * (d * (-wb1 * np.sin(delta) + wb2 * np.cos(delta)) *
                              (wb1 * np.cos(delta) + wb2 * np.sin(delta))
                              + d * wh3p)
    components['cross2'] = -d * mH * (-ds[0] * wb2 ** 2 + ds[2] * wb2p -
                                (ds[0] * wb3 - ds[2] * wb1) * wb3 + av1) * np.sin(delta)
    components['cross3'] = d * mH * (ds[0] * wb1 * wb2 + ds[0] * wb3p + ds[2] * wb2 *
                               wb3 - ds[2] * wb1p + av2) * np.cos(delta)
    components['viscous'] = (damping  *  (-wb3 + wh3)) / 2.
    components['coulomb'] = np.sign(-wb3 + wh3) * friction / 2.
    components['steerColumn'] = steerColumnTorque

    return components

def steer_torque(components):
    """Returns the steer torque given the components.

    Parameters
    ----------
    components : dictionary
        A dictionary containing the ten components of the rider applied steer
        torque.

    Returns
    -------
    steerTorque : ndarray, shape(n,)
        The steer torque applied by the rider.

    """

    return np.sum(components.values(), axis=0)

def rear_wheel_contact_rate(rearRadius, rearWheelRate, yawAngle):
    """Returns the longitudinal and lateral components of the velocity of the
    rear wheel contact in the ground plane.

    Parameters
    ----------
    rearRadius : float
        The radius of the rear wheel.
    rearWheelRate : ndarray, shape(n,)
        The rear wheel rotation rate.
    yawAngle : ndarray, shape(n,)
        The yaw angle of the bicycle frame.

    Returns
    -------
    longitudinal : ndarray, shape(n,)
        The longitudinal deviation of the rear wheel contact point.
    lateral : ndarray, shape(n,)
        The lateral deviation of the rear wheel contact point.

    """
    longitudinal = -rearWheelRate * rearRadius * np.cos(yawAngle)
    lateral = -rearWheelRate * rearRadius * np.sin(yawAngle)
    return longitudinal, lateral

def sync_error(tau, signal1, signal2, time):
    '''Returns the error between two signal time histories.

    Parameters
    ----------
    tau : float
        The time shift.
    signal1 : ndarray, shape(n,)
        The signal that will be interpolated. This signal is
        typically "cleaner" that signal2 and/or has a higher sample rate.
    signal2 : ndarray, shape(n,)
        The signal that will be shifted to syncronize with signal 1.
    time : ndarray, shape(n,)
        The time vector for the two signals

    Returns
    -------
    error : float
        Error between the two signals for the given tau.

    '''
    # make sure tau isn't too large
    if np.abs(tau) >= time[-1]:
        raise TimeShiftError(('abs(tau), {0}, must be less than or equal to ' +
                         '{1}').format(str(np.abs(tau)), str(time[-1])))

    # this is the time for the second signal which is assumed to lag the first
    # signal
    shiftedTime = time + tau

    # create time vector where the two signals overlap
    if tau > 0:
        intervalTime = shiftedTime[np.nonzero(shiftedTime < time[-1])]
    else:
        intervalTime = shiftedTime[np.nonzero(shiftedTime > time[0])]

    # interpolate between signal 1 samples to find points that correspond in
    # time to signal 2 on the shifted time
    sig1OnInterval = np.interp(intervalTime, time, signal1);

    # truncate signal 2 to the time interval
    if tau > 0:
        sig2OnInterval = signal2[np.nonzero(shiftedTime <= intervalTime[-1])]
    else:
        sig2OnInterval = signal2[np.nonzero(shiftedTime >= intervalTime[0])]

    # calculate the error between the two signals
    error = np.linalg.norm(sig1OnInterval - sig2OnInterval)

    return error

def find_timeshift(niAcc, vnAcc, sampleRate, speed, plotError=False):
    '''Returns the timeshift, tau, of the VectorNav [VN] data relative to the
    National Instruments [NI] data.

    Parameters
    ----------
    niAcc : ndarray, shape(n, )
        The acceleration of the NI accelerometer in its local Y direction.
    vnAcc : ndarray, shape(n, )
        The acceleration of the VN-100 in its local Z direction. Should be the
        same length as NIacc and contains the same signal albiet time shifted.
        The VectorNav signal should be leading the NI signal.
    sampleRate : integer or float
        Sample rate of the signals. This should be the same for each signal.
    speed : float
        The approximate forward speed of the bicycle.

    Returns
    -------
    tau : float
        The timeshift.

    Notes
    -----
    The Z direction for `VNacc` is assumed to be aligned with the steer axis
    and pointing down and the Y direction for the NI accelerometer should be
    aligned with the steer axis and pointing up.

    '''
    # raise an error if the signals are not the same length
    N = len(niAcc)
    if N != len(vnAcc):
        raise TimeShiftError('Signals are not the same length!')

    # make a time vector
    time = time_vector(N, sampleRate)

    # the signals are opposite sign of each other, so fix that
    niSig = -niAcc
    vnSig = vnAcc

    # some constants for find_bump
    wheelbase = 1.02 # this is the wheelbase of the rigid rider bike
    bumpLength = 1.
    cutoff = 30.
    # filter the NI Signal
    filNiSig = butterworth(niSig, cutoff, sampleRate)
    # find the bump in the filtered NI signal
    niBump =  find_bump(filNiSig, sampleRate, speed, wheelbase, bumpLength)

    # remove the nan's in the VN signal and the corresponding time
    v = vnSig[np.nonzero(np.isnan(vnSig) == False)]
    t = time[np.nonzero(np.isnan(vnSig) == False)]
    # fit a spline through the data
    vn_spline = UnivariateSpline(t, v, k=3, s=0)
    # and filter it
    filVnSig = butterworth(vn_spline(time), cutoff, sampleRate)
    # and find the bump in the filtered VN signal
    vnBump = find_bump(filVnSig, sampleRate, speed, wheelbase, bumpLength)

    if vnBump is None or niBump is None:
        guess = 0.3
    else:
        # get an initial guess for the time shift based on the bump indice
        guess = (niBump[1] - vnBump[1]) / float(sampleRate)

    # Since vnSig may have nans we should only use contiguous data around
    # around the bump. The first step is to split vnSig into sections bounded
    # by the nans and then selected the section in which the bump falls. Then
    # we select a similar area in niSig to run the time shift algorithm on.
    if vnBump is None:
        bumpLocation = 800 # just a random guess so things don't crash
    else:
        bumpLocation = vnBump[1]
    indices, arrays = split_around_nan(vnSig)
    for pair in indices:
        if pair[0] <= bumpLocation < pair[1]:
            bSec = pair

    # subtract the mean and normalize both signals
    niSig = normalize(subtract_mean(niSig, hasNans=True), hasNans=True)
    vnSig = normalize(subtract_mean(vnSig, hasNans=True), hasNans=True)

    niBumpSec = niSig[bSec[0]:bSec[1]]
    vnBumpSec = vnSig[bSec[0]:bSec[1]]
    timeBumpSec = time[bSec[0]:bSec[1]]

    if len(niBumpSec) < 200:
        warn('The bump section is only {} samples wide.'.format(str(len(niBumpSec))))

    # set up the error landscape, error vs tau
    # The NI lags the VectorNav and the time shift is typically between 0 and
    # 1 seconds
    tauRange = np.linspace(0., 2., num=500)
    error = np.zeros_like(tauRange)
    for i, val in enumerate(tauRange):
        error[i] = sync_error(val, niBumpSec, vnBumpSec, timeBumpSec)

    if plotError:
        plt.figure()
        plt.plot(tauRange, error)
        plt.xlabel('tau')
        plt.ylabel('error')
        plt.show()

    # find initial condition from landscape
    tau0 = tauRange[np.argmin(error)]

    print "The minimun of the error landscape is %f and the provided guess is %f" % (tau0, guess)

    # Compute the minimum of the function using both the result from the error
    # landscape and the bump find for initial guesses to the minimizer. Choose
    # the best of the two.
    tauBump, fvalBump  = fmin(sync_error, guess, args=(niBumpSec,
        vnBumpSec, timeBumpSec), full_output=True, disp=True)[0:2]

    tauLandscape, fvalLandscape = fmin(sync_error, tau0, args=(niBumpSec, vnBumpSec,
        timeBumpSec), full_output=True, disp=True)[0:2]

    if fvalBump < fvalLandscape:
        tau = tauBump
    else:
        tau = tauLandscape

    #### if the minimization doesn't do a good job, just use the tau0
    ###if np.abs(tau - tau0) > 0.01:
        ###tau = tau0
        ###print "Bad minimizer!! Using the guess, %f, instead." % tau

    print "This is what came out of the minimization:", tau

    if not (0.05 < tau < 2.0):
        raise TimeShiftError('This tau, {} s, is probably wrong'.format(str(tau)))

    return tau

def truncate_data(signal, tau):
    '''
    Returns the truncated vectors with respect to the timeshift tau.

    Parameters
    ---------
    signal : Signal(ndarray), shape(n, )
        A time signal from the NIData or the VNavData.
    tau : float
        The time shift.

    Returns
    -------
    truncated : ndarray, shape(m, )
        The truncated time signal.

    '''
    t = time_vector(len(signal), signal.sampleRate)

    # shift the ni data cause it is the cleaner signal
    tni = t - tau
    tvn = t

    # make the common time interval
    tcom = tvn[np.nonzero(tvn < tni[-1])]

    if signal.source == 'NI':
        truncated = np.interp(tcom, tni, signal)
    elif signal.source == 'VN':
        truncated = signal[np.nonzero(tvn <= tcom[-1])]
    else:
        raise ValueError('No source was defined in this signal.')

    return truncated

def yaw_roll_pitch_rate(angularRateX, angularRateY, angularRateZ,
                        lam, rollAngle=0.):
    '''Returns the bicycle frame yaw, roll and pitch rates based on the body
    fixed rate data taken with the VN-100 and optionally the roll angle
    measurement.

    Parameters
    ----------
    angularRateX : ndarray, shape(n,)
        The body fixed rate perpendicular to the headtube and pointing forward.
    angularRateY : ndarray, shape(n,)
        The body fixed rate perpendicular to the headtube and pointing to the
        right of the bicycle.
    angularRateZ : ndarray, shape(n,)
        The body fixed rate aligned with the headtube and pointing downward.
    lam : float
        The steer axis tilt.
    rollAngle : ndarray, shape(n,), optional
        The roll angle of the bicycle frame.

    Returns
    -------
    yawRate : ndarray, shape(n,)
        The yaw rate of the bicycle frame.
    rollRate : ndarray, shape(n,)
        The roll rate of the bicycle frame.
    pitchRate : ndarray, shape(n,)
        The pitch rate of the bicycle frame.

    '''
    yawRate = -(angularRateX*np.sin(lam) -
                angularRateZ * np.cos(lam)) / np.cos(rollAngle)
    rollRate = angularRateX * np.cos(lam) + angularRateZ * np.sin(lam)
    pitchRate = (angularRateY + angularRateX * np.sin(lam) * np.tan(rollAngle) -
                 angularRateZ * np.cos(lam) * np.tan(rollAngle))

    return yawRate, rollRate, pitchRate
