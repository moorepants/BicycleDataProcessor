#!/usr/bin/env python

# dependencies
from scipy.interpolate import UnivariateSpline
from scipy.optimize import fmin
import matplotlib.pyplot as plt # only for testing

# local dependencies
from signalprocessing import *
from dtk.process import *

#def wheel_contact_point_rates():
    #"""
    #Returns the rate of motion of the rear wheel contact points of a bicycle.
#
    #"""
#
    #d1 = cos(lam) * (c+w-rR * tan(lam))
    #d3 = -cos(lam) * (c-rF * tan(lam))
    #d2 = -(rF-rR-sin(lam) * d1-sin(lam) * d3) / cos(lam)
    #z[1] = cos(q3)
    #z[2] = sin(q3)
    #z[3] = cos(q4)
    #z[4] = sin(q4)
    #z[5] = cos(lam+q5)
    #z[6] = sin(lam+q5)
    #z[7] = cos(q6)
    #z[8] = sin(q6)
    #z[9] = cos(q7)
    #z[10] = sin(q7)
    #z[11] = cos(q8)
    #z[12] = sin(q8)
    #z[13] = z[5] * z[9]
    #z[14] = z[6] * z[9]
    #z[15] = z[5] * z[10]
    #z[16] = z[6] * z[10]
    #z[17] = z[2] * z[3]
    #z[18] = z[1] * z[3]
    #z[19] = z[2] * z[4]
    #z[20] = z[1] * z[4]
    #z[21] = z[1] * z[13] - z[10] * z[17] - z[14] * z[19]
    #z[22] = z[2] * z[13] + z[10] * z[18] + z[14] * z[20]
    #z[23] = z[4] * z[10] - z[3] * z[14]
    #z[24] = z[16] * z[19] - z[1] * z[15] - z[9] * z[17]
    #z[25] = z[9] * z[18] - z[2] * z[15] - z[16] * z[20]
    #z[26] = z[3] * z[16] + z[4] * z[9]
    #z[27] = z[1] * z[6] + z[5] * z[19]
    #z[28] = z[2] * z[6] - z[5] * z[20]
    #z[29] = z[3] * z[5]
    #z[30] = 1 - pow(z[26],2)
    #z[31] = pow(z[30],0.5)
    #z[32] = 1 / z[31]
    #z[33] = z[26] / z[31]
    #z[34] = rF * z[33]
    #z[35] = rF * z[32]
    #z[37] = z[17] * z[20] - z[18] * z[19]
    #z[38] = pow(z[3],2) + pow(z[19],2) + pow(z[20],2)
    #z[40] = z[1] * z[20] + z[2] * z[19]
    #z[41] = z[1] * z[18] + z[2] * z[17]
    #z[45] = z[5] * z[41] + z[6] * z[37]
    #z[48] = z[5] * z[7] - z[6] * z[8]
    #z[49] = z[5] * z[8] + z[6] * z[7]
    #z[50] = -z[5] * z[8] - z[6] * z[7]
    #z[51] = z[37] * z[48] + z[41] * z[50]
    #z[52] = z[38] * z[48]
    #z[53] = z[37] * z[49] + z[41] * z[48]
    #z[54] = z[38] * z[49]
    #z[55] = -z[2] * z[13] - z[10] * z[18] - z[14] * z[20]
    #z[56] = -z[1] * z[14] - z[13] * z[19]
    #z[57] = z[10] * z[19] - z[14] * z[17]
    #z[58] = z[14] * z[18] - z[10] * z[20]
    #z[59] = z[13] * z[20] - z[2] * z[14]
    #z[60] = z[3] * z[10] + z[4] * z[14]
    #z[61] = z[3] * z[13]
    #z[62] = z[2] * z[15] + z[16] * z[20] - z[9] * z[18]
    #z[63] = z[10] * z[17] + z[14] * z[19] - z[1] * z[13]
    #z[64] = z[9] * z[19] + z[16] * z[17]
    #z[66] = -z[9] * z[20] - z[16] * z[18]
    #z[68] = z[3] * z[9] - z[4] * z[16]
    #z[69] = z[3] * z[14] - z[4] * z[10]
    #z[70] = z[3] * z[15]
    #z[71] = z[1] * z[5] - z[6] * z[19]
    #z[72] = z[5] * z[17]
    #z[73] = z[5] * z[20] - z[2] * z[6]
    #z[74] = z[2] * z[5] + z[6] * z[20]
    #z[75] = z[5] * z[18]
    #z[76] = z[4] * z[5]
    #z[77] = z[3] * z[6]
    #z[78] = z[24] * z[28] + z[27] * z[62]
    #z[80] = z[27] * z[64] + z[28] * z[66] + z[29] * z[68]
    #z[83] = z[21] * z[72] - z[22] * z[75] - z[23] * z[76]
    #z[84] = z[21] * z[73] + z[22] * z[27]
    #z[85] = z[21] * z[25] + z[24] * z[55]
    #z[87] = z[24] * z[57] + z[25] * z[58] + z[26] * z[60]
    #z[89] = pow(z[11],2) + pow(z[12],2)
    #z[90] = z[11] * z[78] - z[12] * z[85]
    #z[94] = z[11] * z[85] + z[12] * z[78]
    #z[92] = z[11] * z[80] - z[12] * z[87]
    #z[96] = z[11] * z[87] + z[12] * z[80]
    #z[82] = z[21] * z[71] + z[22] * z[74] - z[23] * z[77]
    #z[65] = z[1] * z[16] + z[15] * z[19]
    #z[67] = z[2] * z[16] - z[15] * z[20]
    #z[81] = z[27] * z[65] + z[28] * z[67] + z[29] * z[70]
    #z[86] = z[24] * z[56] + z[25] * z[59] - z[26] * z[61]
    #z[93] = z[11] * z[81] - z[12] * z[86]
    #z[95] = z[11] * z[86] + z[12] * z[81]
    #z[79] = z[27] * z[63] + z[28] * z[55] + z[29] * z[69]
    #z[88] = pow(z[24],2) + pow(z[25],2) + pow(z[26],2)
    #z[91] = z[11] * z[79] - z[12] * z[88]
    #z[97] = z[11] * z[88] + z[12] * z[79]
    #z[101] = rR * z[38]
    #z[136] = rR * (z[48] * z[52]+z[49] * z[54])
    #z[140] = z[101] - z[136]
    #z[186] = z[26] * z[70]
    #z[189] = z[186] / pow(z[30],0.5)
    #z[192] = z[189] / pow(z[31],2)
    #z[195] = rF * z[192]
    #z[208] = z[3] * z[140]
    #z[198] = (z[26] * z[189]+z[31] * z[70]) / pow(z[31],2)
    #z[201] = rF * z[198]
    #z[204] = z[195] - d1 * z[29] - d2 * z[77] - d3 * z[61] - z[26] * z[201] - z[34] * z[70]
    #z[100] = rR * z[37]
    #z[110] = d1 * z[45]
    #z[113] = d1 * z[40]
    #z[134] = d3 * z[84]
    #z[127] = d2 * z[84]
    #z[173] = z[4] * z[6]
    #z[174] = z[3] * z[10] + z[9] * z[173]
    #z[128] = d3 * z[85] - d2 * z[78]
    #z[176] = z[3] * z[9] - z[10] * z[173]
    #z[161] = z[34] * z[94]
    #z[179] = z[11] * z[174] + z[12] * z[76]
    #z[141] = z[11] * z[21] - z[12] * z[27]
    #z[144] = z[11] * z[27] + z[12] * z[21]
    #z[153] = z[35] * (z[24] * z[84]+z[90] * z[141]+z[94] * z[144])
    #z[142] = z[11] * z[22] - z[12] * z[28]
    #z[145] = z[11] * z[28] + z[12] * z[22]
    #z[158] = z[35] * (z[25] * z[84]+z[90] * z[142]+z[94] * z[145])
    #z[147] = z[34] * z[90]
    #z[182] = z[12] * z[174] - z[11] * z[76]
    #z[214] = z[3] * z[100] + z[3] * z[110] + z[76] * z[113] + z[76] * z[134] + z[127] * z[174] + z[128] * z[176] + z[161] * z[179] - z[1] * z[153] - z[2] * z[158] - z[147] * z[182]
    #z[239] = z[204] * z[214]
    #z[159] = z[25] * z[35] * z[89]
    #z[154] = z[24] * z[35] * z[89]
    #z[213] = z[1] * z[159] - z[2] * z[154]
    #z[254] = z[2] * z[213]
    #z[218] = -z[1] * z[154] - z[2] * z[159]
    #z[236] = z[204] * z[218]
    #z[178] = z[11] * z[13] - z[6] * z[12]
    #z[99] = rR * z[40]
    #z[181] = z[6] * z[11] + z[12] * z[13]
    #z[211] = z[1] * z[158] + z[13] * z[127] + z[161] * z[178] - z[99] - z[2] * z[153] - z[6] * z[113] - z[6] * z[134] - z[15] * z[128] - z[147] * z[181]
    #z[252] = z[2] * z[211]
    #z[257] = z[239] * z[254] - z[236] * z[252]
    #z[157] = z[35] * (z[25] * z[83]+z[92] * z[142]+z[96] * z[145])
    #z[126] = d2 * z[83]
    #z[163] = z[34] * z[96]
    #z[39] = z[1] * z[17] - z[2] * z[18]
    #z[98] = rR * z[39]
    #z[152] = z[35] * (z[24] * z[83]+z[92] * z[141]+z[96] * z[144])
    #z[112] = d1 * z[39]
    #z[133] = d3 * z[83]
    #z[130] = d3 * z[87] - d2 * z[80]
    #z[149] = z[34] * z[92]
    #z[210] = z[1] * z[157] + z[13] * z[126] + z[163] * z[178] - z[98] - z[2] * z[152] - z[6] * z[112] - z[6] * z[133] - z[15] * z[130] - z[149] * z[181]
    #z[135] = rR * (z[48] * z[51]+z[49] * z[53])
    #z[139] = z[100] - z[135]
    #z[207] = z[3] * z[139]
    #z[228] = z[2] * z[207]
    #z[184] = z[26] * z[68]
    #z[187] = z[184] / pow(z[30],0.5)
    #z[190] = z[187] / pow(z[31],2)
    #z[193] = rF * z[190]
    #z[196] = (z[26] * z[187]+z[31] * z[68]) / pow(z[31],2)
    #z[199] = rF * z[196]
    #z[202] = z[193] + rR * z[4] + d1 * z[173] + d3 * z[60] - d2 * z[76] - z[26] * z[199] - z[34] * z[68]
    #z[221] = z[211] * z[218] - z[213] * z[214]
    #z[42] = pow(z[5],2) + pow(z[6],2)
    #z[137] = rR * z[42]
    #z[206] = z[3] * z[18] + z[4] * z[20]
    #z[231] = z[137] * z[206]
    #z[156] = z[35] * (z[25] * z[82]+z[93] * z[142]+z[95] * z[145])
    #z[125] = d2 * z[82]
    #z[162] = z[34] * z[95]
    #z[151] = z[35] * (z[24] * z[82]+z[93] * z[141]+z[95] * z[144])
    #z[114] = d1 * z[42]
    #z[132] = d3 * z[82]
    #z[129] = d3 * z[86] - d2 * z[81]
    #z[150] = z[34] * z[93]
    #z[209] = z[1] * z[156] + z[13] * z[125] + z[162] * z[178] - z[2] * z[151] - z[6] * 
    #z[114] - z[6] * z[132] - z[15] * z[129] - z[150] * z[181]
    #z[216] = z[76] * z[114] + z[76] * z[132] + z[125] * z[174] + z[129] * z[176] + 
    #z[162] * z[179] - z[1] * z[151] - z[2] * z[156] - z[150] * z[182]
    #z[222] = z[209] * z[218] - z[213] * z[216]
    #z[225] = z[2] * z[218] - z[1] * z[213]
    #z[233] = z[137] * z[207]
    #z[268] = z[221] * z[231] + z[222] * z[228] - z[225] * z[233]
    #z[46] = z[6] * z[38]
    #z[111] = d1 * z[46]
    #z[215] = z[3] * z[101] + z[3] * z[111] + z[76] * z[112] + z[76] * z[133] + z[126] * 
    #z[174] + z[130] * z[176] + z[163] * z[179] - z[1] * z[152] - z[2] * z[157] - z[149] * 
    #z[182]
    #z[265] = z[204] * z[213]
    #z[223] = z[1] * z[207]
    #z[205] = -z[3] * z[17] - z[4] * z[19]
    #z[219] = z[1] * z[206] - z[2] * z[205]
    #z[230] = z[1] * z[218] + z[2] * z[213]
    #z[234] = z[204] * (z[223] * z[225]-z[219] * z[221]-z[228] * z[230])
    #z[274] = (z[208] * z[257]+z[210] * z[228] * z[236]-z[202] * z[268]-z[215] * z[228] * 
    #z[265]) / z[234]
    #z[160] = z[35] * (z[91] * z[142]+z[97] * z[145])
    #z[164] = z[34] * z[97]
    #z[155] = z[35] * (z[91] * z[141]+z[97] * z[144])
    #z[131] = d3 * z[88] - d2 * z[79]
    #z[148] = z[34] * z[91]
    #z[212] = z[1] * z[160] + z[164] * z[178] - z[2] * z[155] - z[15] * z[131] - z[148] * 
    #z[181]
    #z[185] = z[26] * z[69]
    #z[188] = z[185] / pow(z[30],0.5)
    #z[191] = z[188] / pow(z[31],2)
    #z[194] = rF * z[191]
    #z[197] = (z[26] * z[188]+z[31] * z[69]) / pow(z[31],2)
    #z[200] = rF * z[197]
    #z[203] = z[194] + d3 * z[26] - z[26] * z[200] - z[34] * z[69]
    #z[217] = z[131] * z[176] + z[164] * z[179] - z[1] * z[155] - z[2] * z[160] - z[148] * 
    #z[182]
    #z[275] = (z[212] * z[228] * z[236]-z[203] * z[268]-z[217] * z[228] * z[265]) / z[234]
    #z[47] = pow(z[7],2) + pow(z[8],2)
    #z[138] = rR * z[47]
    #z[238] = z[206] * z[213]
    #z[235] = z[206] * z[211] - z[2] * z[207]
    #z[241] = z[207] * z[213]
    #z[242] = z[1] * z[204]
    #z[243] = z[238] * z[239] - z[235] * z[236] - z[241] * z[242]
    #z[273] = z[138] * z[243] / z[234]
    #z[246] = z[205] * z[213]
    #z[247] = z[2] * z[204]
    #z[244] = z[205] * z[211] - z[1] * z[207]
    #z[248] = z[239] * z[246] + z[241] * z[247] - z[236] * z[244]
    #z[276] = z[138] * z[248] / z[234]
    #z[260] = z[1] * z[213]
    #z[258] = z[1] * z[211]
    #z[261] = z[239] * z[260] - z[236] * z[258]
    #z[226] = z[137] * z[205]
    #z[269] = z[221] * z[226] + z[222] * z[223] - z[230] * z[233]
    #z[277] = (z[208] * z[261]+z[210] * z[223] * z[236]-z[202] * z[269]-z[215] * z[223] * z[265]) / z[234]
    #z[278] = (z[212] * z[223] * z[236]-z[203] * z[269]-z[217] * z[223] * z[265]) / z[234]
    #u1 = z[274] * u4 + z[275] * u7 - z[273] * u6
    #u2 = z[276] * u6 - z[277] * u4 - z[278] * u7

def find_bump(accelSignal, sampleRate, speed, wheelbase, bumpLength):
    '''Returns the indices that surround the bump in the acceleration signal.

    Parameters
    ----------
    accelSignal : ndarray, shape(n,)
        This is an acceleration signal with a single distinctive large
        acceleration that signifies riding over the bump.
    sampleRate : float
        This is the sample rate of the signal.
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
    # get the indice of the larger of the max and min
    maxmin = (np.nanmax(accelSignal), np.nanmin(accelSignal))
    if np.abs(maxmin[0]) > np.abs(maxmin[1]):
        indice = np.nanargmax(accelSignal)
    else:
        indice = np.nanargmin(accelSignal)

    print 'Bump indice:', indice
    print 'Bump time:', indice / sampleRate

    # give a warning if the bump doesn't seem to be at the beginning of the run
    if indice > len(accelSignal) / 3.:
        print "This signal's max value is not in the first third of the data"
        print("It is at %f seconds out of %f seconds" %
            (indice / sampleRate, len(accelSignal) / sampleRate))

    bumpDuration = (wheelbase + bumpLength) / speed
    print "Bump duration:", bumpDuration
    bumpSamples = int(bumpDuration * sampleRate)
    # make the number divisible by four
    bumpSamples = int(bumpSamples / 4) * 4

    # get the first quarter before the tallest spike and whatever is after
    indices = (indice - bumpSamples / 4, indice, indice + 3 * bumpSamples / 4)

    if np.isnan(accelSignal[indices[0]:indices[1]]).any():
        print 'There is at least one NaN in this bump'

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

def steer_torque(handlebarRate, handlebarAccel, steerRate, steerColumnTorque, handlebarInertia,
        damping, friction):
    '''Returns the steer torque applied by the rider.

    Parameters
    ----------
    handlebarRate : ndarray, shape(3,n)
        The angular velocity of the handlebar in the Newtonian frame expressed
        in body fixed coordinates.
    handlebarAccel : ndarray, shape(3,n)
        The angular acceleration of the handlebar in the Newtonian frame
        expressed in body fixed coordinates.
    steerRate : ndarray, shape(n,)
        The rate of the steer column relative to the frame about the steer
        axis.
    steerColumnTorque : ndarray, shape(n,)
        The torque measured on the steer column between the handlebars and the
        fork and between the upper and lower bearings.
    handlebarInertia : ndarray, shape(3,3)
        The inertia tensor of the handlebars. Includes everything above and including
        the steer tube torque sensor. This is relative to a reference frame
        aligned with the steer axis and is about the center of mass.
    damping : float
        The damping coefficient associated with the bearing friction.
    friction : float
        The columb friction associated with the bearing friction.

    Returns
    -------
    steerTorque : ndarray, shape(n,)
        The steer torque applied by the rider.

    '''
    # this assumes a symmetric handlebar
    Ts = steerColumnTorque
    # bearing friction torque
    # take half of this because we calculated the damping and friction of both
    # sets of bearings combined
    Tf = (damping * steerRate + np.sign(steerRate) * friction) / 2.
    # derivative of the angluar momentum of the handlebar
    I = handlebarInertia
    w = handlebarRate
    a = handlebarAccel
    hdot1 = I[1, 1] * w[0] * w[1]
    hdot2 = I[2, 0] * a[0]
    hdot3 = I[2, 2] * a[2]
    hdot4 = - w[1] * (I[0, 0] * w[1] + I[2, 0] * w[2])
    time = Ts.time()
    plt.figure()
    plt.plot(time, hdot1, time, hdot2, time, hdot3, time, hdot4)
    plt.legend(['hd1', 'hd2', 'hd3', 'hd4'])

    Hdot = (I[1, 1] * w[0] * w[1] + I[2, 0] * a[0] + I[2, 2] * a[2]
            - w[1] * (I[0, 0] * w[1] + I[2, 0] * w[2]))
    plt.figure()
    plt.plot(time, Ts, time, Tf, time, Hdot, time, Ts+Tf+Hdot)
    plt.legend(['Ts', 'Tf', 'Hdot', 'Tdelta'])
    #plt.show()
    return Ts + Tf + Hdot

def sync_error(tau, signal1, signal2, time):
    '''Returns the error between two signal time histories.

    Parameters
    ----------
    tau : float
        The time shift.
    signal1 : ndarray, shape(n, )
        The signal that will be interpolated. This signal is
        typically "cleaner" that signal2 and/or has a higher sample rate.
    signal2 : ndarray, shape(n, )
        The signal that will be shifted to syncronize with signal 1.
    time : ndarray
        Time

    Returns
    -------
    error : float
        Error between the two signals for the given tau.

    '''
    # make sure tau isn't too large
    if np.abs(tau) >= time[-1]:
        raise ValueError(('abs(tau), {0}, must be less than or equal to ' +
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

def find_timeshift(niAcc, vnAcc, sampleRate, speed):
    '''Returns the timeshift, tau, of the VectorNav [VN] data relative to the
    National Instruments [NI] data.

    Parameters
    ----------
    NIacc : ndarray, shape(n, )
        The acceleration of the NI accelerometer in its local Y direction.
    VNacc : ndarray, shape(n, )
        The acceleration of the VN-100 in its local Z direction. Should be the
        same length as NIacc and contains the same signal albiet time shifted.
        The VectorNav signal should be leading the NI signal.
    sampleRate : integer
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
        raise StandardError('Signals are not the same length!')

    # make a time vector
    time = time_vector(N, sampleRate)

    # the signals are opposite sign of each other, so fix that
    niSig = -niAcc
    vnSig = vnAcc

    # some constants for find_bump
    wheelbase = 1.02 # this is the wheelbase of the rigid rider bike
    bumpLength = 1.
    cutoff = 50.
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

    # get an initial guess for the time shift based on the bump indice
    guess = (niBump[1] - vnBump[1]) / float(sampleRate)

    # find the section that the bump belongs to
    indices, arrays = split_around_nan(vnSig)
    for pair in indices:
        if pair[0] <= vnBump[1] < pair[1]:
            bSec = pair

    # subtract the mean and normalize both signals
    niSig = normalize(subtract_mean(niSig, hasNans=True), hasNans=True)
    vnSig = normalize(subtract_mean(vnSig, hasNans=True), hasNans=True)

    niBumpSec = niSig[bSec[0]:bSec[1]]
    vnBumpSec = vnSig[bSec[0]:bSec[1]]
    timeBumpSec = time[bSec[0]:bSec[1]]

    if len(niBumpSec) < 200:
        raise Warning('The bump section is mighty small.')

    # set up the error landscape, error vs tau
    # The NI lags the VectorNav and the time shift is typically between 0 and
    # 0.5 seconds
    tauRange = np.linspace(0., .5, num=500)
    error = np.zeros_like(tauRange)
    for i, val in enumerate(tauRange):
        error[i] = sync_error(val, niBumpSec, vnBumpSec, timeBumpSec)

    # find initial condition from landscape
    tau0 = tauRange[np.argmin(error)]

    print "The minimun of the error landscape is %f and the provided guess is %f" % (tau0, guess)

    # if tau is not close to the other guess then say something
    isNone = guess == None
    isInRange = 0. < guess < 1.
    isCloseToTau = guess - .1 < tau0 < guess + .1

    if not isNone and isInRange and not isCloseToTau:
        print("This tau0 may be a bad guess, check the error function!" +
              " Using guess instead.")
        tau0 = guess

    print "Using %f as the guess for minimization." % tau0

    tau  = fmin(sync_error, tau0, args=(niBumpSec, vnBumpSec, timeBumpSec))[0]

    print "This is what came out of the minimization:", tau

    # if the minimization doesn't do a good job, just use the tau0
    if np.abs(tau - tau0) > 0.01:
        tau = tau0
        print "Bad minimizer!! Using the guess, %f, instead." % tau

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

def steer_rate(forkRate, angularRateZ):
    '''Returns the steer rate.'''
    return forkRate - angularRateZ
