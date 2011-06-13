import DataProcessor as dp
import numpy as np

def test_unsize_vector():
    n = 3
    a = np.ones(n)
    b = np.append(a, np.array([np.nan, np.nan]))
    c = dp.unsize_vector(a, n)
    assert (a == c).all()

def test_time_vector():
    numSamples = 100
    sampleRate = 50
    time = dp.time_vector(numSamples, sampleRate)
    assert (time == np.linspace(0., 2. - 1. / 50., num=100)).all()

def test_split_around_nan():
    # build an array of length 25 with some nan values
    a = np.ones(25) * np.nan
    b = np.arange(25)
    for i in b:
        if i not in [0, 5, 20, 24]:
            a[i] = b[i]
    # run the function and test the results
    indices, arrays = dp.split_around_nan(a)
    assert len(indices) == 7
    assert indices[0] == (0, 1)
    assert indices[1] == (1, 5)
    assert indices[2] == (5, 6)
    assert indices[3] == (6, 20)
    assert indices[4] == (20, 21)
    assert indices[5] == (21, 24)
    assert indices[6] == (24, 25)
    # build an array of length 25 with some nan values
    a = np.ones(25) * np.nan
    b = np.arange(25)
    for i in b:
        if i not in [5, 20]:
            a[i] = b[i]
    # run the function and test the results
    indices, arrays = dp.split_around_nan(a)
    assert len(indices) == 5
    assert indices[0] == (0, 5)
    assert indices[1] == (5, 6)
    assert indices[2] == (6, 20)
    assert indices[3] == (20, 21)
    assert indices[4] == (21, 25)
    a = np.array([np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, np.nan])
    # run the function and test the results
    indices, arrays = dp.split_around_nan(a)
    assert len(indices) == 6
    assert indices[0] == (0, 1)
    assert indices[1] == (1, 4)
    assert indices[2] == (4, 5)
    assert indices[3] == (5, 6)
    assert indices[4] == (6, 8)
    assert indices[5] == (8, 9)

def test_vnav_checksum():
    s = ('$VNCMV,' +
         '+1.045402E+00,+6.071629E-01,-4.171332E-01,' +
         '+3.659531E+00,-3.211054E-01,-9.641082E+00,' +
         '+9.549000E-03,-2.366146E-02,-2.832810E-02,' +
         '+3.179132E+02'
         '*4F\r\n')
    assert dp.vnav_checksum(s) == '4F'

    s = '$VNRRG,04,1*6A\r\n'
    assert dp.vnav_checksum(s) == '6A'

    s = '$VNRRG,08,-027.33,-005.33,+002.63*65\r\n'
    assert dp.vnav_checksum(s) == '65'

def test_parse_vnav_string():
    strings = []
    # this is an example output from a read register (VNRRG) command
    s = ('$VNRRG,' +
         '252,' +
         '+2.088900E-01,-5.476115E-01,-1.835284E+00,' +
         '+2.077137E-02,+3.030512E-01,-9.876616E+00,' +
         '+7.808615E-02,+9.382707E-02,+1.179649E-02,' +
         '+2.099097E+01*45\r\n')
    vnList, chkPass, vnrrg = dp.parse_vnav_string(s)
    print vnList
    l = ['VNRRG', '252',
         '+2.088900E-01', '-5.476115E-01', '-1.835284E+00',
         '+2.077137E-02', '+3.030512E-01', '-9.876616E+00',
         '+7.808615E-02', '+9.382707E-02', '+1.179649E-02',
         '+2.099097E+01']
    assert vnList == l
    assert chkPass
    assert vnrrg
    # this is another VNRRG example with a string instead of floats
    strings.append('$VNRRG,03,067200383733335843046264*58\r\n')

    # this is an example of an asyn output
    strings.append('$VNCMV,' +
                   '+1.070661E+00,+6.999261E-01,-1.310137E-01,' +
                   '+2.804529E+00,-3.269150E-01,-7.918662E+00,' +
                   '+8.545322E-02,-3.140700E-02,-5.392097E-02,' +
                   '+3.179056E+02*41\r\n')

    # these are examples of ones with different line endings
    strings.append('$VNRRG,5,9600*65')
    strings.append('$VNRRG,5,9600*65\r')
    strings.append('$VNRRG,5,9600*65\n')

    # here are a couple of typical corrupt ones
    strings.append('987319E-01,-1.340267E-01,' +
                   '+3.290921E+00,-6.257143E-02,-8.909048E+00,' +
                   '+8.488191E-02,-2.269540E-02,-5.960917E-02,' +
                   '+3.179056E+02*42\r\n')
    strings.append('$VNCMV,' +
                   '+1.080685E+00,+7.243825E-01,-1.458829E-01,' +
                   '+3.421932E+00,' +
                   '+1.361301E+00611385E-01,' + # this the bad one
                   '-8.552538E+00,' +
                   '-4.502851E-02,-2.731314E-02,+1.610722E-01,' +
                   '+3.179069E+02*4F\r\n')
