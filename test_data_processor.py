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
    assert indices[0] == (1, 5)
    assert indices[1] == (6, 20)
    assert indices[2] == (21, 24)
    # build an array of length 25 with some nan values
    a = np.ones(25) * np.nan
    b = np.arange(25)
    for i in b:
        if i not in [5, 20]:
            a[i] = b[i]
    # run the function and test the results
    indices, arrays = dp.split_around_nan(a)
    assert indices[0] == (0, 5)
    assert indices[1] == (6, 20)
    assert indices[2] == (21, 25)
