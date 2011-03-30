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
