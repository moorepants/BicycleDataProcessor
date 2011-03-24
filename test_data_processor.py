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
    # 1 to 25
    a = np.arange(1, 26)
    for i in [0, 5, 20, 24]:
        a[i] = np.nan
    arrays, indices = split_around_nan(a)
    assert arrays[0] = a[1:5]
    assert arrays[1] = a[6:20]
    assert arrays[2] = a[21:24]
    assert indices[0] = (0, 0)
    assert indices[1] = (6, 20)
    assert indices[2] = (21, 24)


