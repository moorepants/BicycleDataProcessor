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
