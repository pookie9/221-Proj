import numpy as np

def mulaw(x, mu=255):
    result = np.sign(x) * np.log(1 + mu*abs(x)) / np.log(1 + mu)
    return one_hot(int((result + 1) / 2.0 * mu), mu+1)

def inverse_mulaw(x, mu=255.0):
    result = ((list(x).index(1) / mu) - 0.5) * 2.0
    return (np.sign(result) * (1/mu) * ((1 + mu)**abs(result) - 1))

def one_hot(x, size):
    return np.eye(size)[x]
