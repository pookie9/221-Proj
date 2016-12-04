import numpy as np

def mulaw(x, mu=255):
    result = np.sign(x) * np.log(1 + mu*abs(x)) / np.log(1 + mu)
    return int((result + 1) / 2.0 * mu) 

def inverse_mulaw(x, mu=255.0):
    result = ((x / mu) - 0.5) * 2.0
    return (np.sign(result) * (1/mu) * ((1 + mu)**abs(result) - 1))
