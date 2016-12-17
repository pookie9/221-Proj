import numpy as np

def mulaw(x, mu=255):
    eye = np.eye(mu+1)
    result = np.sign(x) * np.log(1 + mu*abs(x)) / np.log(1 + mu)
    return eye[int((result + 1) / 2.0 * mu)]

def inverse_mulaw(x, mu=255.0):
    result = ((x.index(1) / mu) - 0.5) * 2.0
    return (np.sign(result) * (1/mu) * ((1 + mu)**abs(result) - 1))
