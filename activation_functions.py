import numpy as np

class ActivationFunctions:
    def ReLU(x, prime=False):
        if prime:
            return x>=0
        return np.maximum(0, x)
    
    def PReLU(x, slope=0.01, prime=False):
        if prime:
            return slope if x < 0 else 1
        return slope*x if x < 0 else x
    
    def sigm(x, prime=False):
        if prime:
            return np.exp(-x) / (1 + np.exp(-x))
        return 1 / (1 + np.exp(-x))
    
    def tanh(x, prime=False):
        if prime:
            return 1 - np.square(np.tanh(x))
        return np.tanh(x)