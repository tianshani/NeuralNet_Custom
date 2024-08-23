import numpy as np

class LossFunctions:
    def MSE(predicted, actual, prime=False):
        if prime:
            return 2 * (predicted - actual) / len(actual)
        return np.mean( np.square(predicted - actual) )