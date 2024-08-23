import numpy as np

class Layer:
    def __init(self):
        self.input = None
        self.output = None

        def forward(self, input_data):
            raise NotImplementedError

        def backward(self, E_gradient, learning_rate):
            raise NotImplementedError
            