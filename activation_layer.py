from layer import Layer
import numpy as np

class ActivationLayer(Layer):
    def __init__(self, activation_function):
        self.active = activation_function

    def forward_propagation(self, input_data):
        self.X = input_data
        return self.active(input_data)
    
    def backward_propagation(self, E_gradient, learning_rate):
        return self.active(self.X, prime=True) * E_gradient

