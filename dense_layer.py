import numpy as np
from layer import Layer

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size) - 0.5
        self.B = np.random.randn(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        # self.X = input_data.reshape((input_data.shape[0], 1))
        self.X = input_data
        self.Y = np.dot(self.X, self.W) + self.B

        return self.Y
    
    def backward_propagation(self, E_gradient, learning_rate):
        X_error = np.dot(E_gradient, self.W.T)  # dE/dX = dE/dY * dZ/dX = dE/dY * W.T
        W_error = np.dot(self.X.T, E_gradient)  # dE/dW = dE/dY * dZ/dW = X.T * dE/dY
        B_error = E_gradient  # dE/dB = dE/dY * dY/dB = dE/dY
        # print(f'W: {self.W}\n E:{E_gradient}\n WE:{W_error}')
        
        self.W -= learning_rate * W_error
        self.B -= learning_rate * B_error

        return X_error

