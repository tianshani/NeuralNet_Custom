import numpy as np


class Network:
    def __init__(self, loss_func):
        self.layers = []
        self.loss = loss_func

    def add(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        sample_size = len(input_data)
        res = []

        for i in range(sample_size):
            layer_output = input_data[i]
            for layer in self.layers:
                layer_output = layer.forward_propagation(layer_output)
            res.append(layer_output)

        return res
    
    def fit(self, X_train, Y_train, epochs, learning_rate):
        sample_size = len(X_train)
        err = 0

        for i in range(epochs):
            E = 0
            if i > 200: learning_rate = 0.1
            if i > 500: learning_rate = 0.001

            for j in range(sample_size):
                Y_j_pred = X_train[j]
                for layer in self.layers:
                    Y_j_pred = layer.forward_propagation(Y_j_pred)
                err = self.loss(Y_train[j], Y_j_pred)

                E = self.loss(Y_j_pred, Y_train[j], prime=True)

                for layer in reversed(self.layers):
                    E = layer.backward_propagation(E, learning_rate)

            print('Error: ', err/len(X_train), 6)

    