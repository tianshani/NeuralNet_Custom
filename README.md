# Neural Networks from scratch
This is a small library built to better understand Machine Learning algorithms. This is not final version and I'll try to add new features as my progress goes further in this field.
## Architecture
Library is build mainly on NumPy without using ML APIs.
### Network
Head class for creating networks. Initialization requires only loss function as an argument. 
With add() we add new layers to the network. 
predict() is simply forward propagation through all layers and data points, it returns list of all output Ys for each data point X.
fit() takes X and Y training data, amount of epochs and learning rate passing forward through every layer, calculating error with derivative of self.loss and then passing it to backward_propagation for each layer. (also it prints final error)
### Layer()
Layer() is an abstract class for layers which defines of 3 functions. \_\_init__() for input X and output Y of the layer, forward() and backward() for respective propagations.
### DenseLayer()
This is main class for fully-connected layers. It initializes random weight matrix W and 0 bias vector B for given input/output dimensions. 
forward-propagation() is simply WX+B and backward utilizes chain-rule for given error E gradient from previous layer. 
backward-propagation() calculates error with respect to weights and biases, updating respective variables with given learning rate using vanilla gradient descent. This function returns error with respect to X to pass it to next layer in backward propagation.
### ActivationLayer()
\_\_init__() takes only activation function and assigns it to self.active. 
forward_propagation() passes input data throught given function and backward_propagation() returns activation function's derivative times gradient of Error.
### ActivationFunctions() and LossFunctions()
Just decided to isolate activation functions and loss functions to separate classes for convinience.
