from Neuron import Neuron
import numpy as np

class NeuralNetwork:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

        self.h1 = Neuron(self.weights, self.bias)
        self.h2 = Neuron(self.weights, self.bias)

        self.o1 = Neuron(self.weights, self.bias)

    def feedforward(self, x):
        out1 = self.h1.feedforward(x)
        out2 = self.h2.feedforward(x)

        out3 = self.o1.feedforward(np.array([out1,out2]))

        print out3;

weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 0                   # b = 4
nn = NeuralNetwork(weights, bias)
nn.feedforward(np.array([2,3]))