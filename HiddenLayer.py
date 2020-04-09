import numpy as np
from Neuron import Neuron

class HiddenLayer:
    def __init__(self, neurons_count = 0):
        self.__neurons = []

        for neuron_number in range(neurons_count):
            self.__n = Neuron()
            self.__neurons.append(self.__n)

    def feedforward(self, inputs):
        for neuron in self.__neurons:
            self.output.append(neuron.feedforward(inputs))

        return self.output

    def neuronByIndex(self, index):
        return self.__neurons[index]

    def neurons(self):
        return self.__neurons