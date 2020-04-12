import numpy as np
from Neuron import Neuron

class HiddenLayer:
    def __init__(self, neurons_count = 0):
        self.__neurons = []
        self.feedforwardResualt = []
        for neuron_number in range(neurons_count):
            self.__n = Neuron()
            self.__neurons.append(self.__n)

    def feedforward(self, inputs):
        self.feedforwardResualt = [neuron.feedforward(inputs) for neuron in self.neurons()]
        return self.getFeedforwardResualt()

    def getFeedforwardResualt(self):
        return self.feedforwardResualt

    def neuronByIndex(self, index):
        return self.__neurons[index]

    def neurons(self):
        return self.__neurons