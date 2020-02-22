import Neuron


class NeuralNetwork:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        

    def feedforward(self):
        pass



weights = np.array([0, 1]) # w1 = 0, w2 = 1
bias = 4                   # b = 4
nn = NeuralNetwork()