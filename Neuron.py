import numpy as np

def activationFunction(functionInput):
    # On this version the activation function is a
    # Sigmoid function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-functionInput))

def derive_activationFunction(functionInput):
  # Derivative of activation function: f'(x) = f(x) * (1 - f(x))
  fx = activationFunction(functionInput)
  return fx * (1 - fx)

class Neuron:
    def __init__(self, weights = None, bias = None):
        self.changeProps(weights,bias)
        self.total_sum = 0
        self.feedforwardResualt = 0


    def calcTotalSum(self, inputs):
        self.total_sum = np.dot(self.__weights, inputs) + self.__bias

    def feedforward(self, inputs):
        self.calcTotalSum(inputs)
        self.feedforwardResualt = activationFunction(self.total_sum)
        return self.feedforwardResualt

    def getFeedforwardResualt(self):
        return self.feedforwardResualt

    def derivedTotal(self):
        return derive_activationFunction(self.total_sum)

    def __changeWeights(self, weights):
        self.__weights = weights

    def changeWeightInIndex(self, index, weight):
        self.__weights[index] = weight

    def changeBias(self, bias):
        self.__bias = bias

    def changeProps(self, weights, bias):
        self.__changeWeights(weights);
        self.changeBias(bias)

    def weights(self):
        return self.__weights

    def weightInIndex(self, index):
        return self.weights()[index]

    def bias(self):
        return self.__bias;


# weights = np.array([0, 1]) # w1 = 0, w2 = 1
# bias = 4                   # b = 4
# n = Neuron(weights, bias)
#
# x = np.array([2, 3])       # x1 = 2, x2 = 3
# print(n.feedforward(x))    # 0.9990889488055994
