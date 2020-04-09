import numpy as np
from Neuron import Neuron
from HiddenLayer import HiddenLayer

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()

class OurNeuralNetwork:

    def __init__(self, data, all_y_trues, neurons_in_HL = 0):
        # Data Members
        self.data = data
        self.all_y_trues = all_y_trues
        self.hidden_layers = []

        # Hidden Layer 1
        self.hl = HiddenLayer(neurons_in_HL[0])

        for neuron in self.hl.neurons():
            neuron.changeProps([np.random.normal() for i in range(len(self.data[0]))], np.random.normal())

        # Output Neuron
        self.o1 = Neuron([np.random.normal() for i in range(len(self.data[0]))], np.random.normal())

    def feedforward(self, x):
        return self.o1.feedforward([neuron.feedforward(x) for neuron in self.hl.neurons()])

    def train(self):

        learn_rate = 0.1
        epochs = 1000  # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(self.data, self.all_y_trues):

                # feedforward
                y_pred = self.feedforward(x)

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                derived_sum_o1 = self.o1.derivedTotal()

                d_ypred_d_w5 = self.hl.neuronByIndex(0).feedforward(x) * derived_sum_o1
                d_ypred_d_w6 = self.hl.neuronByIndex(1).feedforward(x) * derived_sum_o1
                d_ypred_d_b3 = derived_sum_o1

                for neuron in self.hl.neurons():
                    derived_sum = neuron.derivedTotal()
                    d_ypred_d_hl = self.o1.weightInIndex(0) * derived_sum_o1

                    d_hl_d_w1 = x[0] * derived_sum
                    d_hl_d_w2 = x[1] * derived_sum
                    d_hl_d_b  = derived_sum

                    neuron.changeWeightInIndex(0, neuron.weightInIndex(0) - learn_rate * d_L_d_ypred * d_ypred_d_hl * d_hl_d_w1)
                    neuron.changeWeightInIndex(1, neuron.weightInIndex(1) - learn_rate * d_L_d_ypred * d_ypred_d_hl * d_hl_d_w2)
                    neuron.changeBias(neuron.bias() - learn_rate * d_L_d_ypred * d_ypred_d_hl * d_hl_d_b)

                # Neuron o1
                self.o1.changeWeightInIndex(0, self.o1.weightInIndex(0) - learn_rate * d_L_d_ypred * d_ypred_d_w5)
                self.o1.changeWeightInIndex(1, self.o1.weightInIndex(1) - learn_rate * d_L_d_ypred * d_ypred_d_w6)
                self.o1.changeBias(self.o1.bias() - learn_rate * d_L_d_ypred * d_ypred_d_b3)

            # --- Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, self.data)
                loss = mse_loss(self.all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))

# Define dataset
data = np.array([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6],  # Diana
])
all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1,  # Diana
])

# Train our neural network!
network = OurNeuralNetwork(data, all_y_trues, [2])
network.train()

# Make some predictions
emily = np.array([-7, -3])  # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M
