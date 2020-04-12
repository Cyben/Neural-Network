import numpy as np
from Neuron import Neuron
from HiddenLayer import HiddenLayer

def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length.
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork:
    def __init__(self, data, all_y_trues, neurons_in_hl=[0]):
        # Data Members
        self.data = data
        self.all_y_trues = all_y_trues
        self.hidden_layers = []

        # # Hidden Layer 1
        # self.hl = HiddenLayer(neurons_in_hl[0])

        # For each hidden layer save in the hidden_layer array
        amount_of_weights = len(self.data[0])
        for hl_count in range(len(neurons_in_hl)):
            hl = HiddenLayer(neurons_in_hl[hl_count])
            for neuron in hl.neurons():
                neuron.changeProps([np.random.normal() for i in range(amount_of_weights)], np.random.normal())

            amount_of_weights = neurons_in_hl[hl_count]
            self.hidden_layers.append(hl)

        # Output Neuron
        self.o1 = Neuron([np.random.normal() for i in range(amount_of_weights)], np.random.normal())

    def feedforward(self, x):
        input_data = x
        for hl in self.hidden_layers:
            input_data = hl.feedforward(input_data)

        return self.o1.feedforward(input_data)

    def train(self):

        learn_rate = 0.01
        epochs = 10000  # number of times to loop through the entire dataset
        for epoch in range(epochs):
            for x, y_true in zip(self.data, self.all_y_trues):

                # feedforward
                y_pred = self.feedforward(x)

                # --- Calculate partial derivatives.
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                derived_sum_o1 = self.o1.derivedTotal()

                for neuron_count, neuron in enumerate(self.hidden_layers[-1].neurons()):
                    self.o1.changeWeightInIndex(neuron_count, self.o1.weightInIndex(
                        neuron_count) - learn_rate * d_L_d_ypred * neuron.getFeedforwardResualt() * derived_sum_o1)

                self.o1.changeBias(self.o1.bias() - learn_rate * d_L_d_ypred * derived_sum_o1)

                input_arr = x
                for hl in self.hidden_layers:
                    for neuron in hl.neurons():
                        derived_sum = neuron.derivedTotal()
                        d_ypred_d_hl = self.o1.weightInIndex(0) * derived_sum_o1

                        for weight_count in range(len(neuron.weights())):
                            neuron.changeWeightInIndex(weight_count, neuron.weightInIndex(
                                weight_count) - learn_rate * d_L_d_ypred * d_ypred_d_hl * input_arr[
                                                           weight_count] * derived_sum)
                        neuron.changeBias(neuron.bias() - learn_rate * d_L_d_ypred * d_ypred_d_hl * derived_sum)

                    input_arr = hl.getFeedforwardResualt()

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
