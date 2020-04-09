from NeuralNetwork import NeuralNetwork
import numpy as np

weights = np.array([[0, 1], [0, 1]])  # w1 = 0, w2 = 1
bias = [0, 0]  # b = 0
nn = NeuralNetwork([[-2, -1],  # Alice
                    [25, 6]],[1,0], 1, [2])
# print(nn.feedforward(np.array([[-2, -1],  # Alice
#                     [25, 6]])))

nn.train()

# nn = NerualNetwork()
