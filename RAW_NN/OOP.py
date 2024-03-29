import numpy as np
np.random.seed(0)

X = [[1, 2, 3, 4],
     [2, 3, 4, 5],
     [5, 6, 7, 8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight)+ self.biases


layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 3)
layer3 = Layer_Dense(3, 6)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
# print(layer2.output)
layer3.forward(layer2.output)
print(layer3.output)
