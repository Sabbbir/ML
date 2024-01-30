import numpy as np
from dataset import create_data
np.random.seed(0)

X = [[1,2,3,5],
     [2,5,1,2],
     [1.5,2.7,3.3, .8]]

X, y = create_data(100, 3)

# inputs = [-.40,2,3,4,5,10,2,-100]

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weight = 0.1*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weight)+ self.biases



layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)

activation1.forward(X)
# print(layer1.output)
print(activation1.output)
# activation1.forward(layer1.output)
# print(layer1.output)

