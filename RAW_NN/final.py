
import numpy as np
import matplotlib.pyplot as plt
from verticalDataset import create_data
from loss_func import *
X, y = create_data(100, 3)

dense1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_ReLU()

loss_function = Loss_CategoricalCrossEntropy()

lowest_loss = 9999999
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.weights.copy()

for iteration in range(10000):
    dense1.weights += .05 * np.random.randn(2, 3)
    dense1.biases += .05 * np.random.randn(1, 3)
    dense2.weights += .05 * np.random.randn(3, 3)
    dense2.biases += .05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis = 1)
    accuracy = np.mean(predictions==y)
   
    if loss<lowest_loss:
        print("New Weights found, iteration:", iteration, " loss:", loss, " acc:", accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    
    else:

        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
        print("Iteration:", iteration, " Loss:", loss, " Accuracy:", accuracy)



