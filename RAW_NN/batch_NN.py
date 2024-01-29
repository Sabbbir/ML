import numpy as np

X = [[1, 2, 3, 4],
     [2, 3, 4, 5],
     [5, 6, 7, 8]]

weights = [[11, 29, 15, 16],
           [56, 57, 58, 45],
           [14, 95, 36, 34]]

weights2 = [[11, 29, 15],
           [56, 57, 58],
           [14, 95, 36]]

bias =  [3, 4, 5]
bias2 = [1, 2, 3]

layer1_output = np.dot(X, np.array(weights).T) + bias
layer2_output = np.dot(layer1_output, np.array(weights2).T) + bias2
print(layer2_output)
