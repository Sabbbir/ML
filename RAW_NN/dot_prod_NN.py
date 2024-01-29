import numpy as np

inputs = [1,2,3]
weights = [[ 11.1, 29, 15],
           [ 56, 57, 58],
           [ 14, 95, 36]]

biases = [2, 3, 4]

# output = np.dot(weights, inputs) + biases
outputs = np.dot(inputs, weights) + biases
# print(output)
print(outputs)