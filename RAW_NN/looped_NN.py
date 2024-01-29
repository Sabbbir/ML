
inputs = [2, 3, 4]
weights = [[ 11.1, 29, 15],
           [ 56, 57, 58],
           [ 14, 95, 36]]
biases = [13.2, 4.6, .5]

layer_output = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)
        