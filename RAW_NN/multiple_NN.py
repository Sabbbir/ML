inputs = [2, 3, 4]

weight1 = [ 11.1, 29, 15]
weight2 = [ 56, 57, 58]
weight3 = [ 14, 95, 36]

bias1 = 13.2
bias2 = 4.6
bias3 = .5

output = [inputs[0]*weight1[0] + inputs[1]*weight1[1] + inputs[2]*weight1[2] + bias1,
          inputs[0]*weight2[0] + inputs[1]*weight2[1] + inputs[2]*weight2[2] + bias2,
          inputs[0]*weight3[0] + inputs[1]*weight3[1] + inputs[2]*weight3[2] + bias3]
print(output)
