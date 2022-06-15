import basic_nn
import numpy as np

raw_data = np.array([nn.output, 0.5, 0, 0, 0])

# Shaped data: features(x) and weights(w)
x = raw_data.reshape(5, 1)
w = np.array([1, 1, 1, 1, 1])

# Sum of the x * w values
sum_result = sum(x * y for x, y in zip(x, w))[0]

# Activation function f(x). Out put of the perceptron
# if x > 0 f(x) = 1
# if x < 0 f(x) = 0
output = 1 if sum_result > 1 else 0


print('Sum: ', sum_result)
print('Output: ', output)
