import numpy as np

# from keras import models, layers, utils, backend as K


def calculate_layer_output(i, w, layer_id, print_summary=False):
    # Shaped data: features(x)
    x = i.reshape(5,1)

    # Sum of the x * w values
    sum_result = []

    for idx, i in enumerate(w1):
        sum_result.append(sum(x * y for x, y in zip(x, w[idx]))[0])

    # Activation function f(x). Output of the perceptron
    # if x > 0 f(x) = 1
    # if x < 0 f(x) = 0
    outputs = []

    for idx, i in enumerate(sum_result):
        output = 1 if sum_result[idx] > 1 else 0
        outputs.append(output)

        if print_summary is True:
            print(
                'Perceptron: {}-{} | Activation Function Result: {} | Output: {}'
                .format(str(layer_id), idx+1, str(sum_result[idx]), str(output)))

    return np.array(outputs)


def calculate_neuro_net_output(i, print_summary=True):
    sum_result = np.sum(i)

    output = 1 if sum_result > 1 else 0

    if print_summary is True:
        print('Network output | Activation Function Result: {} | Output: {}'.format(str(sum_result), str(output)))

    return output


# First layer inputs and calculations
i1 = np.array([0.5, 0.1, 0, 0, 0.1])
w1 = np.array([
              [1, 1, 1, 1, 1],
              [2, 3, 5, 1, 2],
              [1, 0.5, 0.4, 2, 0.77],
              [2, 0.51, 0.422, 5, 2],
              [3, 4, 0.1, 0.2, 0.3]
])


o1 = calculate_layer_output(i1, w1, 1, print_summary=True)

# Second layer inputs and calculations
# Input data for the second layer is the output of the first layer
i2 = o1
w2 = np.array([
             [0, 0, 0, 0, 0],
             [1, 2, 3, 4, 5],
             [0.1, 0.22, 3, 4, 5],
             [0.11, 22, 31, 4, 5],
             [0.12, 0.22, 0.31, 0, 0]
])

o2 = calculate_layer_output(i2, w2, 2, print_summary=True)

# Calculate output of the neural network
calculate_neuro_net_output(np.array(o2))
