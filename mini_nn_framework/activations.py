import numpy as np


def sigmoid(x: np.array) -> np.array:
    """
    Apply a sigmoid function to an 1xm array of values, element wise.

    :param x: The 1xm array.
    :return: An array with the generated values.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.array) -> np.array:
    """
    Apply the derivative of the sigmoid function to an 1xm array of values, element wise.

    :param x: The 1xm array.
    :return: An array with the generated values.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: np.array) -> np.array:
    """
    Apply the ReLU function to an 1xm array of values, element wise.

    :param x: The 1xm array.
    :return: An array with the generated values.
    """
    result = [[]]

    for value in x[0]:
        if value > 0:
            result[0].append(value)
        else:
            result[0].append(0)

    return np.array(result)

# # Test cases
# print(sigmoid(np.array([[-4, 0, 4]])))
# print(sigmoid_prime(np.array([[-4, 0, 4]])))
#
# print(relu(np.array([[-4, 0, 4]])))
