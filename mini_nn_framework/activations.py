from copy import deepcopy
import numpy as np


def sigmoid(x: np.array) -> np.array:
    """
    Apply a sigmoid function to a matrix of values, element wise.

    :param x: The numpy matrix.
    :return: A numpy matrix with the generated values.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x: np.array) -> np.array:
    """
    Apply the derivative of the sigmoid function to a matrix of values, element wise.

    :param x: The The numpy matrix.
    :return: A numpy matrix with the generated values.
    """
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x: np.array) -> np.array:
    """
    Apply the rectified linear unit function to a matrix of values, element wise.

    :param x: The numpy matrix.
    :return: A numpy matrix with the generated values.
    """
    return np.maximum(x, 0)


def relu_prime(x: np.array) -> np.array:
    """
    Apply the derivative of the rectified linear unit function to a matrix of values,
    element wise.

    :param x: The numpy matrix.
    :return: A numpy matrix with the generated values.
    """
    result = deepcopy(x)

    result[result <= 0] = 0
    result[result > 0] = 1
    return result


# # Test cases
# print(sigmoid(np.array([[-4, 0, 4]])))
# print(sigmoid_prime(np.array([[-4, 0, 4]])))
#
# print(relu(np.array([[-4, 0, 4]])))
# print(relu_prime(np.array([[-4, 0, 4]])))
