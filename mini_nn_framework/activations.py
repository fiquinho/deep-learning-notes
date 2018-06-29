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

# # Test cases
# print(sigmoid(np.array([[-4, 0, 4]])))
# print(sigmoid_prime(np.array([[-4, 0, 4]])))
#
# print(relu(np.array([[-4, 0, 4]])))
