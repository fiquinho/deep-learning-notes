import numpy as np


def binary_logistic_regression_loss(labels: np.array, predictions: np.array) -> np.array:
    """
    Get the losses for the binary logistic regression task, from a matrix with real
    labels and a matrix with predictions.

    :param labels: A matrix with the true labels.
    :param predictions: A matrix with the predictions.
    :return: A matrix with the losses.
    """
    return -(labels * np.log(predictions)) - ((1 - labels) * np.log(1 - predictions))


def binary_logistic_regression_loss_prime(labels: np.array, predictions: np.array) -> np.array:
    """
    Get the derivatives of the losses for the binary logistic regression task, with respect to
    the predictions, from a matrix with real labels and a matrix with predictions.

    :param labels: A matrix with the true labels.
    :param predictions: A matrix with the predictions.
    :return: A matrix with the derivative of the losses with respect to the predictions.
    """
    return -(labels * (1 / predictions)) + ((1 - labels) / (1 - predictions))
