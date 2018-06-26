import numpy as np

def binary_logistic_regression_loss(y: np.array, predictions: np.array) -> np.array:
    return -(y * np.log(predictions)) - (y - 1) * np.log(1 - predictions)

