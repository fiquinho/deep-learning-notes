import os
import sys
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt


# Fixing random state for reproducibility
random.seed(1584)


def train_linear_regression_model(data_file: Path) -> (float, float):

    # Extract data from file to lists of values
    x = []
    y = []
    with open(data_file, "r") as file:
        reader = csv.DictReader(file)
        for data in reader:
            x.append(float(data["x"]))
            y.append(float(data["y"]))

    print("Data points: x = {}".format(x))
    print("Data points: y = {}".format(y))

    # TODO: Finish the function. It should return a final "weight" and
    # TODO: "bias" for the trained model.

    weight = None
    bias = None

    return weight, bias


def plot_model(data_file: Path, w: float, b: float):

    # Extract data from file to lists of values
    x = []
    y = []
    with open(data_file, "r") as file:
        reader = csv.DictReader(file)
        for data in reader:
            x.append(float(data["x"]))
            y.append(float(data["y"]))

    # Data and line graph
    plt.figure(figsize=(5, 5))
    plt.plot(x, y, 'o')
    plt.plot(x, [value * w + b for value in x])
    plt.title("Data points and generated line")
    plt.grid()

    plt.show()


def main():

    data_file = Path("data.csv")
    weight, bias = train_linear_regression_model(data_file)

    print("The final weight is: {}".format(weight))
    print("The final bias is: {}".format(bias))

    plot_model(data_file, weight, bias)


if __name__ == '__main__':
    main()
