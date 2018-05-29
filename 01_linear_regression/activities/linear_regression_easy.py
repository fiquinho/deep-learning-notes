import csv
import random
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt


# Fixing random state for reproducibility
random.seed(1584)


def linear_function(x: List[float], w: float, b: float) -> List[float]:
    """
    Apply the function f(y) = x * w + b to all the values in a list.

    :param x: The list of values to be processed
    :param w: The w parameter of the function (weight)
    :param b: The b parameter of the function (bias)
    :return: A list of the values obtained from the function
    """

    # TODO: Create the linear function
    pass


def mean_squared_error(y: List[float], results: List[float]) -> float:
    """
    Calculate the mean square error from a list of predicted values,
    and a list of real values.

    :param y: A list of real values
    :param results: A list of predicted values
    :return: The mean squared error of the predictions
    """

    # TODO: Create the mean squared error function
    pass


def train_linear_regression_model(data_file: Path=Path(Path.cwd().parent, "data", "notebook_example_data.csv"),
                                  epochs: int= 10000, learning_rate: float=0.005, plot_graph: bool=None):

    # Extract data from file to lists of values
    x = []
    y = []
    with open(data_file, "r") as file:
        reader = csv.DictReader(file)
        for data in reader:
            x.append(float(data["x"]))
            y.append(float(data["y"]))

    # Create the random values of the parameters (w and b)
    # TODO: Create w and b
    w = None
    b = None

    print("The starting weight is: {}".format(w))
    print("The starting bias is: {}".format(b))

    print("Starting function: result = x * {} + {}".format(w, b))

    # Start training
    print()
    print("Training the linear function")
    costs = []
    w_history = []
    b_history = []
    for e in range(epochs):

        # Get the results of the actual linear function
        # TODO: Apply the linear function to the data
        results = None

        # Get the cost of the actual linear function
        # TODO: Get the cost of the actual function
        cost = None

        # Print information every some epochs
        if e % 20 == 0:
            print("Epoch N° = {} - Cost of the actual function = {}".format(e, cost))
            print("Epoch N° = {} - Weight = {} - Bias = {}".format(e, w, b))

        # Save training step values
        costs.append(cost)
        w_history.append(w)
        b_history.append(b)

        # Get the derivative of the cost function with respect to the weight
        # TODO: Get the derivative of the cost function with respect to the weight
        dw = None

        # Get the derivative of the cost function with respect to the bias
        # TODO: Get the derivative of the cost function with respect to the bias
        db = None

        # Update weight and bias
        # TODO: Update weight and bias
        w = None
        b = None

    # Final function
    print()
    print("######## Final results ########")
    print()

    print("The final weight is: {}".format(w))
    print("The final bias is: {}".format(b))

    print("Final function: result = x * {} + {}".format(w, b))
    final_cost = mean_squared_error(y, linear_function(x, w, b))
    print("Cost of the final function = {}".format(final_cost))

    if plot_graph:
        # Three subplots sharing both x/y axes
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        f.subplots_adjust(hspace=0.5, wspace=0.3)

        # Data and line graph
        ax1.plot(x, y, 'o')
        ax1.plot(x, linear_function(x, w, b))
        ax1.set_title("Data points and generated line")
        ax1.grid()

        # Cost graph
        ax2.plot(list(range(epochs)), costs)
        ax2.set_title("Cost of the linear function")
        ax2.grid()

        # Weight graph
        ax3.plot(list(range(epochs)), w_history)
        ax3.set_title("Weight value")
        ax3.grid()

        # Bias graph
        ax4.plot(list(range(epochs)), b_history)
        ax4.set_title("Bias value")
        ax4.grid()

        f.show()


def main():

    train_linear_regression_model(plot_graph=True)


if __name__ == '__main__':
    main()
