# TODO: Add docstrings, comments and documentation.

import sys
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent))

from mini_nn_framework.activations import sigmoid, relu
from mini_nn_framework.loss_functions import binary_logistic_regression_loss


np.random.seed(0)


class BasicLayer(object):

    def __init__(self):

        self.output = None
        self.neurons = None
        self.name = None

    def __str__(self):
        return self.name


class Input(BasicLayer):

    INSTANCES_COUNTER = 0

    def __init__(self, input_size: int, name: str=None):
        super().__init__()

        self.neurons = input_size

        self.output_layer = None

        if name is None:
            self.name = "input_{}".format(Input.INSTANCES_COUNTER)
            Input.INSTANCES_COUNTER += 1
        else:
            self.name = name

    def feed_input(self, inputs: np.array):

        if inputs.shape[0] != self.neurons:
            raise ValueError("The inputs fed to {} should be of shape [{}, m]."
                             "Found [{}, m] instead.".format(self.name, self.neurons, inputs.shape[0]))

        self.output = inputs


class FullyConnected(BasicLayer):

    INSTANCES_COUNTER = 0

    def __init__(self, neurons: int, input_layer: BasicLayer, activation: str="sigmoid",
                 name: str=None):
        super().__init__()

        self.neurons = neurons
        self.input_layer = input_layer
        self.activation = activation

        self.output_layer = None
        self.input_layer.output_layer = self

        if name is None:
            self.name = "fully_connected_{}".format(FullyConnected.INSTANCES_COUNTER)
            FullyConnected.INSTANCES_COUNTER += 1
        else:
            self.name = name

        self.W = None
        self.B = None

        self.initialize_parameters()

        self.Z = None

    def initialize_parameters(self):
        self.W = np.random.rand(self.neurons, self.input_layer.neurons)
        self.B = np.zeros([self.neurons, 1])

    def forward_pass(self, print_parameters: bool=False):
        self.Z = np.dot(self.W, self.input_layer.output) + self.B

        if self.activation == "sigmoid":
            self.output = sigmoid(self.Z)
        elif self.activation == "relu":
            self.output = relu(self.Z)

        if print_parameters:
            print("Layer {} - Z\n{}".format(self.name, self.Z))
            print("Layer {} - A\n{}".format(self.name, self.output))


class OutputBinary(BasicLayer):

    INSTANCES_COUNTER = 0

    def __init__(self, input_layer: BasicLayer, activation: str="sigmoid", name: str=None):
        super().__init__()

        self.neurons = 1
        self.input_layer = input_layer
        self.activation = activation

        self.input_layer.output_layer = self

        if name is None:
            self.name = "binary_output_{}".format(OutputBinary.INSTANCES_COUNTER)
            OutputBinary.INSTANCES_COUNTER += 1
        else:
            self.name = name

        self.W = None
        self.B = None

        self.initialize_parameters()

        self.Z = None
        self.losses = None

    def initialize_parameters(self):
        self.W = np.random.rand(self.neurons, self.input_layer.neurons)
        self.B = np.zeros([self.neurons, 1])

    def forward_pass(self, print_parameters: bool=False):
        self.Z = np.dot(self.W, self.input_layer.output) + self.B

        if self.activation == "sigmoid":
            self.output = sigmoid(self.Z)
        elif self.activation == "relu":
            self.output = relu(self.Z)

        if print_parameters:
            print("Layer {} - Z\n{}".format(self.name, self.Z))
            print("Layer {} - A\n{}".format(self.name, self.output))

    def loss(self, targets: np.array) -> np.array:
        self.losses = binary_logistic_regression_loss(labels=targets, predictions=self.output)
        return self.losses

    def cost(self):
        return np.sum(self.losses) / len(self.losses[0])


data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).transpose()
labels = np.array([[0, 0, 1, 1]])
print("Data:\n {}".format(data))
print("Labels:\n {}".format(labels))

print("\n------------------------------\n")

inputs_layer = Input(2)
fully_layer = FullyConnected(4, inputs_layer, "sigmoid")
output_layer = OutputBinary(inputs_layer, "sigmoid")

print("Layer {} - W\n{}".format(output_layer, output_layer.W))
print("Layer {} - B\n{}".format(output_layer, output_layer.B))

inputs_layer.feed_input(data)
fully_layer.forward_pass(print_parameters=True)
output_layer.forward_pass(print_parameters=True)

print("\n------------------------------\n")

loss = output_layer.loss(targets=labels)
print("Losses\n{}".format(loss))

cost = output_layer.cost()
print("Cost\n{}".format(cost))
