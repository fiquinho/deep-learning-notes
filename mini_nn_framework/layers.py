# TODO: Add docstrings, comments and documentation.

import sys
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent))

from mini_nn_framework.activations import sigmoid, relu, sigmoid_prime, relu_prime
from mini_nn_framework.loss_functions import binary_logistic_regression_loss, binary_logistic_regression_loss_prime


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

        self.dA = None
        self.dZ = None
        self.dW = None
        self.dB = None

    def initialize_parameters(self):
        self.W = np.random.rand(self.neurons, self.input_layer.neurons)
        self.B = np.zeros([self.neurons, 1])

    def print_parameters(self):
        print("Layer {} - W\n{}".format(self.name, self.W))
        print("Layer {} - B\n{}".format(self.name, self.B))

    def forward_pass(self, print_parameters: bool=False):
        self.Z = np.dot(self.W, self.input_layer.output) + self.B

        if self.activation == "sigmoid":
            self.output = sigmoid(self.Z)
        elif self.activation == "relu":
            self.output = relu(self.Z)

        if print_parameters:
            print("Layer {} - Z\n{}".format(self.name, self.Z))
            print("Layer {} - A\n{}".format(self.name, self.output))

    def backward_pass(self, print_parameters: bool=False):

        if self.activation == "sigmoid":
            self.dZ = sigmoid_prime(self.Z) * self.dA
        elif self.activation == "relu":
            self.dZ = relu_prime(self.Z) * self.dA

        self.dW = (np.dot(self.dZ, self.input_layer.output.transpose())) / self.dZ.shape[1]
        self.dB = np.sum(self.dZ, axis=1, keepdims=True)

        if print_parameters:
            print("Layer {} - dA\n{}".format(self.name, self.dA))
            print("Layer {} - dZ\n{}".format(self.name, self.dZ))
            print("Layer {} - dW\n{}".format(self.name, self.dW))
            print("Layer {} - dB\n{}".format(self.name, self.dB))

        if type(self.input_layer) != Input:
            self.input_layer.dA = self.dW.transpose() * self.dZ

    def parameters_update(self, learning_rate: float):

        self.W -= learning_rate * self.dW
        self.B -= learning_rate * self.dB


class OutputBinary(BasicLayer):

    INSTANCES_COUNTER = 0

    def __init__(self, input_layer: BasicLayer, name: str=None):
        super().__init__()

        self.neurons = 1
        self.input_layer = input_layer

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

        self.dA = None
        self.dZ = None
        self.dW = None
        self.dB = None

    def initialize_parameters(self):
        self.W = np.random.rand(self.neurons, self.input_layer.neurons)
        self.B = np.zeros([self.neurons, 1])

    def print_parameters(self):
        print("Layer {} - W\n{}".format(self.name, self.W))
        print("Layer {} - B\n{}".format(self.name, self.B))

    def forward_pass(self, print_parameters: bool=False):
        self.Z = np.dot(self.W, self.input_layer.output) + self.B

        self.output = sigmoid(self.Z)

        if print_parameters:
            print("Layer {} - Z\n{}".format(self.name, self.Z))
            print("Layer {} - A\n{}".format(self.name, self.output))

    def loss(self, targets: np.array) -> np.array:
        self.losses = binary_logistic_regression_loss(labels=targets, predictions=self.output)
        return self.losses

    def cost(self):
        return np.sum(self.losses) / len(self.losses[0])

    def backward_pass(self, targets: np.array, print_parameters: bool=False):
        self.dA = binary_logistic_regression_loss_prime(labels=targets, predictions=self.output)
        self.dZ = sigmoid_prime(self.Z) * self.dA

        self.dW = (np.dot(self.dZ, self.input_layer.output.transpose())) / self.dZ.shape[1]
        self.dB = np.sum(self.dZ, axis=1, keepdims=True)

        if print_parameters:
            print("Layer {} - dA\n{}".format(self.name, self.dA))
            print("Layer {} - dZ\n{}".format(self.name, self.dZ))
            print("Layer {} - dW\n{}".format(self.name, self.dW))
            print("Layer {} - dB\n{}".format(self.name, self.dB))

        if type(self.input_layer) != Input:
            self.input_layer.dA = self.dW.transpose() * self.dZ

    def parameters_update(self, learning_rate: float):

        self.W -= learning_rate * self.dW
        self.B -= learning_rate * self.dB


# np.random.seed(0)
#
# data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).transpose()
# labels = np.array([[0, 0, 1, 1]])
# print("Data:\n {}".format(data))
# print("Labels:\n {}".format(labels))
#
# print("\n------------------------------\n")
# print("## Layers creation ##\n")
#
# inputs_layer = Input(2)
# # fully_layer = FullyConnected(neurons=4, input_layer=inputs_layer, activation="sigmoid")
# # fully_layer.print_parameters()
# output_layer = OutputBinary(input_layer=inputs_layer)
# output_layer.print_parameters()
#
# print("\n------------------------------\n")
# print("## Forward pass ##\n")
#
# inputs_layer.feed_input(data)
# # fully_layer.forward_pass(print_parameters=True)
# output_layer.forward_pass(print_parameters=True)
#
# print("\n------------------------------\n")
# print("## Loss and cost computation ##\n")
# loss = output_layer.loss(targets=labels)
# print("Losses\n{}".format(loss))
# cost = output_layer.cost()
# print("Cost\n{}".format(cost))
#
# print("\n------------------------------\n")
# print("## Backward pass ##\n")
# output_layer.backward_pass(targets=labels, print_parameters=True)
# output_layer.parameters_update(learning_rate=0.01)
#
# print("\n------------------------------\n")
# print("## Parameters update and new forward pass ##\n")
#
# output_layer.print_parameters()
# output_layer.forward_pass(print_parameters=True)
