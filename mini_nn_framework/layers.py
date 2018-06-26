import sys
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent))

from mini_nn_framework.activations import sigmoid, relu


np.random.seed(0)


class BasicLayer(object):

    def __init__(self):

        self.output = None
        self.neurons = None


class Input(BasicLayer):

    INSTANCES_COUNTER = 0

    def __init__(self, input_size: int, name: str=None):
        super().__init__()

        self.neurons = input_size

        if name is None:
            self.name = "input_{}".format(Input.INSTANCES_COUNTER)
            Input.INSTANCES_COUNTER += 1
        else:
            self.name = name

    def feed_input(self, inputs: np.array):

        if inputs.shape[0] != self.neurons:
            raise ValueError("The inputs fed to {} should be of shape [{}, m]."
                             "Found [{}, m] instead.".format(self.name, self.neurons, inputs.shape[0]))

        self.output = input


class FullyConnected(BasicLayer):

    INSTANCES_COUNTER = 0

    def __init__(self, neurons: int, input_layer: BasicLayer, activation: str,
                 name: str=None):
        super().__init__()

        self.neurons = neurons
        self.input = input_layer
        self.activation = activation

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
        self.W = np.random.rand(self.neurons, self.input.neurons)
        self.B = np.zeros([self.neurons, 1])

    def forward_pass(self, print_parameters: bool=False):
        self.Z = np.dot(self.W, self.input.output) + self.B

        if self.activation == "sigmoid":
            self.output = sigmoid(self.Z)
        elif self.activation == "relu":
            self.output = relu(self.Z)

        if print_parameters:
            print("Layer {} - Z = {}".format(self.name, self.Z))
            print("Layer {} - A = {}".format(self.name, self.output))


data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).transpose()
print("Data: {}".format(data))

inputs_layer = Input(2)
fully_layer = FullyConnected(4, inputs_layer, "sigmoid")
output_layer = FullyConnected(1, fully_layer, "sigmoid")

print(fully_layer.W)
print(fully_layer.B)

inputs_layer.feed_input(data)
fully_layer.forward_pass(print_parameters=True)
output_layer.forward_pass(print_parameters=True)
