import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent))

from mini_nn_framework.layers import Input, FullyConnected, OutputBinary


np.random.seed(0)

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).transpose()
labels = np.array([[0, 0, 1, 1]])
print("Data:\n {}".format(data))
print("Labels:\n {}".format(labels))

print("\n------------------------------\n")
print("## Layers creation ##\n")

inputs_layer = Input(2)
# fully_layer = FullyConnected(neurons=4, input_layer=inputs_layer, activation="sigmoid")
output_layer = OutputBinary(input_layer=inputs_layer)
output_layer.print_parameters()

print("\n------------------------------\n")
print("## Training ##\n")

epochs = 1000
learning_rate = 0.1

costs = []
for i in range(epochs):

    inputs_layer.feed_input(data)
    # fully_layer.forward_pass()
    output_layer.forward_pass()

    output_layer.loss(targets=labels)
    costs.append(output_layer.cost())

    output_layer.backward_pass(targets=labels)
    # fully_layer.backward_pass()
    output_layer.parameters_update(learning_rate=learning_rate)
    # fully_layer.parameters_update(learning_rate=learning_rate)

print("\n------------------------------\n")
print("## Final model ##\n")

# fully_layer.print_parameters()
output_layer.print_parameters()

print("\n------------------------------\n")
inputs_layer.feed_input(data)
# fully_layer.forward_pass()
output_layer.forward_pass()
print("Predictions:\n{}".format(output_layer.output))
print("Targets:\n{}".format(labels))

plt.plot(range(epochs), costs)
plt.show()
