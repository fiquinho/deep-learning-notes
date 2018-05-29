import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt


def plot_data(x: List[float], y: List[float], line: Tuple[float, float]=None):

    plt.figure(figsize=(5, 5))
    plt.plot(x, y, 'o')

    if line is not None:
        if len(line) != 2:
            raise ValueError("The line tuple should have 2 values (weight, bias).")

        w = line[0]
        b = line[1]

        line_extra_space = 0.25
        x_line = []
        x_range = max(x) - min(x)
        line_x_min = min(x) - x_range * line_extra_space
        line_x_max = max(x) + x_range * line_extra_space
        line_range = line_x_max - line_x_min
        for i in range(len(x)):
            x_line.append(line_x_min + (line_range / len(x)) * i)

        y_line = [w * value + b for value in x_line]

        if b < 0:
            bias_text = " - {}".format(abs(round(b, 3)))
        else:
            bias_text = " + {}".format(round(b, 3))

        plt.text(x_line[0], y_line[-1],
                 "Linear function:\n"
                 "{} * x".format(round(w, 3), round(b, 3)) + bias_text,
                 fontweight='bold',
                 bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})

        plt.plot(x_line, y_line)

    plt.xlim((-2, 11))
    plt.axis('equal')
    plt.title("Data points and linear function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()

    plt.show()


def main():

    data_file = Path(Path.cwd().parent, "data", "notebook_example_data.csv")

    # Extract data from file to lists of values
    x = []
    y = []
    with open(data_file, "r") as file:
        reader = csv.DictReader(file)
        for data in reader:
            x.append(float(data["x"]))
            y.append(float(data["y"]))

    w = 0.35
    b = 4

    plot_data(x=x, y=y, line=(w, b))

if __name__ == '__main__':
    main()