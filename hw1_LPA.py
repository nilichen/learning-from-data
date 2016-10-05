%matplotlib inline
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def sign(x):
    if abs(x) < 0.000000001:
        return 0

    return x > 0 and 1 or -1
signv = np.vectorize(sign)

def generate_line():
    x1, x2, y1, y2 = np.random.uniform(-1, 1, 4)
    slope = (y2 - y1) / (x2 - x1)
    intercept = y2 - slope * x1
    return np.array([-intercept, -slope, 1])

def generate_points(n):
    return np.array([(1, x, y) for x, y in zip(np.random.uniform(-1, 1, n), np.random.uniform(-1, 1, n))])

def cal_signs(points, line):
    return signv(points.dot(line))

def train(n_points, max_iterations):
    line = generate_line()
    points = generate_points(n_points)
    signs = cal_signs(points, line)
#     print signs
    train_line = np.array([0.0, 0.0, 0.0])
    n_iterations = 0
    while n_iterations <= max_iterations:
#         plt.scatter(points[:, 1], points[:, 2])
#         x = np.array([-1, 1])
#         train_y = -train_line[1]/train_line[2] * x - train_line[0] / train_line[2]
#         plt.plot(x, train_y, 'k-')
#         y = -x * line[1] - line[0]
#         plt.plot(x, y, 'r-')
#         plt.show()

        train_signs = cal_signs(points, train_line)
#         print train_signs
#         print train_line
        misclassified_points = points[train_signs != signs]
#         print misclassified_points
        if len(misclassified_points) == 0:
            break
        else:
            selected_point = misclassified_points[np.random.randint(low=0, high=len(misclassified_points), size=1)][0]
#             print selected_point

            sign = cal_signs(selected_point, line)
#             print selected_point.dot(train_line)
            train_line += selected_point * sign

            n_iterations += 1


    # print train_signs != signs
    error = sum(train_signs != signs) / len(points)
    return error, n_iterations

# train(10, 15)
for iteration in [15, 300, 5000, 10000]:
    total_error, total_iterations = 0.0, 0.0
    for i in range(500):
        error, iterations = train(10, iteration)
        total_error += error
#         print error, total_error
        total_iterations += iterations

    print total_error / 500, total_iterations / 500


for iteration in [50, 100, 500, 1000, 5000]:
    total_error, total_iterations = 0.0, 0.0
    for i in range(500):
        error, iterations = train(100, iteration)
        total_error += error
#         print error, total_error
        total_iterations += iterations

    print total_error / 500, total_iterations / 500