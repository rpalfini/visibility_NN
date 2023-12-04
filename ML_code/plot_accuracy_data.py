import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data from the LaTeX table
num_obstacles = [1, 2, 3, 4, 5, 10, 20]
training_accuracy = [99.64, 99.25, 99.21, 98.46, 97.46, 96.22, 97.03]
validation_accuracy = [99.57, 99.13, 99.08, 98.36, 97.39, 96.14, 96.77]
test_accuracy = [99.57, 99.14, 99.09, 98.36, 97.42, 96.14, 96.79]

# Fit a curve (you can replace 'your_curve_function' with an appropriate function)
def your_curve_function(x, a, b):
    return a * np.exp(b * x)

popt, pcov = curve_fit(your_curve_function, num_obstacles, training_accuracy)

# Plot the data points
plt.plot(num_obstacles, training_accuracy, label='Training Accuracy')
plt.plot(num_obstacles, validation_accuracy, label='Validation Accuracy')
plt.plot(num_obstacles, test_accuracy, label='Test Accuracy')

# Plot the curve
# x_curve = np.linspace(min(num_obstacles), max(num_obstacles), 100)
# y_curve = your_curve_function(x_curve, *popt)
# plt.plot(x_curve, y_curve, label='Fitted Curve', color='black')

plt.ylim(0, 100)

# Add labels and legend
plt.xlabel('Number of Obstacles')
plt.ylabel('Accuracy (%)')
plt.legend()

# Show the plot
plt.show()
