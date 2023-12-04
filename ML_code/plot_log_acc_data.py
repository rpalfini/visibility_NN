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
plt.scatter(num_obstacles, np.log10(training_accuracy), label='Training Accuracy')
plt.scatter(num_obstacles, np.log10(validation_accuracy), label='Validation Accuracy')
plt.scatter(num_obstacles, np.log10(test_accuracy), label='Test Accuracy')

# # Plot the curve
# x_curve = np.linspace(min(num_obstacles), max(num_obstacles), 100)
# y_curve = np.log10(your_curve_function(x_curve, *popt))
# plt.plot(x_curve, y_curve, label='Fitted Curve', color='black')

# Set y-axis limits to log-scaled range
plt.ylim(np.log10(0.1), np.log10(100))

# Set y-axis to logarithmic scale with base 10
# plt.yscale('log', base=10)

# Add labels and legend
plt.xlabel('Number of Obstacles')
plt.ylabel('Accuracy (%)')
plt.legend()

# Show the plot
plt.show()
