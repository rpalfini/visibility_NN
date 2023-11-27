import numpy as np
import matplotlib.pyplot as plt

# Define the ReLU function
def relu(x):
    return np.maximum(0, x)

# Define the Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate x values
x = np.linspace(-5, 5, 100)

# Calculate corresponding y values using ReLU and Sigmoid functions
y_relu = relu(x)
y_sigmoid = sigmoid(x)

# Plot the ReLU function
plt.figure(figsize=(8, 4))
plt.plot(x, y_relu, color='blue', label='ReLU Function',linewidth=2)
plt.title('ReLU Function')
plt.xlabel(r'$s$')
plt.ylabel(r'$R(s)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.2)
plt.text(-4.5, 4.3, r'$R(s) = \max(0, s)$', fontsize=14, color='black')
plt.savefig('relu_plot.png')  # Save the ReLU plot as a PNG file
plt.show()

# Plot the Sigmoid function
plt.figure(figsize=(8, 4))
plt.plot(x, y_sigmoid, label='Sigmoid Function', color='blue',linewidth=2)
plt.title('Sigmoid Function')
plt.xlabel(r'$s$')
plt.ylabel(r'$\sigma(s)$')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5,alpha=0.2)
plt.text(-4.5, 0.87, r'$\sigma(s) = \frac{1}{1 + e^{-s}}$', fontsize=16, color='black')
# plt.text(-4.5, 0.4, r'$\sigma(s) = \begin{Bmatrix} 1, & \text{for } x \rightarrow + \infty\\ 0, & \text{for } x \rightarrow - \infty \end{Bmatrix}$', fontsize=16, color='black')
plt.text(-4.5, 0.4, r'$\sigma(s) = \left\{ \begin{array}{ll} 1, & \text{for } x \rightarrow + \infty\\ 0, & \text{for } x \rightarrow - \infty \end{array} \right.$', fontsize=16, color='black')

plt.savefig('sigmoid_plot.png')  # Save the Sigmoid plot as a PNG file
plt.show()
