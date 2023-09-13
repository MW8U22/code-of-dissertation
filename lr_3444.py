import numpy as np
import matplotlib.pyplot as plt

def function(x):
    return x**2

def gradient(x):
    return 2*x

def gradient_descent(starting_point, learning_rate, n_iterations):
    path = []
    x = starting_point
    for _ in range(n_iterations):
        path.append(x)
        x = x - learning_rate * gradient(x)
    return path

starting_point = 5
n_iterations = 10

# 学习率过大
lr_large = 1.1
path_large = gradient_descent(starting_point, lr_large, n_iterations)

# 学习率过小
lr_small = 0.01
path_small = gradient_descent(starting_point, lr_small, n_iterations)

x = np.linspace(-6, 6, 400)
y = function(x)

plt.figure(figsize=(12, 5))

# Plot for large learning rate
plt.subplot(1, 2, 1)
plt.plot(x, y, '-b')
plt.plot(path_large, function(np.array(path_large)), 'ro')
plt.title(f"Learning rate too large: {lr_large}")

# Plot for small learning rate
plt.subplot(1, 2, 2)
plt.plot(x, y, '-b')
plt.plot(path_small, function(np.array(path_small)), 'ro')
plt.title(f"Learning rate too small: {lr_small}")

plt.tight_layout()
plt.show()
