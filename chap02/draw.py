import numpy as np
import matplotlib.pyplot as plt

points = np.genfromtxt("data.csv", delimiter=",")
plt.scatter(points[:, 0], points[:, 1])
x = np.arange(0, 100)
y = 1.4777 * x + 0.088
plt.xlabel('x')
plt.ylabel('y')
plt.title("y = wx + b")
plt.plot(x, y, color='r', linewidth=2.5)
plt.show()