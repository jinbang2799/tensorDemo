import numpy as np
import matplotlib.pyplot as plt

func = np.poly1d(np.array([1, 2, 3, 4]).astype(float))
func2 = func.deriv(m=2)
x = np.linspace(-10, 10, 30)
y = func(x)
y2 = func2(x)

plt.plot(x, y)
plt.plot(x, y2, 'r>')
plt.xlabel('x')
plt.ylabel('y(x)')
plt.show()