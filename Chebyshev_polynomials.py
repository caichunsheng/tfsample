import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev

# Function to approximate
def f(x):
    return np.exp(x)

# Approximate exp(x) on [-1, 1] with degree 5
x = np.linspace(-1, 1, 200)
y = f(x)

cheb = Chebyshev.fit(x, y, deg=5)

y_approx = cheb(x)

plt.plot(x, y, label="true exp(x)")
plt.plot(x, y_approx, "--", label="Chebyshev approximation")
plt.legend()
plt.grid(True)
plt.show()

print(cheb)