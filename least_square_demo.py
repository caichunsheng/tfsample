import numpy as np

# Sample data points: (1, 2), (2, 5), (3, 7)
x_data = np.array([1, 2, 3])
y_data = np.array([2, 5, 7])

# Formulate matrix A: Column of x and a column of ones for intercept
A = np.vstack([x_data, np.ones(len(x_data))]).T

# Solve Ax = b
# lstsq returns: coefficients, residuals, rank, singular values
coefficients, residuals, _, _ = np.linalg.lstsq(A, y_data, rcond=None)

m, c = coefficients
print(f"Slope (m): {m:.2f}, Intercept (c): {c:.2f}")
