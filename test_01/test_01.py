import numpy as np

x = np.array([[1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 5], [1, 4, 6]])
y = np.array([2, 1.5, 4, 2, 5])

x_hat = np.matmul(x.T, x)
beta = np.matmul(np.linalg.inv(x_hat), x.T)
beta = np.matmul(beta, y)

print(beta)
