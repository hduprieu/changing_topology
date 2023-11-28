import numpy as np

A = np.ones((3,3))
b = np.array([[1],[4],[5]])



print(A.dot(A))
print(b.dot(b.T))
print(b.T.dot(b))