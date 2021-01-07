import numpy as np

a = np.array([[2, -1, -1], [2, -1, -2], [-1, 1, 2]])
p = np.array([[1, 1, 1], [1, 2, 0], [0, -1, 0]])  # 自己答案
j = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])
p_new = np.array([[1, -1, 1], [1, -2, 0], [0, 1, 0]])  # 答案

# a = np.array([[2, -1, -1], [2, -1, -2], [-1, 1, 2]])
# p = np.array([[1, 1, 1], [1, 2, 0], [0, -1, 0]])  # 自己答案
# j = np.array([[1, 0, 0], [0, 1, 1], [0, 0, 1]])

print(np.dot(a, p))
print(np.dot(p, j))
print("--------------------")
print(np.dot(a, p_new))
print(np.dot(p_new, j))

