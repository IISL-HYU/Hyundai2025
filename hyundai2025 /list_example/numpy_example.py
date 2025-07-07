import numpy as np

arr = np.array([1, 2, 3, 4])

print(arr * 2)                           # [2 4 6 8]
print(arr + np.array([10, 20, 30, 40]))  # [11 22 33 44]

arr_2d = np.array([[1, 2], [3, 4]])

print(arr_2d.shape)            # (2, 2)
print(arr_2d * 2)              # 요소별 곱
print(arr_2d @ arr_2d)         # 행렬 곱 (2x2 @ 2x2)