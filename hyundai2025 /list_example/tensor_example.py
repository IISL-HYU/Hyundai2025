import torch

tensor = torch.tensor([1, 2, 3, 4])

print(tensor * 2)  # tensor([2, 4, 6, 8])

tensor_2d = torch.tensor([[1, 2], [3, 4]])

print(tensor_2d.shape)         # torch.Size([2, 2])
print(tensor_2d * 2)           # 요소별 곱
print(torch.matmul(tensor_2d, tensor_2d))  # 행렬 곱