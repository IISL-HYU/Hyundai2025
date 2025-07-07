import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 결과 재현을 위해 랜덤 시드 고정
torch.manual_seed(1)

# 간단한 신경망 모델 정의 (nn.Module 상속)
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleNeuralNetwork, self).__init__()
        # 입력 차원(input_dim)에서 은닉층 차원(hidden_dim)으로 연결하는 선형 계층 정의
        self.fully_connected_layer = nn.Linear(input_dim, hidden_dim)
        # 은닉층(hidden_dim)에서 출력층(1)으로 연결하는 선형 계층 정의
        self.output_layer = nn.Linear(hidden_dim, 1)

    # 순전파 함수 (forward) 정의
    def forward(self, input_value):
        # 입력값을 첫 번째 선형 계층에 통과시키고 ReLU 활성화 함수 적용
        hidden_value = torch.relu(self.fully_connected_layer(input_value))
        # 은닉층 출력을 출력층에 통과시켜 최종 출력 반환
        return self.output_layer(hidden_value)

# 입력 데이터 차원 설정 (1차원 입력)
input_dim = 1
# 은닉층 뉴런 개수 설정
hidden_dim = 10

# 모델 인스턴스 생성, 입력 및 은닉 차원 지정
model = SimpleNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim)

# 임의의 입력 데이터 생성 (3개의 샘플, 각 샘플은 input_dim 차원)
X = torch.rand(3, input_dim)  # 0과 1 사이의 균등분포에서 생성된 텐서

print("X : ", X)             # 입력 데이터 출력
print("X Shape : ", X.shape) # 입력 데이터의 텐서 크기 출력 (torch.Size([3, 1]))

# 모델에 입력 데이터를 넣어 순전파 수행하여 출력 계산
output = model(X)

print("Output : ", output)           # 모델 출력 결과 출력 (3x1 텐서)
print("Output Shape : ", output.shape)  # 출력 텐서 크기 출력 (torch.Size([3, 1]))
