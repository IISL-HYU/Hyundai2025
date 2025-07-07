import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 재현 가능한 결과를 위해 랜덤 시드 고정
torch.manual_seed(1)

# 간단한 신경망 모델 클래스 정의 (nn.Module 상속)
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        # 입력층(1차원)에서 은닉층(10차원)으로 연결하는 선형 계층 정의
        self.fully_connected_layer = nn.Linear(1, 10)
        # 은닉층(10차원)에서 출력층(1차원)으로 연결하는 선형 계층 정의
        self.output_layer = nn.Linear(10, 1)

    # 순전파 함수 정의 (forward 메서드 재정의)
    def forward(self, input_value):
        # 입력값을 첫 번째 선형 계층에 통과시키고 ReLU 활성화 함수 적용
        hidden_value = torch.relu(self.fully_connected_layer(input_value))
        # 은닉층 출력을 출력층에 통과시켜 최종 출력 반환
        return self.output_layer(hidden_value)

# 모델 인스턴스 생성
model = SimpleNeuralNetwork()

# 임의의 입력 데이터 생성
# torch.rand(3, 1): 0과 1 사이의 균등분포에서 3행 1열 크기의 텐서 생성
X = torch.rand(3, 1)  # 입력 데이터: 3개의 샘플, 각 샘플은 1차원 값

print("X : ", X)            # 입력 데이터 출력
print("X Shape : ", X.shape)  # 입력 데이터 형태 출력 (torch.Size([3, 1]))

# 모델에 입력 데이터를 넣어 순전파 실행 (예측값 계산)
output = model(X)

print("Output : ", output)           # 모델 출력 결과 출력 (3x1 텐서)
print("Output Shape : ", output.shape)  # 출력 데이터 형태 출력 (torch.Size([3, 1]))
