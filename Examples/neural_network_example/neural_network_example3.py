import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 결과 재현을 위해 랜덤 시드 고정
torch.manual_seed(1)

# 간단한 신경망 모델 클래스 정의 (nn.Module 상속)
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleNeuralNetwork, self).__init__()
        # 입력층에서 은닉층으로 연결하는 선형 계층 정의
        self.fully_connected_layer = nn.Linear(input_dim, hidden_dim)
        # 은닉층에서 출력층(1차원)으로 연결하는 선형 계층 정의
        self.output_layer = nn.Linear(hidden_dim, 1)

    # 순전파 함수 정의
    def forward(self, input_value):
        # 입력값을 첫 번째 선형 계층에 통과시키고 ReLU 활성화 함수 적용
        hidden_value = torch.relu(self.fully_connected_layer(input_value))
        # 은닉층 출력을 출력층에 통과시켜 최종 출력 반환
        return self.output_layer(hidden_value)

# 입력 데이터 차원과 은닉층 뉴런 수 정의
input_dim = 1
hidden_dim = 10

# 모델 인스턴스 생성
model = SimpleNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim)

# 확률적 경사 하강법(SGD) 옵티마이저 생성
# 모델 파라미터를 최적화하며 학습률은 0.01로 설정
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 랜덤 데이터 생성: y = 2x + 1 + 노이즈
# X는 0~10 사이 값, 크기는 (100, 1)
X = torch.rand(100, 1) * 10
# y는 선형 함수 결과에 정규분포 노이즈(torch.randn) 추가
y = 2 * X + 1 + torch.randn(100, 1)

# 학습 전 모델 출력 시각화를 위해 torch.no_grad()로 그래디언트 비활성화
with torch.no_grad():
    # X를 오름차순 정렬 (시각화용)
    sorted_X, indices = torch.sort(X, dim=0)
    # 정렬된 X에 대해 모델 예측값 계산
    sorted_predicted = model(sorted_X)

# 산점도로 원본 데이터 시각화 (X, y)
plt.scatter(X.numpy(), y.numpy(), label='Original Data')
# 모델 예측값을 빨간 선으로 시각화
plt.plot(sorted_X.numpy(), sorted_predicted.numpy(), color='red', label='Fitted Line')

# 범례 추가
plt.legend()
# x축, y축 라벨 지정
plt.xlabel("X")
plt.ylabel("y")
# 그래프 제목 지정
plt.title("Simple Neural Network Regression")

# 그래프를 파일로 저장 (현재 경로에 neural_network_example3.png)
plt.savefig("neural_network_example3.png")
