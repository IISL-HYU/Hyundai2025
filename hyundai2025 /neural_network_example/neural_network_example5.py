import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 랜덤 시드 고정으로 결과 재현 가능하게 설정
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

# 입력 차원과 은닉층 뉴런 수 설정
input_dim = 1
hidden_dim = 10

# 모델 인스턴스 생성
model = SimpleNeuralNetwork(input_dim=input_dim, hidden_dim=hidden_dim)
# SGD 옵티마이저 생성, 학습률 0.001 설정
optimizer = optim.SGD(model.parameters(), lr=0.001)

# 데이터 생성: y = 2x + 1 + 노이즈
# X는 0~10 사이 임의의 값 (100 samples, 1 feature)
X = torch.rand(100, 1) * 10
# y는 선형 함수 값에 정규분포 노이즈 추가
y = 2 * X + 1 + torch.randn(100, 1)

# 학습 반복 횟수 설정
epochs = 100

# 학습 루프 시작
for epoch in range(epochs):
    model.train()  # 모델을 학습 모드로 전환 (드롭아웃, 배치 정규화 등 활성화)

    # 순전파: 입력 X를 모델에 넣어 예측값 계산
    outputs = model(X)
    # 손실 함수 계산 (평균 제곱 오차)
    loss = nn.MSELoss()(outputs, y)

    # 역전파 및 파라미터 업데이트 준비
    optimizer.zero_grad()  # 기존 그래디언트 초기화
    loss.backward()        # 손실 기준으로 그래디언트 계산
    optimizer.step()       # 옵티마이저가 파라미터 업데이트 수행

    # 20 에폭마다 현재 손실 출력
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 학습 완료 후 평가 모드로 전환 (드롭아웃, 배치 정규화 비활성화)
model.eval()

# 평가 시 그래디언트 계산 비활성화
with torch.no_grad():
    # X를 오름차순 정렬 (그래프 시각화를 위함)
    sorted_X, indices = torch.sort(X, dim=0)
    # 정렬된 X에 대해 모델 예측값 계산
    sorted_predicted = model(sorted_X)

# 원본 데이터 산점도로 시각화
plt.scatter(X.numpy(), y.numpy(), label='Original Data')
# 모델 예측 결과를 빨간 선으로 시각화
plt.plot(sorted_X.numpy(), sorted_predicted.numpy(), color='red', label='Fitted Line')

# 범례 추가
plt.legend()
# x축과 y축 라벨 설정
plt.xlabel("X")
plt.ylabel("y")
# 그래프 제목 설정
plt.title("Simple Neural Network Regression")
# 그래프를 이미지 파일로 저장
plt.savefig("neural_network_example5.png")
