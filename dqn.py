import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from replaybuffer import ReplayBuffer  # 사용자 정의 리플레이 버퍼 임포트


# Q-네트워크 클래스 정의 (딥러닝 모델)
class QNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=128):
        super(QNetwork, self).__init__()

        # 첫 번째 은닉층: 상태 차원 -> 은닉층 크기
        self.fully_connected_layer1 = nn.Linear(state_dim, hidden_dim)
        # 두 번째 은닉층: 은닉층 크기 -> 은닉층 크기
        self.fully_connected_layer2 = nn.Linear(hidden_dim, hidden_dim)
        # 출력층: 은닉층 크기 -> 가능한 행동 개수(Q값 출력)
        self.q_layer = nn.Linear(hidden_dim, num_actions)

    def forward(self, input_value):
        # 입력: 상태 (batch_size, state_dim)
        hidden_value1 = torch.relu(self.fully_connected_layer1(input_value))  # ReLU 활성화 적용
        hidden_value2 = torch.relu(self.fully_connected_layer2(hidden_value1))  # 두 번째 은닉층 ReLU
        q_value = self.q_layer(hidden_value2)  # 최종 Q값 (batch_size, num_actions)

        return q_value  # 각 행동에 대한 Q값 반환


class DQN:
    def __init__(
        self, 
        environment,
        gamma=0.99, 
        batch_size=64, 
        buffer_size=20000, 
        hidden_dim=128, 
        update_after=1000,
        learning_rate=0.001, 
        tau=0.001,
        epsilon_start=1.0, 
        epsilon_decay=0.99, 
        epsilon_min=0.1, 
        device="cuda:0", 
        save_path=None,
    ): 
        # DQN 주요 하이퍼파라미터 초기화
        self.gamma = gamma  # 할인율
        self.batch_size = batch_size  # 미니배치 크기
        self.buffer_size = buffer_size  # 리플레이 버퍼 최대 크기
        self.update_after = update_after  # 학습 시작 전 최소 경험 수 (사용하지는 않음)
        self.learning_rate = learning_rate  # 학습률
        self.tau = tau  # 타겟 네트워크 소프트 업데이트 계수
        self.epsilon = epsilon_start  # 탐험 확률 초기값 (ε-greedy)
        self.epsilon_decay = epsilon_decay  # 탐험 확률 감소율
        self.epsilon_min = epsilon_min  # 탐험 확률 최솟값
        self.device = device  # 연산 디바이스 (GPU/CPU)
        self.save_path = save_path  # 모델 저장 경로

        # 환경 정보 저장
        self.environment = environment
        self.state_shape = environment.observation_space.shape[0]  # 상태 공간 크기 (연속형 상태)
        self.num_actions = environment.action_space.n  # 행동 공간 크기 (이산형 행동)

        # Q-네트워크와 타겟 네트워크 초기화 및 복사
        self.q_network = QNetwork(self.state_shape, self.num_actions, hidden_dim).to(self.device)
        self.target_q_network = QNetwork(self.state_shape, self.num_actions, hidden_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # 파라미터 복사
        self.target_q_network.eval()  # 타겟 네트워크는 평가 모드로 설정 (학습 안 함)

        # 옵티마이저 초기화 (Adam)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        # 경험 저장용 리플레이 버퍼 초기화
        self.buffer = ReplayBuffer(self.buffer_size)

    def choose_action(self, state, deterministic=False):
        # 상태를 텐서로 변환 (batch 차원 추가 후 device 이동)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_value = self.q_network(state)  # Q값 예측 (1, num_actions)

        # ε-greedy 정책: 탐험 또는 최대 Q값 행동 선택
        if deterministic or np.random.random() > self.epsilon:
            action = torch.argmax(q_value).item()  # 최대 Q값 행동 선택 (활용)
        else:
            action = self.environment.action_space.sample()  # 무작위 행동 (탐험)

        return action

    def update_target_network(self):
        # 타겟 네트워크 파라미터를 Q-네트워크 파라미터로 부드럽게 업데이트 (Polyak averaging)
        with torch.no_grad():
            for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def dqn_learn(self, states, actions, td_targets):
        # 네트워크 학습 함수
        self.optimizer.zero_grad()  # 기울기 초기화
        q_values = self.q_network(states)  # 현재 상태들에 대한 Q값 예측 (batch_size, num_actions)

        # 행동에 해당하는 Q값만 선택 (batch_size, 1)
        q_values = q_values.gather(1, actions)

        # TD 오차 기반 손실 함수 (MSE)
        loss = nn.MSELoss()(q_values, td_targets)
        loss.backward()  # 역전파
        self.optimizer.step()  # 파라미터 업데이트

    def td_target(self, rewards, target_qs, dones):
        # 타겟 Q값 계산 (r + γ * max_a Q_target(s', a')) 형태
        max_q = target_qs.max(1)[0].unsqueeze(1)  # 다음 상태에서 최대 Q값 (batch_size, 1)
        return rewards + (1 - dones) * self.gamma * max_q  # 종료된 상태는 다음 값 고려 안함

    def train(self, train_episodes, evaluation_episodes, evaluation_interval):
        total_time_step = 0  # 전체 시간 스텝 누적 변수
        for episode in range(int(train_episodes)):
            episode_reward = 0  # 한 에피소드의 누적 보상
            time_step = 0
            done = False

            # 환경과 행동 공간 랜덤 시드 초기화 (재현성 확보)
            self.environment.action_space.seed(random.randint(0, 2**32 - 1))
            state, _ = self.environment.reset(seed=random.randint(0, 2**32 - 1))

            while not done:
                # 행동 선택 (탐험+활용 혼합)
                action = self.choose_action(state=state, deterministic=False)
                # 행동 실행, 다음 상태 및 보상 수신
                next_state, reward, done, truncated, _ = self.environment.step(action)

                # 경험 저장 (done은 종료 신호, truncated은 시간 제한 등 기타 종료)
                self.buffer.add_buffer(state, action, reward, next_state, done)
                done = done or truncated  # 둘 중 하나라도 True면 종료 처리

                # 상태 및 보상 업데이트
                state = next_state
                episode_reward += reward
                time_step += 1
                total_time_step += 1

                # 리플레이 버퍼에서 미니배치 샘플링
                batch = self.buffer.sample_batch(self.batch_size)
                states = torch.tensor(batch[0], dtype=torch.float32).to(self.device)        # 상태 배치
                actions = torch.tensor(batch[1], dtype=torch.long).to(self.device)          # 행동 배치
                rewards = torch.tensor(batch[2], dtype=torch.float32).to(self.device)       # 보상 배치
                next_states = torch.tensor(batch[3], dtype=torch.float32).to(self.device)   # 다음 상태 배치
                dones = torch.tensor(batch[4], dtype=torch.float32).to(self.device)         # 종료 여부 배치

                # 타겟 Q값 계산
                with torch.no_grad():
                    target_q_values = self.target_q_network(next_states)  # 타겟 네트워크로 다음 상태 Q값 추정
                    td_targets = self.td_target(rewards, target_q_values, dones)  # TD 타겟 계산

                # 네트워크 업데이트 (손실 최소화)
                self.dqn_learn(states=states, actions=actions, td_targets=td_targets)
                self.update_target_network()  # 타겟 네트워크 소프트 업데이트

            # 에피소드 종료 시 출력
            print(f'Episode: {episode + 1}, Time Step: {time_step}, Total Time Step: {total_time_step}, Reward: {episode_reward}, Epsilon: {self.epsilon}')

            # ε 감소 (탐험에서 활용 비중 증가)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
                self.epsilon = max(self.epsilon, self.epsilon_min)

            # 일정 주기로 평가 수행
            if (episode + 1) % evaluation_interval == 0:
                self.evaluation(evaluation_episodes)

    def evaluation(self, num_episodes):
        self.q_network.eval()  # 평가 모드 활성화 (드롭아웃, 배치정규화 비활성화)

        episode_reward_list = []
        for episode in range(num_episodes):
            # 환경 시드 초기화 (재현성 확보)
            self.environment.action_space.seed(random.randint(0, 2**32 - 1))
            state, _ = self.environment.reset(seed=random.randint(0, 2**32 - 1))
            
            done = False
            episode_reward = 0
            time_step = 0
            while not done:
                # 탐험 없이 최대 Q값 행동 선택 (결과 평가)
                action = self.choose_action(state=state, deterministic=True)
                next_state, reward, done, truncated, _ = self.environment.step(action)

                done = done or truncated  # 종료 조건 병합
                state = next_state
                time_step += 1
                episode_reward += reward

            episode_reward_list.append(episode_reward)
            print(f"Evaluation Episode: {episode+1}, Time Step: {time_step}, Reward: {episode_reward}")

        self.q_network.train()  # 학습 모드 복귀
        return np.array(episode_reward_list)  # 평가 결과 반환

    def save_model(self, path=None):
        """모델과 옵티마이저 상태를 저장"""
        if path is None:
            path = self.save_path
        if path is None:
            print("❌ 저장할 경로가 설정되지 않았습니다.")
            return
        
        # path가 디렉터리인지 확인하고, 파일 경로를 명확히 지정
        checkpoint_path = os.path.join(path, "checkpoint.pth")

        # 디렉터리 생성 (파일 경로가 주어졌을 경우 dirname을 사용)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # 저장할 모델 상태 정의
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_q_network": self.target_q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon
        }

        # 모델 저장 (예외 처리 포함)
        try:
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ 모델이 {checkpoint_path}에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 모델 저장에 실패했습니다: {e}")

    def load_model(self, path=None):
        """저장된 모델을 불러와 네트워크 및 옵티마이저 상태 복원"""
        if path is None:
            path = self.save_path
        if path is None:
            print("❌ 저장된 모델 파일을 찾을 수 없습니다.")
            return
        path = os.path.join(path, "checkpoint.pth")
        
        if not os.path.exists(path):
            print("❌ 저장된 모델 파일을 찾을 수 없습니다.")
            return

        try:
            checkpoint = torch.load(path, map_location=self.device)
        except Exception as e:
            print(f"❌ 모델 파일을 로드하는 데 실패했습니다: {e}")
            return

        # 각 키가 checkpoint에 존재하는지 확인
        required_keys = ["q_network", "target_q_network", "optimizer", "epsilon"]
        for key in required_keys:
            if key not in checkpoint:
                print(f"❌ '{key}'가 체크포인트에 없습니다.")
                return
        
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(checkpoint["target_q_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]

        print(f"✅ 모델이 {path}에서 성공적으로 불러와졌습니다.")