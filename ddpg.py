import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
from replaybuffer import ReplayBuffer  # 사용자 정의 리플레이 버퍼 모듈 임포트


# Actor 네트워크 정의: 상태를 받아서 행동을 출력하는 함수 근사기
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, action_bound):
        super(Actor, self).__init__()
        self.action_bound = action_bound  # 행동의 최대 절댓값 (행동 공간 범위)

        # 2개의 은닉층 정의
        self.fully_connected_layer1 = nn.Linear(state_dim, hidden_dim)
        self.fully_connected_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_layer = nn.Linear(hidden_dim, action_dim)  # 행동 차원으로 출력

    def forward(self, state):
        # 은닉층1: ReLU 활성화 함수 적용
        hidden_value1 = torch.relu(self.fully_connected_layer1(state))
        # 은닉층2: ReLU 활성화 함수 적용
        hidden_value2 = torch.relu(self.fully_connected_layer2(hidden_value1))
        # 출력층: tanh로 출력 범위를 [-1, 1]로 제한하고, action_bound로 스케일링
        action = torch.tanh(self.action_layer(hidden_value2)) * self.action_bound

        return action


# Critic 네트워크 정의: 상태와 행동을 받아서 Q값(가치함수)를 출력하는 함수 근사기
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        # 상태와 행동을 연결(concatenate)한 입력 크기
        self.fully_connected_layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fully_connected_layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_layer = nn.Linear(hidden_dim, 1)  # Q값은 스칼라

    def forward(self, state_action):
        # 은닉층1: ReLU 활성화
        hidden_value1 = torch.relu(self.fully_connected_layer1(state_action))
        # 은닉층2: ReLU 활성화
        hidden_value2 = torch.relu(self.fully_connected_layer2(hidden_value1))
        # 출력층: Q값 출력
        q_value = self.q_layer(hidden_value2)

        return q_value


# DDPG 에이전트 클래스
class DDPG:
    def __init__(
        self,
        environment,
        gamma=0.99,                  # 할인율
        batch_size=256,              # 미니배치 크기
        buffer_size=50000,           # 리플레이 버퍼 크기
        hidden_dim=256,              # 은닉층 차원
        actor_learning_rate=0.001,   # Actor 네트워크 학습률
        critic_learning_rate=0.001,  # Critic 네트워크 학습률
        tau=0.005,                  # 타겟 네트워크 소프트 업데이트 계수
        noise_std=0.1,               # 행동에 더할 가우시안 노이즈 표준편차
        device="cuda:0",             # 연산 장치 (GPU/CPU)
        save_path=None,              # 모델 저장 경로 (사용 시)
    ):
        # 하이퍼파라미터 초기화
        self.gamma = gamma
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.hidden_dim = hidden_dim
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.tau = tau
        self.noise_std = noise_std
        self.device = device
        self.save_path = save_path

        # 환경 정보 저장 (상태 및 행동 차원, 행동 범위)
        self.environment = environment
        self.state_dim = environment.observation_space.shape[0]  # 상태 차원
        self.action_dim = environment.action_space.shape[0]     # 행동 차원 (연속형)
        self.action_bound = environment.action_space.high[0]    # 행동의 최대값 (assume symmetric bounds)

        # Actor 네트워크 및 타겟 네트워크 초기화
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_bound).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.hidden_dim, self.action_bound).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())  # 가중치 동기화
        self.target_actor.eval()  # 평가 모드로 설정 (그래디언트 계산X)
        
        # Critic 네트워크 및 타겟 네트워크 초기화
        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())  # 가중치 동기화
        self.target_critic.eval()  # 평가 모드로 설정

        # 최적화 함수 설정 (Adam)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_learning_rate)

        # 리플레이 버퍼 초기화
        self.buffer = ReplayBuffer(self.buffer_size)

    # 행동 선택 함수 (노이즈 추가 가능)
    def choose_action(self, state, add_noise=True):
        # 상태를 tensor로 변환 및 배치 차원 추가
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Actor 네트워크를 통해 행동 계산 (deterministic)
            action = self.actor(state).cpu().numpy()[0]
        
        if add_noise:
            # 탐험을 위해 가우시안 노이즈 추가
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            # 행동 범위를 벗어나지 않도록 클리핑
            action = np.clip(action + noise, -self.action_bound, self.action_bound)
            
        return action

    # 타겟 네트워크 소프트 업데이트 함수 (Polyak averaging)
    def update_target_network(self):
        with torch.no_grad():
            # Actor 네트워크 가중치 업데이트
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            # Critic 네트워크 가중치 업데이트
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # Critic 네트워크 학습 함수
    def critic_learn(self, states, actions, td_targets):
        self.critic_optimizer.zero_grad()  # 옵티마이저 초기화
        # 상태와 행동을 연결하여 입력으로 사용
        state_actions = torch.cat([states, actions], dim=-1)
        q_values = self.critic(state_actions)  # 현재 Q값 예측
        # 타겟 Q값과 현재 Q값 사이의 MSE 손실 계산
        loss = nn.MSELoss()(q_values, td_targets)
        loss.backward()  # 역전파
        self.critic_optimizer.step()  # 파라미터 업데이트

    # Actor 네트워크 학습 함수
    def actor_learn(self, states):
        self.actor_optimizer.zero_grad()  # 옵티마이저 초기화
        actions = self.actor(states)  # 현재 상태에서 행동 생성
        state_actions = torch.cat([states, actions], dim=-1)
        q_values = self.critic(state_actions)  # Critic이 평가한 Q값
        # Actor의 목적은 Q값을 최대화하는 것이므로, 음수 평균을 최소화
        loss = -q_values.mean()
        loss.backward()  # 역전파
        self.actor_optimizer.step()  # 파라미터 업데이트

    # TD 타겟 계산 함수
    def td_target(self, rewards, target_qs, dones):
        # done 상태가 아니면 할인된 다음 상태 가치 추가
        return rewards + (1 - dones) * self.gamma * target_qs

    # 학습 함수 (주어진 에피소드 수만큼 반복)
    def train(self, train_episodes, evaluation_episodes, evaluation_interval):
        total_time_step = 0  # 전체 시간 스텝 카운터
        for episode in range(int(train_episodes)):
            episode_reward = 0  # 한 에피소드 보상 초기화
            time_step = 0
            done = False
            # 환경 시드 설정 (재현성 향상)
            self.environment.action_space.seed(random.randint(0, 2**32 - 1))
            state, _ = self.environment.reset(seed=random.randint(0, 2**32 - 1))

            while not done:
                # 현재 상태에서 행동 선택 (노이즈 포함)
                action = self.choose_action(state, add_noise=True)
                # 환경에 행동 수행, 다음 상태 및 보상 관측
                next_state, reward, done, truncated, _ = self.environment.step(action)
                
                # 경험 저장 (상태, 행동, 보상, 다음 상태, 종료 여부)
                self.buffer.add_buffer(state, action, reward, next_state, done)
                done = done or truncated  # truncated도 종료 조건에 포함
                
                state = next_state  # 상태 갱신
                episode_reward += reward  # 누적 보상 업데이트
                time_step += 1
                total_time_step += 1

                # 리플레이 버퍼에서 무작위 미니배치 샘플링
                batch = self.buffer.sample_batch(self.batch_size)
                states = torch.tensor(batch[0], dtype=torch.float32).to(self.device)
                actions = torch.tensor(batch[1], dtype=torch.float32).to(self.device)
                rewards = torch.tensor(batch[2], dtype=torch.float32).to(self.device)
                next_states = torch.tensor(batch[3], dtype=torch.float32).to(self.device)
                dones = torch.tensor(batch[4], dtype=torch.float32).to(self.device)

                with torch.no_grad():
                    # 타겟 Actor 네트워크를 사용해 다음 상태에 대한 행동 생성
                    next_actions = self.target_actor(next_states)
                    # 다음 상태-행동 쌍 생성
                    next_state_actions = torch.cat([next_states, next_actions], dim=-1)
                    # 타겟 Critic 네트워크를 사용해 타겟 Q값 계산
                    target_q_values = self.target_critic(next_state_actions)
                    # TD 타겟값 계산
                    td_targets = self.td_target(rewards, target_q_values, dones)

                # Critic과 Actor 학습 수행
                self.critic_learn(states, actions, td_targets)
                self.actor_learn(states)
                # 타겟 네트워크 소프트 업데이트
                self.update_target_network()

            # 한 에피소드 결과 출력
            print(f'Episode: {episode + 1}, Time Step: {time_step}, Total Time Step: {total_time_step}, Reward: {episode_reward}')

            # 일정 주기마다 평가 수행
            if (episode + 1) % evaluation_interval == 0:
                self.evaluation(evaluation_episodes)

    # 평가 함수: 탐험 노이즈 없이 행동 선택 후 성능 측정
    def evaluation(self, num_episodes):
        self.actor.eval()  # 평가 모드로 변경 (드롭아웃, 배치정규화 등 비활성화)

        episode_reward_list = []
        for episode in range(num_episodes):
            self.environment.action_space.seed(random.randint(0, 2**32 - 1))
            state, _ = self.environment.reset(seed=random.randint(0, 2**32 - 1))
            
            done = False
            episode_reward = 0
            time_step = 0
            
            while not done:
                # 탐험 노이즈 없이 행동 선택 (deterministic)
                action = self.choose_action(state, add_noise=False)
                next_state, reward, done, truncated, _ = self.environment.step(action)
                
                done = done or truncated
                state = next_state
                time_step += 1
                episode_reward += reward

            episode_reward_list.append(episode_reward)
            print(f"Evaluation Episode: {episode+1}, Time Step: {time_step}, Reward: {episode_reward}")

        self.actor.train()  # 학습 모드로 복귀
        return np.array(episode_reward_list)

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
            "actor": self.actor.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
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
        required_keys = ["actor", "target_actor", "critic", "target_critic", "actor_optimizer", "critic_optimizer"]
        for key in required_keys:
            if key not in checkpoint:
                print(f"❌ '{key}'가 체크포인트에 없습니다.")
                return
        
        self.actor.load_state_dict(checkpoint["actor"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        print(f"✅ 모델이 {path}에서 성공적으로 불러와졌습니다.")