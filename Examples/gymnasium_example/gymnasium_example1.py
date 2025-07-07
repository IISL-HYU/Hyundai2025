import gymnasium as gym  # OpenAI Gym의 후속 라이브러리인 gymnasium을 import

# Acrobot-v1 환경 생성
# Acrobot은 2개의 연결된 막대가 중력에 대해 수직으로 서도록 하는 강화학습 문제
environment = gym.make("Acrobot-v1")

# 환경을 초기화하고, 초기 상태를 반환받음
# Gymnasium은 초기화 시 (observation, info) 튜플을 반환함
state, _ = environment.reset()
print("Initial State : ", state)  # 환경 초기화 후의 상태 출력

# 가능한 행동 공간에서 임의의 행동을 하나 샘플링
# Acrobot의 행동 공간은 이산적인 정수 (0, 1, 2)로 구성됨
action = environment.action_space.sample()
print("Current Action : ", action)  # 선택된 임의의 행동 출력

# 한 스텝 환경을 진행
# 반환 값은 다음 상태, 보상, 종료 여부, 제한 조건 종료 여부, 추가 정보
next_state, reward, done, truncated, _ = environment.step(action)

# 다음 상태 출력 (6차원 연속 값: 각도와 속도 등)
print("Next State : ", next_state)

# Acrobot는 목표 상태에 도달했을 때 -1 보상만 제공함 (모든 스텝에서 -1)
print("Reward : ", reward)

# 에피소드가 정상적으로 종료되었는지 여부 (목표 도달로 인한 종료 등)
print("Done : ", done)

# 환경이 시간 제한 등으로 강제 종료되었는지 여부 (truncated == True일 경우 종료)
print("Truncated : ", truncated)
