import gymnasium as gym  # OpenAI Gym의 후속 패키지인 Gymnasium을 import

# "Acrobot-v1" 환경 생성
# 이 환경은 2개의 연결된 막대를 제어해 끝단을 목표 높이까지 올리는 강화학습 문제
environment = gym.make("Acrobot-v1")

# 환경을 초기화하고 초기 상태(observation)를 받음
# reset()은 (state, info) 튜플을 반환함
state, _ = environment.reset()

# 에피소드가 끝났는지를 나타내는 변수
episode_over = False

# 시간 스텝 카운터 (에피소드 내 스텝 수)
time_step = 0

# 누적 보상을 저장할 변수 (Acrobot에서는 스텝마다 보상이 -1)
episode_reward = 0

# 에피소드가 끝날 때까지 반복
while not episode_over:
    # 가능한 행동 중 하나를 무작위로 선택
    action = environment.action_space.sample()  # 0, 1, 2 중 하나 (왼쪽 토크, 정지, 오른쪽 토크)

    # 선택한 행동을 환경에 적용하여 다음 상태와 결과들을 얻음
    next_state, reward, done, truncated, _ = environment.step(action)

    # 에피소드 종료 조건 확인
    # - done: 목표 상태 도달 등으로 에피소드가 종료됨
    # - truncated: 타임리밋 등 외부 조건으로 에피소드가 종료됨
    episode_over = done or truncated

    # 시간 스텝 증가
    time_step += 1

    # 누적 보상 업데이트
    episode_reward += reward

# 에피소드 종료 후 결과 출력
print("Episode Time : ", time_step)         # 총 몇 번의 스텝이 있었는지 출력
print("Episode Reward : ", episode_reward)  # 총 누적 보상 출력 (보통 -스텝 수와 동일)
