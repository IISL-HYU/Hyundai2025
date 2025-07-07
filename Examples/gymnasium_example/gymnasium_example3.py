import gymnasium as gym  # OpenAI Gym의 후속 라이브러리인 Gymnasium을 import

# "Acrobot-v1" 환경 생성
# 이 환경은 2개의 막대(링크)를 조작해서 끝단을 위로 들어올리는 강화학습 문제입니다
environment = gym.make("Acrobot-v1")

# 전체 에피소드 수 설정
num_episodes = 10

# 각 에피소드의 누적 보상을 저장할 리스트
episode_reward_list = []

# 여러 에피소드를 반복
for episode in range(num_episodes):
    # 환경 초기화: 상태(state)와 추가 정보(info) 반환
    state, _ = environment.reset()

    # 에피소드 진행 상태를 나타내는 플래그
    episode_over = False

    # 시간 스텝 수 초기화 (각 에피소드 내 스텝 수 카운팅)
    time_step = 0

    # 누적 보상 초기화
    episode_reward = 0

    # 에피소드가 종료될 때까지 반복
    while not episode_over:
        # 무작위 행동 선택 (0: -1 torque, 1: 0 torque, 2: +1 torque)
        action = environment.action_space.sample()

        # 선택한 행동을 환경에 적용하고, 결과를 받음
        next_state, reward, done, truncated, _ = environment.step(action)

        # 에피소드 종료 여부 확인 (정상 종료 or 타임리밋 등)
        episode_over = done or truncated

        # 시간 스텝 및 누적 보상 업데이트
        time_step += 1
        episode_reward += reward

    # 에피소드 종료 후 누적 보상을 저장
    episode_reward_list.append(episode_reward)

    # 현재 에피소드 결과 출력
    print(f'Episode: {episode + 1}, Time Step: {time_step}, Reward: {episode_reward}')

# 전체 에피소드 보상 목록 출력
print("Episode Reward List : ", episode_reward_list)
