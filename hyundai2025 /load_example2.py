# 강화학습 모델 DQN과 DDPG 클래스 import
from dqn import DQN
from ddpg import DDPG

# Gymnasium 환경 불러오기 (Acrobot 등 강화학습용 환경 제공)
import gymnasium as gym

# 렌더링된 프레임을 GIF로 저장하는 함수
from load_example1 import save_frames_as_gif


def main():
    # 사용할 환경 이름 지정
    environment_name = 'Acrobot-v1'

    # 저장된 모델의 경로 (모델 파라미터가 저장되어 있는 디렉터리)
    save_path = f"hyundai2025_logs/dqn/{environment_name}"

    # Gymnasium 환경 생성, 렌더링 모드는 'rgb_array'로 설정하여 이미지 프레임을 얻을 수 있게 함
    environment = gym.make(environment_name, render_mode="rgb_array")

    # DQN 에이전트 생성, 환경 정보 전달
    dqn = DQN(environment=environment)

    # 저장된 모델 파라미터 불러오기
    dqn.load_model(path=save_path)

    # 환경 초기화 및 초기 상태(observation) 획득
    # reset()은 (state, info) 형태로 반환함
    state, _ = environment.reset()

    # 에피소드 종료 여부 플래그
    episode_over = False

    # 시간 스텝 수 카운터 (몇 번 행동했는지)
    time_step = 0

    # 누적 보상을 저장할 변수
    episode_reward = 0

    # 환경을 렌더링한 프레임을 저장할 리스트 (후에 GIF로 만들기 위함)
    frames = []

    # 에피소드가 끝날 때까지 루프 반복
    while not episode_over:
        # 현재 환경 상태의 렌더링 이미지 저장
        frames.append(environment.render())

        # 훈련된 DQN 모델을 통해 행동 선택
        # deterministic=True → ε-greedy가 아닌 결정론적 선택 (가장 Q값이 높은 행동 선택)
        action = dqn.choose_action(state, deterministic=True)

        # 선택한 행동을 환경에 적용하여 다음 상태, 보상, done 여부 등을 얻음
        next_state, reward, done, truncated, _ = environment.step(action)

        # 환경이 종료되었는지 판단 (done: 성공/실패 조건, truncated: 시간 초과 등)
        episode_over = done or truncated

        # 시간 스텝 수 증가
        time_step += 1

        # 누적 보상 업데이트
        episode_reward += reward

        # 상태 갱신
        state = next_state

    # 에피소드 종료 후 결과 출력
    print("Episode Time : ", time_step)         # 총 몇 번의 스텝이 있었는지 출력
    print("Episode Reward : ", episode_reward)  # 누적 보상 출력 (보통 -스텝 수와 비슷)

    # 저장한 프레임을 기반으로 GIF 생성 및 저장
    save_frames_as_gif(frames, path=f"trained_{environment_name}.gif")


# 이 파이썬 파일이 메인으로 실행될 경우 main() 함수 호출
if __name__ == "__main__":
    main()
