# Gymnasium은 강화학습 시뮬레이션 환경을 제공하는 라이브러리로, OpenAI의 Gym을 계승한 버전입니다.
import gymnasium as gym

# Matplotlib의 animation 기능을 활용해 시각적으로 에이전트의 동작을 저장하기 위함
from matplotlib import animation
import matplotlib.pyplot as plt


def save_frames_as_gif(frames, path="./gym_animation.gif"):
    """
    렌더링된 이미지 프레임들을 받아서 GIF 파일로 저장하는 함수

    Args:
        frames (list of np.array): 각 스텝에서 환경으로부터 받은 이미지 프레임
        path (str): 저장할 GIF 파일 경로
    """

    # 프레임 크기에 맞게 그래프 사이즈 조정 (72 DPI 기준)
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    # 첫 번째 프레임을 그래프에 띄우고, 축은 보이지 않게 설정
    patch = plt.imshow(frames[0])
    plt.axis('off')

    # 프레임을 순차적으로 업데이트하는 내부 함수
    def animate(i):
        patch.set_data(frames[i])  # 현재 프레임을 patch에 설정

    # FuncAnimation 객체 생성: 지정된 interval(밀리초)마다 animate() 호출
    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50
    )

    # GIF로 저장, imagemagick 백엔드를 사용하고 FPS를 60으로 설정
    anim.save(path, writer='imagemagick', fps=60)


def main():
    """
    Acrobot-v1 환경을 실행하고, 무작위 정책을 사용하여 에이전트의 동작을 시뮬레이션한 뒤,
    그 결과를 GIF로 저장하는 메인 함수
    """
    # 사용할 환경 이름 지정
    environment_name = 'Acrobot-v1'
    
    # render_mode='rgb_array'는 렌더링 출력을 RGB 이미지 배열로 받겠다는 뜻
    environment = gym.make(environment_name, render_mode="rgb_array")

    # 환경 초기화 후 초기 상태(observation)와 추가 정보(info) 반환
    state, _ = environment.reset()

    # 에피소드 종료 여부를 나타내는 플래그 변수
    episode_over = False

    # 시간 스텝 수 (각 에피소드에서 몇 번의 행동이 있었는지)
    time_step = 0

    # 누적 보상 (Acrobot는 매 스텝마다 -1의 보상을 줌)
    episode_reward = 0

    # 렌더링된 프레임들을 저장할 리스트
    frames = []

    # 에피소드 종료 시점까지 반복 수행
    while not episode_over:
        # 현재 환경 상태를 렌더링한 이미지 프레임 저장
        frames.append(environment.render())

        # 가능한 행동 중 무작위로 하나 선택 (총 3가지 행동 존재: 0=왼쪽 토크, 1=정지, 2=오른쪽 토크)
        action = environment.action_space.sample()

        # 선택한 행동을 환경에 적용 → 다음 상태, 보상, done 여부, truncated 여부, 추가 정보 반환
        next_state, reward, done, truncated, _ = environment.step(action)

        # 에피소드 종료 여부를 done 또는 truncated로 판단
        episode_over = done or truncated

        # 스텝 수 증가
        time_step += 1

        # 보상 누적
        episode_reward += reward

        # 현재 상태 업데이트
        state = next_state

    # 에피소드가 끝난 후 결과 출력
    print("Episode Time : ", time_step)         # 전체 스텝 수
    print("Episode Reward : ", episode_reward)  # 누적 보상 (Acrobot는 -스텝 수가 될 것)

    # 수집한 프레임들을 GIF로 저장
    save_frames_as_gif(frames, path=f"random_{environment_name}.gif")


# 스크립트로 실행될 때만 main() 실행
if __name__ == "__main__":
    main()
