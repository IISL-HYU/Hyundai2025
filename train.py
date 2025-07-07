from dqn import DQN
from ddpg import DDPG
import gymnasium as gym

def main():
    train_episodes = 200
    evaluation_episodes = 5
    evaluation_interval = 20

    ################################################################################
    # ## Experiment 1 DQN
    # environment_name = 'Acrobot-v1'

    # environment = gym.make(environment_name)
    # dqn = DQN(environment=environment)

    # dqn.train(train_episodes=train_episodes, evaluation_episodes=evaluation_episodes, evaluation_interval=evaluation_interval)

    # dqn.save_model(path=f"hyundai2025_logs/dqn/{environment_name}")
    ################################################################################


    ################################################################################
    # Experiment 2 DDPG
    environment_name = "Pendulum-v1"

    environment = gym.make(environment_name)
    ddpg = DDPG(environment=environment)

    ddpg.train(train_episodes=train_episodes, evaluation_episodes=evaluation_episodes, evaluation_interval=evaluation_interval)

    ddpg.save_model(path=f"hyundai2025_logs/ddpg/{environment_name}")
    ################################################################################


if __name__=="__main__":
    main()