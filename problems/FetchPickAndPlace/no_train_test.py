import gymnasium as gym
import gymnasium_robotics
import time
gym.register_envs(gymnasium_robotics)


def run_random():
    env = gym.make("FetchPickAndPlace-v4", render_mode="human")

    obs, _ = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.1)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    run_random()