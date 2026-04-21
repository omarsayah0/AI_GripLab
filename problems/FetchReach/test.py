from stable_baselines3 import PPO
import time
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from env.FetchReach_env import create_env


def test_model():
    env = create_env(render=True)

    model = PPO.load(os.path.join(_HERE, "models/fetch_reach_ppo"))

    obs, _ = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        time.sleep(0.01)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    test_model()
