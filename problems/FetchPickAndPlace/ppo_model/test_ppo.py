from stable_baselines3 import PPO

import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
from env.FetchPickAndPlace_env import create_env


def test_model():
    env = create_env(render=True)

    model_path = os.path.join(_HERE, '../models/fetch_pick_and_place_ppo')
    model = PPO.load(model_path)

    obs, _ = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    test_model()
