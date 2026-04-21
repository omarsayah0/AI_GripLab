from stable_baselines3 import SAC
import time
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
from env.FetchPickAndPlace_env import create_env


def test_model():
    env = create_env(render=True)
    model_path = os.path.join(_HERE, '../models/fetch_pick_and_place_sac_her')
    model = SAC.load(model_path, env=env)

    obs, _ = env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.1)
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()


if __name__ == "__main__":
    test_model()
