from stable_baselines3 import PPO
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from env.FetchReach_env import create_env


def train_model():
    env = create_env(render=False)

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
    model.save(os.path.join(_HERE, "models/fetch_reach_ppo"))

    env.close()


if __name__ == "__main__":
    train_model()
