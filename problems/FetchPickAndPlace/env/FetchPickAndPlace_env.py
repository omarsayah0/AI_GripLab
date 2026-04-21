import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


def create_env(render=True):
    mode = "human" if render else None
    return gym.make("FetchPickAndPlace-v4", render_mode=mode)