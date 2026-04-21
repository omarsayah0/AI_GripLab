import gymnasium as gym 
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

def create_env(render=True):
    if render:
        mode = "human"
    else:
        mode = None
    return gym.make("FetchReach-v4", render_mode=mode)