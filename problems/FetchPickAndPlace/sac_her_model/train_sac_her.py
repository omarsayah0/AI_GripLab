from stable_baselines3 import SAC, HerReplayBuffer
import sys
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))
from env.FetchPickAndPlace_env import create_env

MODEL_PATH = os.path.join(_HERE, '../models/fetch_pick_and_place_sac_her')
BUFFER_PATH = os.path.join(_HERE, '../models/fetch_pick_and_place_sac_her_buffer.pkl')


def train_or_continue():
    env = create_env(render=False)

    if os.path.exists(MODEL_PATH + ".zip") and os.path.exists(BUFFER_PATH):
        print("🔁 Loading existing model and buffer...")
        model = SAC.load(MODEL_PATH, env=env)
        model.load_replay_buffer(BUFFER_PATH)

        model.learn(total_timesteps=500_000, reset_num_timesteps=False)

    else:
        print("🆕 Training from scratch...")
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs=dict(
                n_sampled_goal=4,
                goal_selection_strategy="future",
            ),
            verbose=1,
        )

        model.learn(total_timesteps=500_000)

    model.save(MODEL_PATH)
    model.save_replay_buffer(BUFFER_PATH)

    env.close()


if __name__ == "__main__":
    train_or_continue()
