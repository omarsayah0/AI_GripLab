# 🤖 AI_GripLab

## Table of Contents

- [Overview](#overview)
- [Project Structure](#️-project-structure)
- [Environments](#environments)
- [Methods](#methods)
  - [PPO (Baseline)](#1️⃣-ppo-baseline)
  - [SAC + HER (Final Solution)](#2️⃣-sac--her-final-solution)
- [Results](#results-fetchpickandplace)
- [Demo](#demo)
- [Installation](#installation)
- [How to run](#how-to-run)
- [Another way to Run](#another-way-to-run)
- [Experiment Summary](#experiment-summary)
- [Key Insights](#key-insights)
- [Conclusion](#conclusion)
- [Contributors](#-contributors)
- [License](#-license)

---

## Overview

This project explores reinforcement learning approaches for robotic manipulation tasks using the Fetch environments from Gymnasium Robotics.

The goal is to train a robotic arm to:

- reach a target position (**FetchReach**)
- pick and place an object (**FetchPickAndPlace**)

---

## 🗂️ Project Structure

```
RL/
├── problems/
│   ├── FetchReach/                     (FR)
│   │   ├── env/
│   │   │   └── FetchReach_env.py       # Environment setup for FetchReach-v4
│   │   ├── models/
│   │   │   └── fetch_reach_ppo.zip     # Saved PPO model
│   │   ├── train.py                    # Train PPO on FetchReach
│   │   ├── test.py                     # Evaluate PPO on FetchReach
│   │   └── no_train_test.py            # Run FetchReach with random actions
│   │
│   └── FetchPickAndPlace/              (FPAP)
│       ├── env/
│       │   └── FetchPickAndPlace_env.py  # Environment setup for FetchPickAndPlace-v4
│       ├── models/
│       │   ├── fetch_pick_and_place_ppo.zip          # Saved PPO model
│       │   ├── fetch_pick_and_place_sac_her.zip      # Saved SAC+HER model
│       │   └── fetch_pick_and_place_sac_her_buffer.pkl  # Replay buffer
│       ├── ppo_model/
│       │   ├── train_ppo.py            # Train PPO on PickAndPlace
│       │   └── test_ppo.py             # Evaluate PPO on PickAndPlace
│       ├── sac_her_model/
│       │   ├── train_sac_her.py        # Train SAC+HER on PickAndPlace (resume-aware)
│       │   └── test_sac_her.py         # Evaluate SAC+HER on PickAndPlace
│       └── no_train_test.py            # Run FetchPickAndPlace with random actions
│
├── Videos/
│   ├── FetchReach/
│   │   ├── before_train.mp4            # Agent behavior before training
│   │   └── PPO_model.mp4               # PPO agent
│   └── FetchPickAndPlace/
│       ├── before_train.mp4            # Agent behavior before training
│       ├── PPO_model.mp4               # PPO agent (failure case)
│       └── SAC_HER_model.mp4           # SAC+HER agent (success case)
│
└── pyproject.toml                      # Project config and CLI commands
```

> **Note:** The `models/` folders are **not included** in this repository because the trained model files are too large. You must train the models yourself before running the test scripts. See the [How to Run](#how-to-run) section for training instructions.

---

## Environments

| Environment | Description |
|---|---|
| **FetchReach-v4** | 4-DOF robotic arm must move its end-effector to a target position |
| **FetchPickAndPlace-v4** | Robotic arm must grasp a block and place it at a target location |

Both environments use **sparse rewards** (0 or -1 per step), making them challenging for standard RL algorithms.

---

## Methods

### 1️⃣ PPO (Baseline)

Proximal Policy Optimization was used as a baseline. Both environments use a `MultiInputPolicy` to handle the dictionary observation space (observation + goal).

#### FetchReach (Simple Task)

PPO performed very well on the simpler reaching task:

- Successfully learned the task in **100,000 timesteps**
- Achieved near **100% success rate**
- Stable and fast convergence

| File | Purpose |
|---|---|
| `FetchReach/train.py` | Trains PPO for 100k steps, saves to `models/fetch_reach_ppo.zip` |
| `FetchReach/env/FetchReach_env.py` | Wraps `FetchReach-v4` with optional render mode |
| `FetchReach/test.py` | Loads saved model and runs 1000 evaluation steps |

---

#### FetchPickAndPlace (Complex Task)

PPO was tested on the pick-and-place task as part of experimentation.

- Trained for **500,000 timesteps**
- The agent quickly converged to **non-optimal repetitive behavior**
- Observed behavior: the robotic arm repeatedly moved backward instead of interacting with the object

Performance (early training):

- Success rate ≈ **3.7%**
- Mean reward ≈ **-48.5**

> **Insight:** This highlights a known limitation of PPO in **sparse-reward environments**, where the agent struggles to discover meaningful sequences of actions without dense reward shaping.

---

### 2️⃣ SAC + HER (Final Solution)

To address the limitations of PPO, we used:

- **Soft Actor-Critic (SAC)** — an off-policy actor-critic algorithm optimizing a maximum-entropy objective
- **Hindsight Experience Replay (HER)** — replays failed episodes with substitute goals, turning failures into learning signal

HER configuration used:
- `n_sampled_goal = 4` — 4 hindsight goals sampled per real transition
- `goal_selection_strategy = "future"` — goals are sampled from future states in the same episode

The training script (`train_sac_her.py`) supports **resuming from a checkpoint**: if a saved model and replay buffer exist, training continues from where it left off rather than starting from scratch.

---

## Results (FetchPickAndPlace)

After training for ~3.75M timesteps (7–8 runs × 500k steps each):

- ✅ Success rate: **96% – 98%**
- 📉 Mean reward: **~ -11 to -13**
- 📈 Stable performance across training

---

## Demo

### FetchReach

#### Before Training

Shows the FetchReach agent acting randomly before any training.

📁 [`Videos/FetchReach/before_train.mp4`](Videos/FetchReach/before_train.mp4)

---

#### PPO (Success Case)

Shows the PPO agent successfully moving the end-effector to the target position.

📁 [`Videos/FetchReach/PPO_model.mp4`](Videos/FetchReach/PPO_model.mp4)

---

### FetchPickAndPlace

#### Before Training

Shows the FetchPickAndPlace agent acting randomly before any training.

📁 [`Videos/FetchPickAndPlace/before_train.mp4`](Videos/FetchPickAndPlace/before_train.mp4)

---

#### PPO (Failure Case)

Shows the agent converging to ineffective behavior (avoiding the object).

📁 [`Videos/FetchPickAndPlace/PPO_model.mp4`](Videos/FetchPickAndPlace/PPO_model.mp4)

---

#### SAC + HER (Success Case)

Shows the agent successfully:

- grasping the object
- lifting it
- placing it at the target

📁 [`Videos/FetchPickAndPlace/SAC_HER_model.mp4`](Videos/FetchPickAndPlace/SAC_HER_model.mp4)

---

## Installation

Requires Python **3.10 – 3.13**.

Clone the repository:

```bash
git clone https://github.com/omarsayah0/AI_GripLab.git
cd AI_GripLab
```

Make a virtual environment.
```bash
python3.10 -m venv venv
```
Activate it:

Windows

    venv\Scripts\activate

Linux / macOS

    source venv/bin/activate


Then install the project (this also registers all CLI commands):

```bash
pip install -e .
```

Once installed, all 8 CLI commands (`FR_PPO_train`, `FR_PPO_test`, etc.) become available globally in your terminal — see the [How to run](#how-to-run) section below.

Or install dependencies manually:

```bash
pip install gymnasium>=1.2.3 gymnasium-robotics>=1.4.1 stable-baselines3>=2.8.0 mujoco>=3.5.0 numpy>=1.26
```

---

## How to run

### CLI Commands

After installing the project with `pip install -e .`, you can run any script directly from any directory using the following commands.

### Command Prefix Reference

| Prefix | Meaning | Environment |
|--------|---------|-------------|
| **FR** | **F**etch**R**each | `FetchReach-v4` — robotic arm moves its end-effector to a target position |
| **FPAP** | **F**etch**P**ick**A**nd**P**lace | `FetchPickAndPlace-v4` — robotic arm grasps a block and places it at a target location |

### All Available Commands

| Command | Description |
|---------|-------------|
| `FR_PPO_train` | Train the PPO model on FetchReach |
| `FR_PPO_test` | Test the trained PPO model on FetchReach (renders visually) |
| `FR_NoTrain_test` | Run FetchReach with random actions — no model needed |
| `FPAP_PPO_train` | Train the PPO model on FetchPickAndPlace |
| `FPAP_PPO_test` | Test the trained PPO model on FetchPickAndPlace (renders visually) |
| `FPAP_SAC_train` | Train (or resume) the SAC+HER model on FetchPickAndPlace |
| `FPAP_SAC_test` | Test the trained SAC+HER model on FetchPickAndPlace (renders visually) |
| `FPAP_NoTrain_test` | Run FetchPickAndPlace with random actions — no model needed |

> **Note:** `FPAP_SAC_train` automatically resumes from the last checkpoint if a saved model and replay buffer already exist.

### Quick Examples

```bash
# Train the best-performing model (SAC+HER on PickAndPlace)
FPAP_SAC_train

# Watch the trained SAC+HER agent
FPAP_SAC_test

# Train PPO on the simpler FetchReach task
FR_PPO_train

# Watch what the robot does before any training (random actions)
FPAP_NoTrain_test
```

---

## Another way to Run

> **Note:** All commands should be run from inside the relevant problem directory (`problems/FetchReach/` or `problems/FetchPickAndPlace/`).

### FetchReach — PPO

```bash
cd problems/FetchReach

# Train
python train.py

# Test
python test.py
```

### FetchPickAndPlace — PPO

```bash
cd problems/FetchPickAndPlace/ppo_model

# Train
python train_ppo.py

# Test
python test_ppo.py
```

### FetchPickAndPlace — SAC + HER

```bash
cd problems/FetchPickAndPlace/sac_her_model

# Train (or resume from saved checkpoint automatically)
python train_sac_her.py

# Test
python test_sac_her.py
```

---

## Experiment Summary

| Method    | Task             | Timesteps | Result            |
|-----------|------------------|-----------|-------------------|
| PPO       | FetchReach       | 100k      | ✅ ~100% success  |
| PPO       | FetchPickAndPlace| 500k      | ❌ Failed (3.7%)  |
| SAC + HER | FetchPickAndPlace| ~3.75M    | ✅ 96–98% success |

---

## Key Insights

- PPO works well for **simple, dense-reward tasks** (e.g., reaching)
- PPO struggles in **sparse-reward, multi-step tasks** — the agent needs to chain many correct actions before receiving any signal
- SAC + HER significantly improves:
  - **Exploration** — entropy regularization in SAC encourages diverse behavior
  - **Sample efficiency** — HER extracts learning signal from every episode, even failed ones
  - **Learning stability** — off-policy replay buffer stabilizes updates over millions of steps

---

## Conclusion

This project demonstrates how algorithm choice is critical in reinforcement learning:

- Simple on-policy algorithms (PPO) may fail in environments with sparse rewards and multi-step dependencies
- Combining an off-policy method (SAC) with goal-conditioned replay (HER) can dramatically improve performance, taking the success rate from **3.7% → 97%**

---

## 👨‍💻 Contributors

**Omar Al ethamat** – AI Engineer

Feel free to open issues or pull requests.

---

## 📄 License

This project is licensed under the MIT License.