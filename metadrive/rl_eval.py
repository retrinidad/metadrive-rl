import os
import sys
import logging
import numpy as np
import cv2

# Add parent directory to path to use local metadrive
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAVE_PATH = "./models"
NUM_EPISODES = 5
VIDEO_PATH = os.path.join(SAVE_PATH, "eval_video.mp4")
SCREEN_SIZE = (800, 800)
FPS = 30

# --------------------------------------------------
# Environment Configuration (headless — no Panda3D window)
# --------------------------------------------------
env_config = {
    "map_config": {
        "type": "block_sequence",
        "config": "r",
        "lane_num": 1,
        "lane_width": 3.5,
        "exit_length": 30,
    },
    "use_render": False,
    "traffic_density": 0.0,
    "random_traffic": False,
    "accident_prob": 0.0,
    "crash_vehicle_done": True,
    "out_of_road_done": True,
    "success_reward": 10.0,
    "out_of_road_penalty": 5.0,
    "crash_vehicle_penalty": 5.0,
}

# --------------------------------------------------
# Load normalization stats and model
# --------------------------------------------------
vec_env = DummyVecEnv([lambda: MetaDriveEnv(env_config)])
vec_env = VecNormalize.load(os.path.join(SAVE_PATH, "vec_normalize.pkl"), vec_env)
vec_env.training = False      # freeze running stats
vec_env.norm_reward = False   # don't normalize rewards at eval time

model = PPO.load(os.path.join(SAVE_PATH, "ppo_final"), device="cpu")

# --------------------------------------------------
# Video writer
# --------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_writer = cv2.VideoWriter(VIDEO_PATH, fourcc, FPS, SCREEN_SIZE)

logger.info("Starting headless evaluation with BEV recording...")

# --------------------------------------------------
# Evaluation loop
# --------------------------------------------------
results = []

for episode in range(NUM_EPISODES):
    obs = vec_env.reset()
    done = False
    episode_reward = 0.0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = vec_env.step(action)
        done = dones[0]
        episode_reward += reward[0]
        step += 1

        # Capture bird's-eye-view frame
        frame = vec_env.envs[0].render(
            mode="topdown",
            window=False,
            screen_size=SCREEN_SIZE,
            draw_target_vehicle_trajectory=True,
            film_size=(2000, 2000),
        )
        if frame is not None:
            # pygame surface comes out RGB, cv2 expects BGR
            video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    info = infos[0]
    outcome = "success" if info.get("arrive_dest", False) else \
              "crash" if info.get("crash", False) or info.get("crash_vehicle", False) else \
              "out_of_road" if info.get("out_of_road", False) else "timeout"

    results.append({
        "episode": episode + 1,
        "reward": episode_reward,
        "steps": step,
        "outcome": outcome,
    })
    logger.info(f"Episode {episode + 1}: reward={episode_reward:.2f}  steps={step}  outcome={outcome}")

video_writer.release()
vec_env.close()

# --------------------------------------------------
# Summary
# --------------------------------------------------
rewards = [r["reward"] for r in results]
lengths = [r["steps"] for r in results]
successes = sum(1 for r in results if r["outcome"] == "success")

logger.info("=" * 40)
logger.info(f"Episodes:     {NUM_EPISODES}")
logger.info(f"Avg reward:   {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
logger.info(f"Avg length:   {np.mean(lengths):.0f}")
logger.info(f"Success rate: {successes}/{NUM_EPISODES} ({100 * successes / NUM_EPISODES:.0f}%)")
logger.info(f"Video saved:  {os.path.abspath(VIDEO_PATH)}")
