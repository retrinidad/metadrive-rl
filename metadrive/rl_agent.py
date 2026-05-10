import os
import logging
import sys
from importlib.util import find_spec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
TOTAL_TIMESTEPS = 500_000
CHECKPOINT_FREQ = 10_000
SAVE_PATH = "./models"
LOG_PATH = "./logs"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)
TENSORBOARD_LOG = LOG_PATH if find_spec("tensorboard") is not None else None
PROGRESS_BAR = False #find_spec("tqdm") is not None and find_spec("rich") is not None
LOG_INTERVAL = 5000

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

def make_env():
    env = MetaDriveEnv(env_config)
    env = Monitor(env, LOG_PATH)
    return env

# MetaDrive uses a singleton engine — only one active environment per process is supported.
# A single env is used for both training and evaluation.
train_env = DummyVecEnv([make_env])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

# --- Callbacks ---
checkpoint_cb = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=SAVE_PATH,
    name_prefix="ppo_metadrive"
)


class SimpleLoggingCallback(BaseCallback):
    """Minimal logging callback to report training progress without extra deps."""
    def __init__(self, log_freq: int = LOG_INTERVAL, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # self.num_timesteps is maintained by SB3
        if self.num_timesteps % self.log_freq == 0:
            logger.info(f"Training timesteps: {self.num_timesteps}")
        return True

# --- Model ---
model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    device="cpu",           # MlpPolicy is faster on CPU
    tensorboard_log=TENSORBOARD_LOG,
    verbose=1,
)

logger.info("Starting training...")
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_cb, SimpleLoggingCallback(log_freq=LOG_INTERVAL)],
    progress_bar=PROGRESS_BAR
)

# Save final model + normalization stats
model.save(os.path.join(SAVE_PATH, "ppo_final"))
train_env.save(os.path.join(SAVE_PATH, "vec_normalize.pkl"))
logger.info("Model saved.")

# --- Evaluation ---
# Switch the env to eval mode: freeze normalization stats, stop normalizing reward
train_env.training = False
train_env.norm_reward = False

model = PPO.load(os.path.join(SAVE_PATH, "ppo_final"), env=train_env)

obs = train_env.reset()
episode_reward = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = train_env.step(action)
    episode_reward += reward
    if done:
        logger.info(f"Episode reward: {episode_reward}")
        episode_reward = 0
        obs = train_env.reset()

train_env.close()
