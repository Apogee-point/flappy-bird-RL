from stable_baselines3 import PPO,DQN

from stable_baselines3.common.vec_env import DummyVecEnv
from flappy_bird_env import FlappyBirdEnv
import gymnasium as gym
import os
import time


# env = DummyVecEnv([lambda: FlappyBirdEnv()])

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)
	

env = gym.make('FlappyBird-v0',render_mode="human")
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

# model = PPO.load('models/1700295867/1000000.zip',env=env)

TIMESTEPS = 10000
iters = 0
for i in range(150):
	print(f"Training iteration {i}")
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=logdir)
	model.save(f"{models_dir}/{TIMESTEPS*iters}")



# def evaluate(model, num_episodes=100):
#     for i in range(num_episodes):
#         obs = env.reset()
#         done = False
#         while not done:
#             action, _ = model.predict(obs)
#             obs, _, done, _ = env.step(action)
#             env.render()  # Add your render mode here
            
# evaluate(model)