from stable_baselines3 import PPO,DQN
import os
import time
import gymnasium as gym

import flappy_bird_env

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = gym.make("FlappyBird-v0")
env.reset()

model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=1e-5, gamma=0.90)
#model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=logdir, learning_rate=1e-5, gamma=0.90);

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	print(iters)
	env.reset()
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"DQN")
	# model.save(f"{models_dir}/{TIMESTEPS*iters}")