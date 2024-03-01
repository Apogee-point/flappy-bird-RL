from stable_baselines3 import PPO,DQN

from stable_baselines3.common.vec_env import DummyVecEnv
from flappy_bird_env import FlappyBirdEnv
import gymnasium as gym
import os
import time

# env = DummyVecEnv([lambda: FlappyBirdEnv()])
env = gym.make('FlappyBird-v0',render_mode="human")
env.reset()


model = PPO.load(path='models/1700488450/1000000.zip',env=env)
episodes=100

for ep in range(episodes):
	obs=env.reset()
	done=False
	totalReward=0
	while not done:
		action,_states=model.predict(obs)
		obs,reward,done,_,info=env.step(action)
		totalReward+=reward
		env.render()
	print(f"Episode {ep} score: {info['score']}, reward: {totalReward}")
		
	