import flappy_bird_env
import gymnasium as gym
import time,os
from stable_baselines3 import PPO,DQN
env = gym.make("FlappyBird-v0",render_mode="human")
import numpy as np
model = DQN("MlpPolicy", env, verbose=1,learning_rate=1e-5,gamma=0.99)
# model.learn(total_timesteps=1000)

obs_tuple = env.reset()
obs = np.concatenate(obs_tuple)
for i in range(100):
    obs = env.reset()
    total_reward = 0
    while True:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        print("Total Reward : ",total_reward)
        if done:
            break
print("Total Reward : ",total_reward)
env.close()

# observation -> action space is planned to passed as an input layer for the cnn model and output layer with 2 o/p -> action
# dqn agent is gonna be built using the above model using .... (memory,explo)
