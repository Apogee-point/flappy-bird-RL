from stable_baselines3 import DQN
import flappy_bird_env
import gymnasium as gym
# Load the saved model
env = gym.make('FlappyBird-v0',render_mode="human")
model = DQN.load("/Users/saiganeshs/Developer/flappy-bird-env-main/logs/1699335153/DQN_0")

# Use the loaded model to make predictions
obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()