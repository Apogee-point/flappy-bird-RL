import flappy_bird_env
import gymnasium as gym
import random

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory

import tensorflow as tf
from tensorflow import keras

print(keras.__version__)

print(tf.__version__)

env = gym.make("FlappyBird-v0")

model = Sequential()
model.add(Flatten(input_shape=(1, states)))
model.add(Dense(24, activation="relu"))
model.add(Dense(24, activation="relu"))
model.add(Dense(actions, activation="linear"))

agent = DQNAgent(
    model=model,
    memory=SequentialMemory(limit=50000, window_length=1),
    policy=BoltzmannQPolicy(),
    nb_actions=actions,
    nb_steps_warmup=10,
    target_model_update=0.01,
)

agent.compile(Adam(lr=0.001), metrics=["mae"])
agent.fit(env, nb_steps=100000, visualize=False, verbose=1)

results = agent.test(env, nb_episodes=10, visualize=True)
print(np.mean(results.history["episode_reward"]))

