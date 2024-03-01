import gymnasium as gym
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

# Define the Flappy Bird environment
env = gym.make("FlappyBird-v0")

# Define the number of actions
nb_actions = env.action_space.n

# Define the shape of the input state
input_shape = (84, 84)

# Build a neural network model
model = Sequential()
model.add(
    Conv2D(
        32,
        kernel_size=(8, 8),
        strides=(4, 4),
        activation="relu",
        input_shape=(1,) + input_shape,
    )
)
model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation="relu"))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))

# Initialize the DQN agent
memory = SequentialMemory(limit=2000, window_length=1)
policy = EpsGreedyQPolicy(eps=0.1)
dqn = DQNAgent(
    model=model,
    nb_actions=nb_actions,
    memory=memory,
    nb_steps_warmup=500,
    target_model_update=1e-2,
    policy=policy,
)

dqn.compile(Adam(lr=1e-3), metrics=["mae"])

# Train the agent
dqn.fit(env, nb_steps=100000, visualize=False, verbose=1)

# Evaluate the agent (optional)
# dqn.test(env, nb_episodes=10, visualize=True)

# Save the trained model
dqn.save_weights("flappy_bird_dqn_weights.h5", overwrite=True)
