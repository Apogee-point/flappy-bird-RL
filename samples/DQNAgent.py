import flappy_bird_env
import gymnasium as gym
import tensorflow as tf
import numpy as np
import random
from collections import deque
from PIL import Image

env = gym.make("FlappyBird-v0",render_mode="human")


state_size = env.observation_space.shape  
action_size = env.action_space.n 
learning_rate = 0.001
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration rate
memory = deque(maxlen=2000)  # Replay memory



model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(800, 576, 3)),
    tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(action_size, activation='linear')
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))


batch_size = 32
num_episodes = 1000


for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state[0], (1, *state_size))
    target_size = (84, 84)  # Adjust the size as needed
    state = Image.fromarray(state[0])
    state = state.resize(target_size)
    state = np.array(state) / 255.0
    #TODO print(state)
    total_reward = 0  # Initialize total reward for each episode

    while True:
        if np.random.rand() <= epsilon:
            action = random.randrange(action_size)
        else:
            action = np.argmax(model.predict(state)[0])

        next_state, reward, done, _,_ = env.step(action)
        next_state = np.reshape(next_state, (1, *state_size))
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward  # Accumulate the reward

        if done:
            break

        if len(memory) >= batch_size:
            minibatch = random.sample(memory, batch_size)

            for state, action, reward, next_state, done in minibatch:
                target = reward
                if not done:
                    target = (reward + gamma * np.amax(model.predict(next_state)[0]))

                target_f = model.predict(state)
                target_f[0][action] = target
                model.fit(state, target_f, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    if episode % 100 == 0:
        model.save("flappy_bird_dqn_model.h5")

    print(f"Episode: {episode}, Total Reward: {total_reward}")

model.save("flappy_bird_dqn_model.h5")
