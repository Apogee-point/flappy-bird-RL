import flappy_bird_env
from stable_baselines3.common.callbacks import EvalCallback

from stable_baselines3 import PPO,DQN
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

# Create environment without rendering
env = gym.make('FlappyBird-v0',render_mode="human")

eval_env = gym.make('FlappyBird-v0',render_mode="human")
# Initialize agent with different hyperparameters
model = DQN("CnnPolicy", env, verbose=1,learning_rate=1e-5,gamma=0.80)
#model = DQN("CnnPolicy", env, verbose=1)
#TODO learning_rate=0.0005, buffer_size=1000000, batch_size=64, gamma=0.99
obs = env.reset()

# Now you can render the environment
env.render()

eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=500,
                             deterministic=True, render=False)

# Train the agent with the callback
model.learn(total_timesteps=1000, callback=eval_callback)


# Save the agent
model.save("ppo_flappybird")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

print(f"Mean reward: {mean_reward} +/- {std_reward}")
