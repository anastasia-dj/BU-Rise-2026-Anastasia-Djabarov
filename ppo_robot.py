"""
Early implementation of PPO RL on the robot.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from robotenv import RobotEnv, replay_model
robot_images = ['images/initial.png', 'images/goal.png', 'images/robot.png']

# Create environment
env = RobotEnv(map_dimensions=(420, 420), goal=(210.0, 210.0), position_boundary=14, cell_size=28, theta_bins=12, v_magnitude=10.0, t_step=2.8, w_step=0.5, include_reverse=True, max_steps=200, step_penalty=-1.0, goal_reward=100.0, continuous_obs=True)
check_env(env) 
env = Monitor(env, filename="ppo_training_log")

# Instantiate the agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent and display a progress bar
model.learn(total_timesteps=int(1000000), progress_bar=True)

# Save the agent
model.save("ppo_robot")

del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO.load("ppo_robot", env=env)

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=50)

# Enjoy trained agent
replay_model(env.unwrapped, robot_images, model)
