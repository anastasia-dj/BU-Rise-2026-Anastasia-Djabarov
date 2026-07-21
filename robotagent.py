import argparse
import math
import numpy as np
import qlearning
from robotenv import RobotEnv, replay

# Parser Arguments (CLI)
parser = argparse.ArgumentParser(description='Q-learning for a nonholonomic car reaching a fixed goal from any start. (Aka, the RL mirror of kRRT reversibility trials).')
parser.add_argument('-pb', '--position_boundary', type=int, metavar='', required=False, default=20, help='Allowed distance (px) from the goal.')
parser.add_argument('-goal', '--goal', nargs=2, type=float, metavar='', required=False, default=[340.0, 260.0], help='Goal position in x(px), y(px).')
parser.add_argument('-ts', '--t_step', type=float, metavar='', required=False, default=4.0, help='Seconds each action is held for, the equivalent of the edge_length_duration for kRRT.')
parser.add_argument('-sp', '--step_penalty', type=float, metavar='', required=False, default=-1.0, help='The penalty given for every action that does not reach the goal.')
parser.add_argument('-gr', '--goal_reward', type=float, metavar='', required=False, default=100.0, help='Reward given the instant the goal is reached.')
parser.add_argument('-cs', '--cell_size', type=int, metavar='', required=False, default=40, help='The size of the cells in the grid')
parser.add_argument('-tb', '--theta_bins', type=int, metavar='', required=False, default=8, help='Number of theta bins')
parser.add_argument('-vmag', '--v_magnitude', type=float, metavar='', required=False, default=10, help='The magnitude of linear velocity used.')
parser.add_argument('-wmag', '--w_magnitude', type=float, metavar='', required=False, default=0.5, help='The magnitude of angular velocity used. (0 angular velocity is also always an action the robot can take regardless of w_mag)')
parser.add_argument('-rev', '--use_reverse', type=bool, action=argparse.BooleanOptionalAction, metavar='', required=False, default=False, help='Whether or not to allow the robot to pursue -v_magnitude')
args = parser.parse_args()

# Constants
MAP_DIMENSIONS = (680, 520) # Used to be 640 480

# CLI arguments
goal = args.goal
position_boundary = args.position_boundary # Default 20, TODO: Work down to 10 or 5
cell_size = args.cell_size # Default 40
theta_bins = args.theta_bins # Default 8
v_magnitude = args.v_magnitude # Default 10
w_step = args.w_magnitude # Default 0.5
include_reverse = args.use_reverse # Default off
t_step = args.t_step # Default 4.0
step_penalty = args.step_penalty
goal_reward = args.goal_reward

# TODO: Add CLI arguments for below. Currently placeholder values:
n_training_episodes = 30000
learning_rate = 0.7
gamma = 0.9
max_steps = 200
n_eval_episodes = 200
max_epsilon = 1.0
min_epsilon = 0.05
decay_rate = 0.0002

# Sanity check
px_per_action = v_magnitude * t_step
actions_per_cell = cell_size/px_per_action
print(f"[SANITY] One t_step {t_step} moves {px_per_action:.2f}px. {actions_per_cell} actions to cross one cell (Assuming w=0).")
if actions_per_cell > 4:
    print(f"[SANITY] actions_per_cell has been flagged as too high. Shrink cell_size, raise t_step, or raise max_steps per action.")

env = RobotEnv(
    map_dimensions=MAP_DIMENSIONS,
    goal=goal,
    position_boundary=position_boundary,
    cell_size=cell_size,
    theta_bins=theta_bins,
    v_magnitude=v_magnitude,
    t_step=t_step,
    w_step=w_step,
    include_reverse=include_reverse,
    max_steps=max_steps,
    step_penalty=step_penalty,
    goal_reward=goal_reward
)

print("Observation Space", env.observation_space)
print("Action Space", env.action_space)

agent = qlearning.QLearning(env, learning_rate=learning_rate, gamma=gamma)
Qtable_robot = agent.train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps)

np.save("robot_qtable.npy", Qtable_robot)

mean_reward, std_reward = agent.evaluate_agent(max_steps, n_eval_episodes, seed=None)
print(f"Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")

robot_images = ['images/initial.png', 'images/goal.png', 'images/robot.png']
replay(env, robot_images, Qtable_robot, max_steps=200)