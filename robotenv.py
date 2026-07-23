import csv
import math
import random
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

import environment
import robot as robot_module

class RobotEnv(gym.Env):
    """
    The environment the Q-learning agent (Robot) lives in, including a visualization of the trained agent in the environment.
    """

    def __init__(self, map_dimensions=(640, 480), goal=(320, 240), position_boundary=10, cell_size=40, theta_bins=8, v_magnitude=10.0, t_step=4.0, w_step=0.4, include_reverse=False, max_steps=200, step_penalty=-1.0, goal_reward=100.0):
        super().__init__()
        
        # Map and goal
        self.WIDTH, self.HEIGHT = map_dimensions
        self.goal_x, self.goal_y = goal
        self.position_boundary = position_boundary

        # State discretization 
        self.cell_size = cell_size # Discretization of (x,y), px per grid box (Size of each cell)
        self.theta_bins = theta_bins # Discretization of heading (How many heading slices)
        self.theta_bin_size_deg = 360.0 / theta_bins

        # Physical constants
        self.v_magnitude = v_magnitude
        self.t_step = t_step # Equivalent of edge duration
        self.w_step = w_step 

        self.max_steps = max_steps        
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

        # How many rows Q-Table will be
        self.x_bins = math.ceil(self.WIDTH / cell_size) 
        self.y_bins = math.ceil(self.HEIGHT / cell_size)
        self.goal_x_bin = min(max(int(self.goal_x // self.cell_size), 0), self.x_bins - 1)
        self.goal_y_bin = min(max(int(self.goal_y // self.cell_size), 0), self.y_bins - 1)
        n_states = self.x_bins * self.y_bins * self.theta_bins # Total number of states
        

        # How many columns Q-Table will be
        self.actions = [(self.v_magnitude, 0.0), (self.v_magnitude, self.w_step), (self.v_magnitude, -self.w_step)]
        if include_reverse: # Possibly include reverse (Though kRRT cannot reverse so... Probably not)
            self.actions += [(-self.v_magnitude, 0.0), (-self.v_magnitude, self.w_step), (-self.v_magnitude, -self.w_step)]

        # Create observation and action space
        self.observation_space = spaces.Discrete(n_states)
        self.action_space = spaces.Discrete(len(self.actions))

        # Starting position
        self.x = self.y = self.theta = 0.0
        self.steps_taken = 0

    # --- Robot Physics and Position ---
    def closed_form_step(self, x, y, theta, v, w, t):
        # Copy of Robot's closed_form_step 
        if abs(w) < 1e-3:
            return x + v * t * math.cos(theta), y - v * t * math.sin(theta), theta
        r = v / w
        theta_fin = theta + w * t
        x_fin = x + r * (math.sin(theta_fin) - math.sin(theta))
        y_fin = y + r * (math.cos(theta_fin) - math.cos(theta))
        return x_fin, y_fin, theta_fin

    def in_goal_cell(self, x, y):
        x_bin = min(max(int(x // self.cell_size), 0), self.x_bins-1)
        y_bin = min(max(int(y // self.cell_size), 0), self.y_bins-1)
        return x_bin == self.goal_x_bin and y_bin == self.goal_y_bin

    def distance_to_goal(self, x, y):
        # Returns the distance to goal
        return math.hypot(x - self.goal_x, y - self.goal_y)
    
    # --- Qtable discretization, step and reset field ---
    def discretize(self, x, y, theta_rad):
        # Turns a state into an ID for the Qtable row
        x_bin = min(max(int(x//self.cell_size), 0), self.x_bins-1)
        y_bin = min(max(int(y//self.cell_size), 0), self.y_bins-1)
        theta_deg = math.degrees(theta_rad) % 360
        theta_bin = min(max(int(theta_deg//self.theta_bin_size_deg), 0), self.theta_bins-1)
        return x_bin * self.y_bins * self.theta_bins  +  y_bin * self.theta_bins  +  theta_bin * 1
    
    def step(self, action):
        # Called per each action. Returns the new_state, reward, terminated, truncated, and info
        v, w = self.actions[action]
        n_checks = 15 # Number of times checked whether robot reached goal
        for k in range(1, n_checks + 1):
            t = self.t_step * k / n_checks
            x_new, y_new, theta_new = self.closed_form_step(self.x, self.y, self.theta, v, w, t)
            elapsed_t = t
            if self.in_goal_cell(x_new, y_new):
                elapsed_t = t
                break # Goal is reached midway through
        
        x_new, y_new, theta_new = self.closed_form_step(self.x, self.y, self.theta, v, w, elapsed_t)

        self.x, self.y, self.theta = x_new, y_new, theta_new
        self.steps_taken += 1

        terminated = self.in_goal_cell(x_new, y_new)
        truncated = self.steps_taken >= self.max_steps

        reward = self.goal_reward if terminated else self.step_penalty

        return self.discretize(self.x, self.y, self.theta), reward, terminated, truncated, {"elapsed_t": elapsed_t}
    
    def reset(self, seed=None, options=None):
        # Resets the state by spawning the robot configuration anywhere on the map (Except for the position boundary, with theta restrictions a certain distance from the walls (where there are no solutions))
        super().reset(seed=seed)
        r = self.v_magnitude / self.w_step # The radius of the turn the robot can make (Used for theta bounding)

        while True:
            self.x = random.uniform(0, self.WIDTH)
            self.y = random.uniform(0, self.HEIGHT)
            self.theta = random.uniform(0, 2*math.pi)

            if self.in_goal_cell(self.x, self.y):
                continue
            
            # Check if robot is near the edges of the map or faced towards it
            near_x_bottom = self.x < r and math.cos(self.theta) < 0
            near_x_top = self.x > self.WIDTH - r and math.cos(self.theta > 0)
            near_y_bottom = self.y < r and math.sin(self.theta) < 0
            near_y_top = self.y > self.HEIGHT - r and math.sin(self.theta) > 0

            if near_x_bottom or near_x_top or near_y_bottom or near_y_top:
                continue # Do not consider positions which are impossible to drive out of

            break

        self.steps_taken = 0
        return self.discretize(self.x, self.y, self.theta), {}
    
    # --- Visualization such, such ---
    def get_action(self, action):
        # Given an ID, returns the according control input
        v, w = self.actions[action]
        return v, w, self.t_step


# TODO: Add 3D Matplotlib
def replay(env, robot_images, Qtable, start=None, max_steps=200, delay_ms=5, run_number=None):
    # Replay the animation of the trained agent
    import pygame
    pygame.init()
    env_display = environment.Environment(dimensions=(env.WIDTH, env.HEIGHT), show_tree_building=True, has_obstacles=False)
    agent = robot_module.Robot(start=(0, 0, 0), robot_img=robot_images, length=0.01)

    if run_number is not None:
        random.seed(run_number+1)

    # Allow for user to pass in their own starting position (start). If not, randomly generate.
    if start is None:
        state, _ = env.reset()
    else:
        env.x, env.y, env.theta = start
        state = env.discretize(env.x, env.y, env.theta)

    clock = pygame.time.Clock()
    trail = [] 

    for step in range(max_steps):
        clock.tick(env_display.FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Greedy policy for acting
        action = int(np.argmax(Qtable[state]))
        v, w, t = env.get_action(action)
        x0, y0, theta0 = env.x, env.y, env.theta

        # Advance environment
        state, reward, terminated, truncated, info = env.step(action)

        # Regenerate the arc so it looks like the robot is driving 
        xs, ys, thetas = agent.generate_arc_samples(x0, y0, theta0, v, w, t, delta=0.05)
        for x, y, theta in zip(xs, ys, thetas):
            agent.update_visual(x, y, math.degrees(theta))
            env_display.map.fill(env_display.WHITE)
            # Draw the position boundary. TODO: Add this in kRRT too
            pygame.draw.circle(env_display.map, env_display.RED, (int(env.goal_x),int(env.goal_y)), int(env.position_boundary), width=1)
            trail.append((int(x), int(y)))
            for px, py in trail[::3]:
                pygame.draw.circle(env_display.map, env_display.BROWN, (px, py), 1)
            agent.draw(map=env_display.map)
            pygame.display.update()
            pygame.time.delay(delay_ms)
        
        if terminated:
            print(f"[EVENT] Reached the goal in {step+1} actions")
            break
        
        if truncated:
            print(f"[EVENT] Ran out of steps without finding the goal, with a final distance of {env.distance_to_goal(env.x, env.y):.2f}")
            break

    pygame.time.delay(1500)
    pygame.quit()

def evaluate_and_log(env, Qtable, max_steps, n_eval_episodes, file_name, run_number, **params_to_log):
    """
    QLearning equivalent of kRRT's log_results()
    """
    random.seed(run_number)
    successes = 0
    costs = []
    actions_on_success = []
    failure_distances = []

    # Go through each episode
    for _ in range(n_eval_episodes):
        state, info = env.reset()
        distance_traveled = 0.0
        # Go through each step
        for step in range(max_steps):
            action = int(np.argmax(Qtable[state]))
            state, reward, terminated, truncated, info = env.step(action)
            distance_traveled += env.v_magnitude * info["elapsed_t"] 
            if terminated: # Goal is reached
                successes += 1
                actions_on_success.append(step + 1)
                costs.append(distance_traveled)
                break 
            if truncated: # Max steps 
                failure_distances.append(env.distance_to_goal(env.x, env.y))
                break
    
    # Take the averages (Like kRRT)
    success_rate = successes / n_eval_episodes
    avg_cost = sum(costs)/len(costs) if costs else None
    std_cost = float(np.std(costs)) if costs else None
    avg_actions = sum(actions_on_success)/len(actions_on_success) if actions_on_success else None
    std_actions = float(np.std(actions_on_success)) if actions_on_success else None
    avg_failure_distance = sum(failure_distances)/len(failure_distances) if failure_distances else None
    std_failure_distance = float(np.std(failure_distances)) if failure_distances else None

    # Create a csv file
    file_exists = os.path.isfile(file_name)
    with open(file_name, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            header = (["run_number", "success_rate", 
                        "avg_cost", "std_cost",
                        "avg_actions_on_success", "std_actions",
                        "avg_failure_distance", "std_failure_distances"] 
                        + list(params_to_log.keys()))
            writer.writerow(header)
        writer.writerow([run_number, success_rate, 
                            avg_cost, std_cost,
                            avg_actions, std_actions,
                            avg_failure_distance, std_failure_distance] 
                            + list(params_to_log.values()))

    return success_rate, avg_cost, std_cost, avg_actions, std_actions, avg_failure_distance, std_failure_distance