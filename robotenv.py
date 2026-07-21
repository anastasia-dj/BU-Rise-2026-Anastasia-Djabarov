import math
import random
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
 
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
        self.w_step = w_step # TODO: TUNE!

        self.max_steps = max_steps        
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

        # How many rows Q-Table will be
        self.x_bins = math.ceil(self.WIDTH / cell_size) 
        self.y_bins = math.ceil(self.HEIGHT / cell_size)
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
        x_new, y_new, theta_new = self.closed_form_step(self.x, self.y, self.theta, v, w, self.t_step)

        self.x, self.y, self.theta = x_new, y_new, theta_new
        self.steps_taken += 1

        # TODO: Only checks if position boundary is reached at the END of an action, ignoring the possibility of crossing OVER the goal
        terminated = self.distance_to_goal(self.x, self.y) <= self.position_boundary
        truncated = self.steps_taken >= self.max_steps

        reward = self.goal_reward if terminated else self.step_penalty

        return self.discretize(self.x, self.y, self.theta), reward, terminated, truncated, {}
    
    def reset(self, seed=None, options=None):
        # Resets the state by spawning the robot configuration anywhere on the map (Except for the position boundary)
        super().reset(seed=seed)

        while True:
            self.x = random.uniform(0, self.WIDTH)
            self.y = random.uniform(0, self.HEIGHT)
            if self.distance_to_goal(self.x, self.y) > self.position_boundary:
                break
        self.theta = random.uniform(0, 2*math.pi)
        self.steps_taken = 0
        
        return self.discretize(self.x, self.y, self.theta), {}
    
    # --- Visualization such, such ---
    def get_action(self, action):
        # Given an ID, returns the according control input
        v, w = self.actions[action]
        return v, w, self.t_step


# TODO: Add 3D Matplotlib
def replay(env, robot_images, Qtable, start=None, max_steps=200, delay_ms=5):
    # Replay the animation of the trained agent
    import pygame
    pygame.init()
    env_display = environment.Environment(dimensions=(env.WIDTH, env.HEIGHT), show_tree_building=True, has_obstacles=False)
    agent = robot_module.Robot(start=(0, 0, 0), robot_img=robot_images, length=0.01)

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
        
        if truncated:
            print(f"[EVENT] Ran out of steps without finding the goal, with a final distance of {env.distance_to_goal(env.x, env.y):.2f}")

    pygame.time.delay(1500)
    pygame.quit()