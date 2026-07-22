import numpy as np
import gymnasium as gym
import random
from tqdm import tqdm
from robotenv import RobotEnv

class QLearning():
    """
    Holds the methods for the QLearning algorithm, including initalization, training, and evaluation. 
    Empty currently.
    """

    def __init__(self, env, learning_rate=0.7, gamma=0.95):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.state_space = env.observation_space.n
        self.action_space = env.action_space.n
        self.Qtable = self.initialize_q_table(self.state_space, self.action_space)

    @staticmethod
    def initialize_q_table(state_space, action_space):
        return np.zeros((state_space, action_space))

    # "Updating policy" half of off-policy
    def greedy_policy(self, state):
        # Return the action with highest value (EX: action = 0,1,2,3 - each row is 4 actions)
        return int(np.argmax(self.Qtable[state][:]))
       
    # "Acting policy" half of off-policy
    def epsilon_greedy_policy (self, state, epsilon):
        if random.uniform(0, 1) > epsilon:
            # Be greedy (Exploitation)
            action = self.greedy_policy(state)
        else:
            # Search randomly (Exploration)
            while True:
                action = self.env.action_space.sample()
                v, w = self.env.actions[action]
                x_fin, y_fin, _ = self.env.closed_form_step(self.env.x, self.env.y, self.env.theta, v, w, self.env.t_step)
                if 0 <= x_fin <= self.env.WIDTH and 0 <= y_fin <= self.env.HEIGHT:
                    return action
        return action
    
    
    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps):
        for episode in tqdm(range(n_training_episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode) # Runs from 1 --> 0, expn
            state, info = self.env.reset() # From the gynasium library

            for step in range(max_steps):
                action = self.epsilon_greedy_policy(state, epsilon)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                # Reward = R{t+1}, everything else is the table talking to itself (I think?)
                self.Qtable[state][action] = self.Qtable[state][action] + self.learning_rate*(reward + self.gamma*np.max(self.Qtable[new_state]) - self.Qtable[state][action])
                # Equation with TD Target and such such from the lecture, "nudges" data towards the actual values
                if terminated or truncated:
                    break
                state = new_state
            
        # Returned for cosemetic purposes 
        return self.Qtable
    
    def evaluate_agent(self, max_steps, n_eval_episodes, seed=None):
        episode_rewards = []
        for episode in tqdm(range(n_eval_episodes)):
            if seed: # If a seed is given, reset according to that seed. Otherwise, reset randomly. 
                state, info = self.env.reset(seed=seed[episode])
            else:
                state, info = self.env.reset()
            
            step = 0
            truncated = False
            terminated = False
            total_rewards_ep = 0

            for step in range(max_steps):
                action = self.greedy_policy(state)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                total_rewards_ep += reward
                if terminated or truncated:
                    break
                state = new_state
            episode_rewards.append(total_rewards_ep)

        # Evaluation metrics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        return mean_reward, std_reward
    
    