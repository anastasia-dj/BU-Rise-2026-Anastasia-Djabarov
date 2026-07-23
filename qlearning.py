import csv

import numpy as np
import gymnasium as gym
import random
from tqdm import tqdm
from robotenv import RobotEnv
import matplotlib.pyplot as plt
import os

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

        # Create a copy of the qtable
        q_values = self.Qtable[state].copy()
        for i in range(len(q_values)):
            v, w = self.env.actions[i]
            x_fin, y_fin, _ = self.env.closed_form_step(self.env.x, self.env.y, self.env.theta, v, w, self.env.t_step)
            if not (0 <= x_fin <= self.env.WIDTH and 0 <= y_fin <= self.env.HEIGHT):
                # Temporarily make action = negative infinity if impossible (thus, do not choose it) 
                q_values[i] = -np.inf
        
        return int(np.argmax(q_values))
       
    # "Acting policy" half of off-policy
    def epsilon_greedy_policy (self, state, epsilon):
        if random.uniform(0, 1) > epsilon:
            # Be greedy (Exploitation)
            action = self.greedy_policy(state)
        else:
            # Search randomly (Exploration)
            max_tries = 50

            for _ in range(max_tries):
                # Check action space
                action = self.env.action_space.sample()
                v, w = self.env.actions[action]
                # Check where it would end (NOTE: The robot can go temporarily on the edge as long as it returns to the map. May be an issue in the real-world implementation of this)
                x_fin, y_fin, _ = self.env.closed_form_step(self.env.x, self.env.y, self.env.theta, v, w, self.env.t_step)

                # Only save action if it lands within the map
                if 0 <= x_fin <= self.env.WIDTH and 0 <= y_fin <= self.env.HEIGHT:
                    break
                #     return action
        return action
    
    
    def train(self, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps):
        for episode in tqdm(range(n_training_episodes)):
            epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode) # Runs from 1 --> 0, expn
            state, info = self.env.reset() # From the gynasium library

            for step in range(max_steps):
                action = self.epsilon_greedy_policy(state, epsilon)
                new_state, reward, terminated, truncated, info = self.env.step(action)
                # Reward = R{t+1}, everything else is the table talking to itself (I think?)
                # NOTE: Constant learning rate. Should be updated to be nonconstant?
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
    
    def find_episodes_for_convergence(self, max_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, file_name, checkpoint_interval=500, convergence_threshold=1.0):
        """
        Trains until the largest change in any single state's best Qvalue measured over the checkpoint_interval drops below the convergence_threshold.
        """
        
        # Copy and collapse Qtable into Qvalues for states
        prev_checkpoint = np.max(self.Qtable, axis=1).copy()
        file_exists = os.path.isfile(file_name)
        qvalue_history = []

        with open(file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["episode", "max_delta", "max_delta_ratio"])
            
            for episode in tqdm(range(max_episodes)):
                epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*episode)
                state, info = self.env.reset()

                # From train() (Does not call train becase otherwise the epsilon is wiped)
                for step in range(max_steps):
                    action = self.epsilon_greedy_policy(state, epsilon)
                    new_state, reward, terminated, truncated, info = self.env.step(action)
                    # Reward = R{t+1}, everything else is the table talking to itself (I think?)
                    self.Qtable[state][action] = self.Qtable[state][action] + self.learning_rate*(reward + self.gamma*np.max(self.Qtable[new_state]) - self.Qtable[state][action])
                    # Equation with TD Target and such such from the lecture, "nudges" data towards the actual values
                    if terminated or truncated:
                        break
                    state = new_state
                
                if (episode + 1) % checkpoint_interval == 0:
                    current_checkpoint = np.max(self.Qtable, axis=1).copy()
                    qvalue_history.append((episode+1, current_checkpoint.copy()))
                    state_with_max_delta = np.argmax(np.abs(current_checkpoint - prev_checkpoint))
                    max_delta = float(np.abs(current_checkpoint[state_with_max_delta] - prev_checkpoint[state_with_max_delta])) # Biggest Q value diff within an INDIVIDUAL state
                    ratio = max_delta / (np.abs(prev_checkpoint[state_with_max_delta]) + 1e-8) # Ratio of delta Q (NOTE: Unreliable at early training bc denominator is close to zzero)
                    writer.writerow([episode + 1, max_delta, ratio])

                    if max_delta < convergence_threshold:
                        print(f"[CONVERGE] Converged after {episode + 1} episodes with a max_delta={max_delta:.4f}")
                        self.save_and_plot_qvalue_history(qvalue_history, file_name)
                        return self.Qtable, episode + 1
                    prev_checkpoint = current_checkpoint.copy()
                
        print(f"[CONVERGE] Did not converge within {max_episodes}")
        self.save_and_plot_qvalue_history(qvalue_history, file_name)
        return self.Qtable, max_episodes
    
    def save_and_plot_qvalue_history(self, qvalue_history, file_name):
        base_name = file_name.replace(".csv", "")
        n_states = len(qvalue_history[0][1])
        
        # Creates a file history of the qvalues
        history_file = f"{base_name}_qvalue_history.csv"
        with open(history_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["episode", "state_id", "qvalue"])
            for episode, snapshot in qvalue_history:
                for state_id, qvalue in enumerate(snapshot):
                    writer.writerow([episode, state_id, qvalue])
        
        episodes = [ep for ep, _ in qvalue_history]

        # Absolute Q table:
        plt.figure(figsize=(10, 6))
        for state_id in range(n_states):
            values = [snapshot[state_id] for _, snapshot in qvalue_history]
            plt.plot(episodes, values, linewidth=0.5, alpha=0.3)
        plt.xlabel("Episode")
        plt.ylabel("Qvalue (best action per state)")
        plt.title(f"Absolute Qvalue per State ({n_states} States)")
        plt.tight_layout()
        plt.savefig(f"{base_name}_qvalue_absolute.png", dpi=150)
        plt.close()

        # One line per state showing how much its changed since the previous checkpoint:
        plt.figure(figsize=(10, 6))
        for state_id in range(n_states):
            values = [snapshot[state_id] for _, snapshot in qvalue_history]
            deltas = [0] + [values[i] - values[i - 1] for i in range(1, len(values))]
            plt.plot(episodes, deltas, linewidth=0.5, alpha=0.3)
        plt.xlabel("Episode")
        plt.ylabel("Change in Qvalue Since Last Checkpoint")
        plt.title(f"Qvalue Change per State ({n_states} States)")
        plt.tight_layout()
        plt.savefig(f"{base_name}_qvalue_delta.png", dpi=150)
        plt.close()

        print(f"Saved {history_file}, {base_name}_qvalue_absolute.png, {base_name}_qvalue_delta.png!")
