import gymnasium as gym
import numpy as np
import tqdm
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, epsilon, alpha, gamma, action_space, num_a, num_s):
        self.Q = np.zeros(shape=(num_s,num_a))
        self.epsilon = epsilon
        self.num_actions = num_a
        self.num_states = num_s
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space

    def select_action(self, state):
        P_greedy = self.epsilon / self.num_actions
        P_non_greedy = 1 - self.epsilon + self.epsilon / self.num_actions

        best_a = np.argmax(self.Q[state,:])

        probabilities = np.full(P_non_greedy)
        probabilities[best_a] += P_greedy

        action = np.random.choice(a=self.action_space, p=probabilities)

        return action

    def update_Q(self, state, state_, reward, action, action_):
        P_greedy = self.epsilon / self.num_actions
        P_non_greedy = 1 - self.epsilon + self.epsilon / self.num_actions
        policy = np.full(shape=(self.num_actions), fill_value=P_non_greedy)
        
        TD_error = reward + np.sum(policy * self.Q[state_,self.action_space]) - self.Q[state,action]
        self.Q[state,action] = self.Q[state,action] + self.alpha * TD_error

    def get_Q(self):
        return self.Q

        
env = gym.make('FrozenLake-v1', is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

