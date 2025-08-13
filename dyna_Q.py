import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

class Agent:
    def __init__(self, epsilon, alpha, gamma, num_s, num_a, action_space):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.action_space = action_space
        self.num_actions = num_a
        self.num_states = num_s
        self.Q = np.random.uniform(
            low=0.0, 
            high=0.01, 
            size=(num_s,num_a)
        )
        self.model = np.zeros(
            shape=(num_s,num_a)
        )
        self.times_visited = np.zeros(shape=(num_s,num_a))

    def update_model(self, state, action, reward, state_):
        new_results = np.array([reward, state_])
        self.model[state, action] = (self.times_visited[state,action] * self.model[state, action, :] + new_results) / (self.times_visited[state,action] + 1)
        self.times_visited[state,action] += 1

    def update_Q(self, state, state_, action, action_, reward):
        TD_error = reward + self.gamma * np.max(self.Q[state_, :]) - self.Q[state, action]
        self.Q[state,action] = self.Q[state,action] + self.alpha * TD_error

    def select_action(self, state):
        if np.random.uniform(low=0.0,high=1.0,size=(1,0)) >= self.epsilon:
            action = np.argmax(self.Q[state,:])
        else:
            action = self.action_space.sample()
        return action
    
    def decay_epsilon(self,decay_rate):
        self.epsilon = self.epsilon * decay_rate
    
    def planning(self, num_iterations):
        for n in range(num_iterations):
            state = 0 # need to correct, get from all sampled model states
            action = 0 # same as state
            reward, state_ = self.model[state,action]
            TD_error = reward + self.gamma * np.max(self.Q[state_,:]) - self.Q[state,action]
            self.Q[state,action] = self.Q[state,action] + self.alpha * TD_error

env = gym.make('FrozenLake-v1', is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

seed = 42
episodes = 1000
epsilon_I = 1.0
epsilon_final = 0.01
ep_decay = (epsilon_I - epsilon_final) / (episodes - 20)
gamma = 0.99
Learn_Rate = 0.05

reward_list = []
agent = Agent(

)


        

    


