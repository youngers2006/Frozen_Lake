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
        self.model = np.full(
            shape=(num_s,num_a),
            fill_value=None
        )
        self.times_visited = np.zeros(shape=(num_s,num_a))

    def update_model(self, state, action, reward, state_):
        new_results = np.array([reward, state_])
        self.model[state, action] = np.append(self.model[state, action], new_results, axis=1)
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
        self.epsilon = max(epsilon_final, self.epsilon - decay_rate)
    
    def planning(self, num_iterations):
        for n in range(num_iterations):
            state = np.choose(choices=range(num_states)) # need to correct, get from all sampled model states
            action = np.choose(choices=)
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
action_space = env.action_space
planning_steps = 50

reward_list = []
agent = Agent(
    epsilon_I,
    Learn_Rate,
    gamma,
    num_states,
    num_actions,
    action_space
)

for episode in range(episodes):
    state, info = env.reset(seed=seed)
    action = agent.select_action(state)
    episode_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        state_, reward, terminated, truncated, info = env.step(action)
        action_ = agent.select_action(state_)
        agent.update_Q(state,state_,action,action_,reward)
        agent.update_model(state,action,reward, state_)
        agent.planning(planning_steps)
        episode_reward += reward
    reward_list.append(episode_reward)
    agent.decay_epsilon(ep_decay)
    state = state_
    action = action_






        

    


