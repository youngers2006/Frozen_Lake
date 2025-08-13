import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

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
        self.model = {}

    def update_model(self, state, action, reward, state_, terminated):
        self.model[(state, action)] = (reward, state_, terminated)

    def update_Q(self, state, state_, action, reward, terminated):
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[state_, :])

        TD_error = target - self.Q[state, action]
        self.Q[state,action] = self.Q[state,action] + self.alpha * TD_error

    def select_action(self, state):
        if np.random.uniform(low=0.0,high=1.0,size=(1,)) >= self.epsilon:
            action = np.argmax(self.Q[state,:])
        else:
            action = self.action_space.sample()
        return action
    
    def decay_epsilon(self,decay_rate):
        self.epsilon = max(epsilon_final, self.epsilon - decay_rate)
    
    def planning(self, num_iterations):
        if not self.model:
            return
        
        visited_sa_pairs = list(self.model.keys())

        for _ in range(num_iterations):
            state, action = random.choice(visited_sa_pairs) # need to correct, get from all sampled model states
            reward, state_, terminated = self.model[(state,action)]
            self.update_Q(state, state_, action, reward, terminated)

env = gym.make('FrozenLake-v1', is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

seed = 42
episodes = 20000
epsilon_I = 1.0
epsilon_final = 0.001
ep_decay = (epsilon_I - epsilon_final) / (episodes - 1000)
gamma = 0.99
Learn_Rate = 0.05
action_space = env.action_space
planning_steps = 10

reward_list = []
agent = Agent(
    epsilon_I,
    Learn_Rate,
    gamma,
    num_states,
    num_actions,
    action_space
)

for episode in tqdm(range(episodes), leave=False):
    state, info = env.reset(seed=seed)
    action = agent.select_action(state)
    episode_reward = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        state_, reward, terminated, truncated, info = env.step(action)
        action_ = agent.select_action(state_)
        agent.update_Q(state, state_, action, reward, terminated)
        agent.update_model(state, action, reward, state_, terminated)
        agent.planning(planning_steps)
        episode_reward += reward
    reward_list.append(episode_reward)
    agent.decay_epsilon(ep_decay)
    state = state_
    action = action_
    seed = seed + 1

plt.plot(reward_list)
plt.show()

bin_size = 20
bins = []
for i in range(len(reward_list) // 20):
    idx = i * 20
    np_rew = np.array(reward_list[i:(i + bin_size)]) 
    bin_i = np.sum(np_rew)
    bins.append(bin_i)

plt.plot(bins)
plt.show()






        

    


