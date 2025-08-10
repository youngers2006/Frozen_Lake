import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm

class Agent:
    def __init__(self, epsilon, alpha, gamma, lambda_, action_space, num_a, num_s):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_ = lambda_
        self.action_space = action_space
        self.num_actions = num_a
        self.num_states = num_s
        self.Q = np.random.uniform(
            low=0.0,
            high=0.01,
            size=(num_s,num_a)
        )
        self.E_Trace = np.zeros(shape=(num_s,num_a))
        self.trained = False

    def select_action(self, state):
        if np.random.uniform(low=0.0, high=1.0, size=(1,)) >= self.epsilon:
            action = np.argmax(self.Q[state,:])
        else:
            action = self.action_space.sample()
        return action

    def update_Q_E(self, state, state_, action, action_, reward):
        self.E_Trace[state, action] += 1
        TD_error = reward + gamma * self.Q[state_, action_] - self.Q[state, action]
        self.Q = self.Q + self.alpha * self.E * TD_error
        self.E = self.gamma * self.lambda_ * self.E

    def decay_epsilon(self, decay_rate):
        self.epsilon = self.epsilon * decay_rate



env = gym.make('FrozenLake-v1', is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

seed = 42
episodes = 100
epsilon_I = 1.0
epsilon_final = 0.01
ep_decay = (epsilon_I - epsilon_final) / (episodes - 20)
gamma = 0.99
lambda_ = 0.9
Learn_Rate = 0.05

reward_list = []
agent = Agent(
    epsilon_I,
    Learn_Rate,
    gamma,
    lambda_,
    env.action_space,
    num_actions,
    num_states
)

for episode in tqdm(range(episodes), leave=False):
    initial_observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    state = initial_observation
    action = agent.select_action(state)
    total_reward = 0

    while not (terminated or truncated):
        state_, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        action_ = agent.select_action(state_)
        agent.update_Q_E(
            state,
            state_,
            action,
            action_,
            reward
        )

        state = state_
        action = action_

    reward_list.append(total_reward)
    agent.decay_epsilon(ep_decay)
    seed = seed + episode

plt.plot(reward_list)
plt.show()