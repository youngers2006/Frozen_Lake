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
        policy[action] = P_greedy
        
        TD_error = reward + np.sum(policy * self.Q[state_,self.action_space]) - self.Q[state,action]
        self.Q[state,action] = self.Q[state,action] + self.alpha * TD_error
    
    def decay_ep(self, decay_rate):
        self.epsilon = self.epsilon * decay_rate
    
# Hyper Params
alpha = 0.05
gamma = 0.99
epsilon_I = 1.0
epsilon_final = 0.01
num_episodes = 1000
ep_decay_rate = (epsilon_I - epsilon_final) / num_episodes
seed = 42

# Create environment
env = gym.make('FrozenLake-v1', is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

reward_list = []
agent = Agent(
    epsilon_I,
    alpha,
    gamma,
    env.action_space,
    num_actions,
    num_states
)

for episode in range(num_episodes):
    episode_reward = 0
    state, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    action = agent.select_action(state)
    while not (terminated or truncated):
        state_, reward, terminated, truncated, info= env.step(action)
        episode_reward += reward
        action_ = agent.select_action(state_)
        agent.update_Q(
            state,
            state_,
            reward,
            action,
            action_
        )
        state = state_
        action = action_

    agent.decay_ep(ep_decay_rate)
    reward_list.append(episode_reward)

plt.plot(reward_list)
plt.show()


