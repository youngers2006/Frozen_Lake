import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import gymnasium as gym
from tqdm import tqdm

env = gym.make('FrozenLake-v1', is_slippery=True)
num_states = env.observation_space.n
num_actions = env.action_space.n

seed = 42
episodes = 100
epsilon = 1.0
ep_decay = 0.9
gamma = 0.99
lambda_ = 0.9
Learn_Rate = 0.05

key = jax.random.PRNGKey(seed=seed)

def policy(Q, state, epsilon, key):

    key, subkey = jax.random.split(key)
    rand = jax.random.uniform(key=subkey, shape=(1,), minval=0.0, maxval=1.0)

    if rand >= epsilon:
        action = int(jnp.argmax(Q[state,:]))
    else:
        action = env.action_space.sample()
    
    return action


# Initialise Q(s,a) for all s,a pairs
# Q is tabular and there are 16 states and 4 actions in each state
Q = jax.random.uniform(key=key,shape=(num_states,num_actions), minval=0.0, maxval=0.001)

E = jnp.zeros(shape=(num_states,num_actions))

reward_list = []

for episode in tqdm(range(episodes), leave=False):
    E = jnp.zeros_like(E)
    initial_observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    state = initial_observation
    action = policy(Q, state, epsilon, key)
    total_reward = 0

    while not (terminated or truncated):

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        state_prime = observation
        action_prime = policy(Q, state_prime, epsilon, key)

        TD_error = reward + gamma * Q[state_prime,action_prime] - Q[state,action]
        Esa = E[state,action]
        E = E.at[state,action].set(Esa + 1)

        for s in range(num_states):
            for a in range(num_actions):
                Qsa = Q[s,a]
                Esa = E[s,a]
                Q = Q.at[s,a].set(Qsa + Learn_Rate * TD_error * Esa)
                E = E.at[s,a].set(gamma * lambda_ * Esa)

        state = state_prime
        action = action_prime

    reward_list.append(total_reward)
    epsilon = epsilon * ep_decay
    seed = seed + episode

plt.plot(reward_list)
plt.show()


reward_list_test = []

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')

rewinds = int(input('number of times to watch'))

for episode in range(rewinds):
    initial_observation, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    state = initial_observation
    action = policy(Q, state, epsilon, key)
    total_reward = 0

    while not (terminated or truncated):

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        state_prime = observation
        action_prime = policy(Q, state_prime, epsilon, key)

        state = state_prime
        action = action_prime

    reward_list_test.append(total_reward)

plt.plot(reward_list_test)
plt.show()




        
