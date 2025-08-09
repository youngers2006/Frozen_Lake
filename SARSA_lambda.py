import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import gymnasium as gym

env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='human')
num_states = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

seed = 42
episodes = 100
epsilon = 1
ep_decay = 0.9
gamma = 0.99
lambda_ = 0.9
Learn_Rate = 0.1

rand_key = jax.random.PRNGKey(seed=seed)

def policy(Q, state, epsilon):
    rand = jax.random.uniform(key=rand_key, shape=(1,), minval=0.0, maxval=1.0)

    if rand >= epsilon:
        action = jnp.argmax(Q[state,:])
    else:
        action = env.action_space.sample()
    
    return action


# Initialise Q(s,a) for all s,a pairs
# Q is tabular and there are 16 states and 4 actions in each state
Q = jax.random.uniform(key=rand_key,shape=(num_states,num_actions), minval=0.0, maxval=0.001)

E = jnp.zeros(shape=(num_states,num_actions))

for episode in episodes:
    E = jnp.zeros_like(E)
    initial_observation, info = env.reset(seed=42)
    terminated = False
    state = initial_observation
    action = policy(Q, state, epsilon)

    while not terminated:

        observation, reward, terminated, truncated, info = env.step(action)

        state_prime = observation
        action_prime = policy(Q, state_prime, epsilon)

        TD_error = reward + gamma * Q[state_prime,action_prime] - Q[state,action]
        E[state,action] = E[state,action] + 1

        for s in num_states:
            for a in num_actions:
                Q[s,a] = Q[s,a] + Learn_Rate * TD_error * E[s,a]
                E[s,a] = gamma * lambda_ * E[s,a]

        state = state_prime
        action = action_prime

    epsilon = epsilon * ep_decay


        
