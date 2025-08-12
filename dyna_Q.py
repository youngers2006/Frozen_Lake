import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm

class Agent:
    def __init__(self, epsilon, alpha, gamma, num_s, num_a):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.random.uniform(
            low=0.0, 
            high=0.01, 
            size=(num_s,num_a)
        )
        self.model = np.zeros(
            low=0.0, 
            high=0.01, 
            size=(num_s,num_a)
        )

    def update_model(self, state, action, reward, state_):
        new_results = np.array([reward, state_])
        self.model[state, action] = np.append(self.model[state,action], new_results)
        
        

    


