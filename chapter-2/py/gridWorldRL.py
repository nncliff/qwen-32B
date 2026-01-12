import numpy as np
import random

class GridWorld:
    def __init__(self, size: int = 5, start =(0, 0), goal=(4, 4)):
        self.size = size
        self.start = start  # Start at specified start position
        self.goal = goal  # Goal at specified goal position
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action: str):
        x, y = self.state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.size - 1, y + 1)
        
        self.state = (x, y)
        
        done = self.state == self.goal
        
        return self.state, done

    def get_possible_actions(self):
        return [0, 1, 2, 3]  # up, down, left, right

class Agent:
    def __init__(self, env: GridWorld):
        self.env = env
        self.q_table = np.zeros((env.size, env.size, 4))  # Q-values for each state-action pair
        self.alpha = 0.1  # learning rate
        self.gamma = 0.9  # discount factor
        self.epsilon = 0.1  # exploration rate

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.get_possible_actions())  # Explore
        else:
            x, y = state
            return np.argmax(self.q_table[x, y])  # Exploit

    def learn(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        predict = self.q_table[x, y, action]
        target = reward + self.gamma * np.max(self.q_table[next_x, next_y])
        self.q_table[x, y, action] += self.alpha * (target - predict)

    def predict_next_state(self, state, action):
        x, y = state
        if action == 0:  # up
            x = max(0, x - 1)
        elif action == 1:  # down
            x = min(self.env.size - 1, x + 1)
        elif action == 2:  # left
            y = max(0, y - 1)
        elif action == 3:  # right
            y = min(self.env.size - 1, y + 1)
        return (x, y)

    def self_supervised_reward(self, state, action, next_state):
        # Predict next state and compute error
        predicted_next_state = self.predict_next_state(state, action)
        # Compute L1 error
        error = np.sum(np.abs(np.array(predicted_next_state) - np.array(next_state)))
        return -error  # Negative error as reward, smaller error -> higher reward

    def train(self, episodes: int = 1000):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, done = self.env.step(action)
                reward = self.self_supervised_reward(state, action, next_state)
                self.learn(state, action, reward, next_state)
                state = next_state

env = GridWorld()
agent = Agent(env)

agent.train()

# print learned Q-table
for i in range(env.size):
    for j in range(env.size):
        print(f"State ({i},{j}): {agent.q_table[i,j]}")