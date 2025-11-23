import numpy as np
from random import random, choice, randint

# --> ENVIRONMENT
NUM_ROW = 4
NUM_COL = 3

ENV = [(0, 0, 1), 
       (0, -1, 0), 
       (0, 0, 0), 
       (-1, 0, 0)] 

ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)] # su, giu, sx, dx
NUM_ACTIONS = len(ACTIONS)
REWARD = -0.01

# --> IPERPARAMETRI Q-LEARNING
ALPHA = 0.1    # Learning Rate
GAMMA = 0.9    # Discount Factor
EPSILON = 0.1  # Exploration Rate
EPISODES = 1000 # Episodi da giocare

def step(state, action_ixd):
    r, c = state
    move_r, move_c = ACTIONS[action_ixd]

    next_r = r + move_r
    next_c = c + move_c

    next_r = max(0, min(next_r, NUM_ROW - 1))
    next_c = max(0, min(next_c, NUM_COL - 1))

    reward = ENV[next_r][next_c] + REWARD

    done = False
    if ENV[next_r][next_c] == 1 or ENV[next_r][next_c] == -1:
        done = True
    
    return (next_r, next_c), reward, done

def choose_action(state, Q):
    row, col = state

    if random() < EPSILON:
        action_idx = randint(0, NUM_ACTIONS - 1)
    else:
        action_idx = np.argmax(Q[row][col])

    return action_idx

def q_learning():
    Q = np.zeros((NUM_ROW, NUM_COL, NUM_ACTIONS))

    for episode in range(EPISODES):
        state = (randint(0, NUM_ROW - 1), randint(0, NUM_COL - 1))

        while True:
            action_idx = choose_action(state, Q)
            next_state, reward, done = step(state, action_idx)
            r, c = state
            next_r, next_c = next_state

            old_value = Q[r][c][action_idx]
            next_max = max(Q[next_r][next_c])

            new_value = old_value + ALPHA * (reward + GAMMA * next_max - old_value)
            Q[r][c][action_idx] = new_value

            state = next_state

            if done:
                break
    return Q

def main():
    Q = q_learning()
    print(Q)

main()