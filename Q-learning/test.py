import numpy as np
from random import random, randint

# --> ENV
NUM_ROW = 4
NUM_COL = 3

ENV = [(0, 0, 1), 
       (0, -1, 0), 
       (0, 0, 0), 
       (-1, 0, 0)] 

ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)] # su, giu, sx, dx
NUM_ACTIONS = len(ACTIONS)
REWARD = -0.01

# IPERPARAMETERS
GAMMA = 0.9
EPSILON = 0.1
ALPHA = 0.1

EPISODES = 5000

def step(state, action_idx):
    r, c = state
    move_r, move_c = ACTIONS[action_idx]

    next_r = r + move_r
    next_r = max(0, min(next_r, NUM_ROW - 1))

    next_c = c + move_c
    next_c = max(0, min(next_c, NUM_COL -1))

    reward = ENV[next_r][next_c] + REWARD

    done = False
    if ENV[next_r][next_c] == 1 or ENV[next_r][next_c] == -1:
        done = True

    return (next_r, next_c), reward, done

def choose_action(state, Q):
    r, c = state

    if random() < EPSILON:
        action_idx = randint(0, NUM_ACTIONS - 1)
    else:
        action_idx = np.argmax(Q[r][c])

    return action_idx

def q_learning(Q):
    for episode in range(EPISODES):
        state = (randint(0, NUM_ROW - 1), randint(0, NUM_COL - 1))

        while True:
            r, c = state
            action_idx = choose_action(state, Q)
            (next_r, next_c), reward, done = step(state, action_idx)

            old_val = Q[r][c][action_idx]
            next_max = max(Q[next_r][next_c])

            new_val = old_val + ALPHA * (reward + GAMMA * next_max - old_val)
            Q[r][c][action_idx] = new_val

            state = (next_r, next_c)

            if done:
                break
    return Q

def main():
    Q = np.zeros((NUM_ROW, NUM_COL, NUM_ACTIONS))
    Q = q_learning(Q)
    print(Q)