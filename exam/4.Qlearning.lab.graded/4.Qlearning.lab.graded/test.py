# your code here
import numpy as np
from random import random, randint

ROWS = 3
COLS = 4
nb_action = 4

QTable = np.zeros((ROWS, COLS, nb_action))
WALL = [1, 1]

REWARD_MAP = np.zeros((ROWS, COLS))
REWARD_MAP[0, 3] = 1   
REWARD_MAP[1, 3] = -1 

cost = 0.01
alpha = 0.9
gamma = 0.5

EPSILON_START = 1
EPSILON_DECAY = 0.01
EPISODES = 100

def step(state, QTable, epsilon):
    row, col = state

    if random() < epsilon:
        action_idx = randint(0, nb_action - 1)
    else:
        action_idx = np.argmax(QTable[row][col])

    actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    move_r, move_c = actions[action_idx]

    next_r = row + move_r
    next_c = col + move_c
    next_r = max(0, min(next_r, ROWS - 1))
    next_c = max(0, min(next_c, COLS - 1))

    if next_r == WALL[0] AND next_c == WALL[1]:
        next_r = row
        next_c = col

    next_state = (next_r, next_c)

    act_reward = REWARD_MAP[next_r][next_c] - cost

    isTerminated = False
    if REWARD_MAP[next_r][next_c] = -1 or REWARD_MAP[next_r][next_c] = 1:
        isTerminated = True

    return action_idx, next_state, act_reward, isTerminated

def qLearning(QTable):
    epsilon = EPSILON_START
    for episode in range(EPISODES):
        state = (2, 0)

        while True:
            action_idx, next_state, act_reward, isTerminated = step(state, QTable, epsilon)
            r = state[0]
            c = state[1]
            next_r = next_state[0]
            next_c = next_col[1]

            old_val = QTable[r][c][action_idx]
            next_max = max(QTable[next_r][next_c])

            new_val = old_val + alpha * (act_reward + gamma * next_max - old_val)
            QTable[r][c][action_idx] = new_val

            state = next_state

            if isTerminated:
                break
            
        epsilon -= EPSILON_DECAY
        epsilon = max(0, epsilon) 

    return QTable

def main():
    Q =qLearning(QTable)
    print(Q)

main()