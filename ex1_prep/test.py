import numpy as np
from pprint import pprint

NUM_ROW = 4
NUM_COL = 5
ENV = [(0, 0, 1, 0, 0),
       (0, -1, 0, -1, 0),
       (0, 0, 0, 0, 0),
       (-1, 0, 0, 0, 0)]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

REWARD = -0.1
GAMMA = 0.9
TRESHOLD = 1e-3

def get_next_state(s, a):
    row, col = s
    move_row, move_col = a

    next_row = row + move_row
    next_col = col + move_col

    next_row = max(0, min(next_row, NUM_ROW - 1))
    next_col = max(0, min(next_col, NUM_COL - 1))

    return (next_row, next_col)

def valueIteration(V):
    i = 0
    while True: 
        delta = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                old_v = V[r][c]

                values = []
                for action in ACTIONS:
                    next_row, next_col = get_next_state((r, c), action)
                    val = REWARD + ENV[next_row][next_col] + GAMMA * V[next_row][next_col]
                    values.append(val)
                
                new_v = max(values)
                V[r][c] = new_v

                delta = abs(new_v - old_v)
            
            i +=1
            print(f"Iterazione n {i}")
            print(V, "\n")

        if delta < TRESHOLD:
            break
    return V

def get_policy(V, P):

    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            values = []
            for action in ACTIONS:
                next_row, next_col = get_next_state((r, c), action)
                val = REWARD + ENV[next_row][next_col] + GAMMA * V[next_row][next_col]
                values.append(val)
            
            new_p = np.argmax(values)
            P[r][c] = new_p

    return P

def policyEvaluation(V, P):
    while True:
        delta = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                P_idx = int(P[r][c])
                action = ACTIONS[P_idx]
                next_row, next_col = get_next_state((r, c), action)
                new_v = REWARD + ENV[next_row][next_col] + GAMMA * V[next_row][next_col]

                old_v = V[r][c]
                V[r][c] = new_v
                delta = max(delta, abs(new_v - old_v))
                
        if delta < TRESHOLD:
            break
    return V

def policyImprovement(V, P):
    stability = False
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            old_action = P[r][c]

            values = []
            for action in ACTIONS:
                next_row, next_col = get_next_state((r, c), action)
                val = REWARD + ENV[next_row][next_col] + GAMMA * V[next_row][next_col]
                values.append(val)

            best_action = np.argmax(values)
            P[r][c] = best_action

        if best_action == old_action:
            stability = True
    
    return P, stability

def policyIteration(V, P):
    while True:
        print("Evaluating...")
        V = policyEvaluation(V, P)
        print("Improving...")
        P, stability = policyImprovement(V, P)
        print(P)

        if stability:
            print("Policy ottimizzata")
            break
            
def main():
    V = np.zeros((NUM_ROW, NUM_COL))
    P = np.zeros((NUM_ROW, NUM_COL))
    policyIteration(V, P)

main()