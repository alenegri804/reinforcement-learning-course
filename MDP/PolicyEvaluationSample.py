import numpy as np
from random import randint
from pprint import pprint

# --> ENV
NUM_ROW = 3
NUM_COL = 5
ENV = [(0, 0, -1, 1, 0),
       (-1, 0, 0, 0, 0),
       (0, 0, -1, 0, -1)]

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
# NUM_ACTIONS = len(ACTIONS)

REWARD = -0.1
GAMMA = 0.9
TRESHOLD = 1e-3

def get_next_state(s, a):
    row, col = s
    move_r, move_c = a
    
    next_r = row + move_r
    next_r = max(0, min(next_r, NUM_ROW - 1))

    next_c = col + move_c
    next_c = max(0, min(next_c, NUM_COL - 1))

    return (next_r, next_c)

def policyEvaluation(V, P):
    while True:
        delta = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                action_idx = int(P[r][c])
                action = ACTIONS[action_idx]
                
                next_r, next_c = get_next_state((r, c), action)
                new_v = REWARD + ENV[next_r][next_c] + GAMMA*V[next_r][next_c]

                old_v = V[r][c]
                V[r][c] = new_v

                delta = max(delta, abs(new_v - old_v))
        if delta < TRESHOLD:
            break

    return V

def policyImprovement(V,P):

    policy_stable = False
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            old_action = P[r][c]

            values = []
            for action in ACTIONS:
                next_r, next_c = get_next_state((r, c), action)
                val = REWARD + ENV[next_r][next_c] + GAMMA*V[next_r][next_c]
                values.append(val)

            best_action = np.argmax(values)
            P[r][c] = best_action
                
        if old_action == best_action:
            policy_stable = True

    return P, policy_stable

def policyIteration(V, P):
    print(P, "\n")
    while True:
        print("Valutazione in corso...")
        V = policyEvaluation(V, P)

        print("Miglioramento Policy...\n")
        P, policy_stable = policyImprovement(V, P)
        print(P, "\n")

        if policy_stable:
            print("Policy ottimizzata!")
            break
    



def main():
    V_init = np.zeros((NUM_ROW, NUM_COL))
    P_init = np.zeros((NUM_ROW, NUM_COL))
    policyIteration(V_init, P_init)

main()

