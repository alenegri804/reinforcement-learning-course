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
PROBS = [0.8, 0.1, 0.1]

REWARD = -0.05
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

def get_expected_value(s, action_idx, V):
    # Mappatura delle deviazioni in base all'indice dell'azione in ACTIONS
    # 0(Su) -> devia in 2(Sx), 3(Dx)
    # 1(Giù) -> devia in 2(Sx), 3(Dx)
    # 2(Sx) -> devia in 0(Su), 1(Giù)
    # 3(Dx) -> devia in 0(Su), 1(Giù)
    deviations_map = {
        0: [0, 2, 3],
        1: [1, 2, 3],
        2: [2, 0, 1],
        3: [3, 0, 1]
    }
    
    possible_actions_indices = deviations_map[action_idx]
    expected_value = 0
    
    # Iteriamo su: Azione Principale (0.8), Deviazione 1 (0.1), Deviazione 2 (0.1)
    for i, idx in enumerate(possible_actions_indices):
        prob = PROBS[i]
        action = ACTIONS[idx]
        
        next_r, next_c = get_next_state(s, action)
        value = REWARD + ENV[next_r][next_c] + GAMMA * V[next_r][next_c]
        
        expected_value += prob * value
        
    return expected_value

def policyEvaluation(V, P):
    while True:
        delta = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                action_idx = int(P[r][c])
                
                new_v = get_expected_value((r, c), action_idx, V)

                old_v = V[r][c]
                V[r][c] = new_v

                delta = max(delta, abs(new_v - old_v))
        if delta < TRESHOLD:
            break

    return V

def policyImprovement(V,P):
    policy_stable = True
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
                
        if old_action != best_action:
            policy_stable = False

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

