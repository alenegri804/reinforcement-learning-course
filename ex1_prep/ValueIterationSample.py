import numpy as np
from pprint import pprint
from random import randint

# --> ENVIRONMENT
NUM_ROW = 4
NUM_COL = 3
ENV = [(0, 0, 1), 
       (0, -1, 0), 
       (0, 0, 0), 
       (-1, 0, 0)] 

NUM_ACTIONS = 4
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)] # su, giu, sinistra, destra

REWARD = -0.1
GAMMA = 0.9
THETA = 0.0001

def get_next_state(s, a):
    row, col = s
    move_row, move_col = a

    next_row = row + move_row
    next_row = max(0, min(next_row, NUM_ROW - 1))

    next_col = col + move_col
    next_col = max(0, min(next_col, NUM_COL - 1))

    return (next_row, next_col)

# --> VALUE ITERATION

def valueIteration(V):
    i = 1
    while True:
        delta = 0
        for r in range(NUM_ROW):
            for c in range(NUM_COL):
                old_v = V[r][c]
                values = []
                for action in ACTIONS:
                    next_row, next_col = get_next_state((r, c), action)
                    val = REWARD + ENV[next_row][next_col] + GAMMA*V[next_row][next_col]
                    values.append(val)

                new_v = max(values)
                V[r][c] = new_v

                delta = max(delta, abs(new_v - old_v))

        i += 1
        print(f"Iterazione {i}")
        print(V, "\n")

        if delta < THETA:
            break
    
    return V

# --> POLICY OPTIMISATION

def get_Policy(V, P):
    
    for r in range(NUM_ROW):
        for c in range(NUM_COL):
            values = []
            for action in ACTIONS:
                next_row, next_col = get_next_state((r, c), action)
                val = REWARD + ENV[next_row][next_col] + GAMMA*V[next_row][next_col]
                values.append(val)
            
            new_p = np.argmax(values)
            P[r][c] = new_p
    return P

def print_Policy(P):
    printed_policy = []
    for r in range(NUM_ROW):
        printed_row = []
        for c in range(NUM_COL):
            if P[r][c] == 0:
                printed_row.append("UP")
            elif P[r][c] == 1:
                printed_row.append("DOWN")
            elif P[r][c] == 2:
                printed_row.append("LEFT")
            elif P[r][c] == 3:
                printed_row.append("RIGHT")
        printed_policy.append(printed_row)
    pprint(printed_policy)

# --> TEST

def run_episode(env, policy, start_state):
    # 1. Iniziamo dalla casella di partenza
    current_state = start_state
    r, c = current_state
    
    print(f"🏁 Partenza da: {current_state}")
    
    # Continuiamo a muoverci finché non arriviamo al Tesoro (che ha valore 1)
    # Nota: Usiamo un contatore per evitare loop infiniti per sicurezza
    steps = 0
    max_steps = 20 
    
    while env[r][c] != 1 and steps < max_steps:
        # 2. Chiediamo alla Policy cosa fare qui
        action_idx = int(policy[r][c]) # Es. 0
        action = ACTIONS[action_idx]   # Es. (-1, 0) -> "Su"
        
        # 3. Eseguiamo il movimento
        next_state = get_next_state(current_state, action)
        
        # Stampa di debug per vedere cosa succede
        move_name = ["Su", "Giù", "Sinistra", "Destra"][action_idx]
        print(f"Passo {steps+1}: Sono in {current_state}, la Policy dice '{move_name}' -> Vado in {next_state}")
        
        # 4. Aggiorniamo lo stato per il prossimo giro
        current_state = next_state
        r, c = current_state
        steps += 1

    if env[r][c] == 1:
        print(f"Arrivato al Tesoro in {current_state}!")
    else:
        print("Ho finito i passi prima di trovare il tesoro.")

def main():
    V = np.zeros((NUM_ROW, NUM_COL))
    P = np.zeros((NUM_ROW, NUM_COL))

    V = valueIteration(V)
    P = get_Policy(V, P)
    print_Policy(P)

    print("\n---Test sul Campo ---")
    # Partiamo dalla casella pericolosa vicino al fuoco (1, 0)
    run_episode(ENV, P, (randint(0, NUM_ROW -1), randint(0, NUM_COL -1)))

main()