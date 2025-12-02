import numpy as np
from random import random
import gymnasium as gym

env = gym.make('CliffWalking-v1')
state_space = env.observation_space.n 
action_space = env.action_space.n

QTable = np.zeros((state_space, action_space))

cost = 0.01
alpha = 0.9
gamma = 0.5

EPISODES = 500
EPSILON_START = 1
EPSILON_DECAY = EPSILON_START / (EPISODES / 2)

def choose_action(env, QTable, state, epsilon):
    pass

def qLearning(QTable, env):
    epsilon = EPSILON_START
    
    for episode in range(EPISODES):
        state, info = env.reset()
        
        while True:
            # TODO: da modificare
            if random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(QTable[state]) 
            
            next_state, reward, terminated, truncated, info = env.step(action)

            old_val = QTable[state, action]
            next_max = np.max(QTable[next_state])

            new_val = old_val + alpha * (reward + gamma * next_max - old_val)
            QTable[state, action] = new_val

            state = next_state
            
            if terminated or truncated:
                break
        
        if epsilon > 0:
            epsilon -= EPSILON_DECAY
            epsilon = max(0, epsilon) 

    return QTable

def main():
    # Carichiamo l'ambiente. 
    # Nota: Se 'CliffWalking-v1' da errore, usa 'CliffWalking-v0' (lo standard attuale)
    try:
        env = gym.make('CliffWalking-v1') 
    except:
        print("Versione v1 non trovata, utilizzo CliffWalking-v0")
        env = gym.make('CliffWalking-v0')

    # Dimensioni dinamiche dall'ambiente
    state_space = env.observation_space.n # 48 stati
    action_space = env.action_space.n     # 4 azioni
    
    QTable = np.zeros((state_space, action_space))

    print("Training in corso...")
    Q = qLearning(QTable, env)
    print("Training completato.\n")

    print("Output Legend:")
    print("U: Up, R: Right, D: Down, L: Left")
    print("****: Cliff (Precipizio)")
    
    # Mapping Azioni standard Gymnasium: 0:Up, 1:Right, 2:Down, 3:Left
    moves_symbols = {0: "U", 1: "R", 2: "D", 3: "L"}

    # Dimensioni fisse per CliffWalking
    ROWS = 4
    COLS = 12

    # Stampa della griglia
    for r in range(ROWS):
        row_str = "|"
        for c in range(COLS):
            # Calcoliamo l'indice piatto dello stato (0-47)
            state_idx = r * COLS + c
            
            # Gestione grafica delle zone speciali (basata sulle coordinate)
            if r == 3 and 1 <= c <= 10:
                row_str += " **** |" # Cliff
            elif r == 3 and c == 11:
                row_str += " GOAL |" # Goal
            elif r == 3 and c == 0:
                # Start (mostriamo comunque la mossa migliore per uscire)
                best_action = np.argmax(Q[state_idx])
                row_str += f" S:{moves_symbols[best_action]}|"
            else:
                # Celle normali: mostriamo la policy appresa
                best_action = np.argmax(Q[state_idx])
                row_str += f"  {moves_symbols[best_action]}   |"
        
        print("-" * len(row_str))
        print(row_str)
    print("-" * len(row_str))

main()