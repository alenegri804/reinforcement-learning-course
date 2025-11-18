import numpy as np
import itertools

## ------------------------------------------------------------------
## 1. DEFINIZIONE DELL'AMBIENTE E DELLE REGOLE
## ------------------------------------------------------------------

class TicTacToeDP:
    def __init__(self):
        # 0: Vuoto, 1: Agente (X), -1: Avversario (O)
        self.empty_board = tuple([0] * 9)
        self.gamma = 0.9  # Fattore di sconto
        self.states = set()
        self.V = {}       # Value Function V(s)
        self.policy = {}  # Policy pi(s) -> action
        
        # Generiamo tutti gli stati possibili (raggiungibili)
        print("Generazione degli stati in corso...")
        self.generate_all_states(self.empty_board, 1)
        print(f"Stati totali trovati: {len(self.states)}")
        
        # Inizializzazione V(s) = 0
        for s in self.states:
            self.V[s] = 0.0
            
    def generate_all_states(self, board, current_player):
        """
        DFS ricorsiva per trovare tutti gli stati legali del gioco.
        """
        if board in self.states:
            return
        
        self.states.add(board)
        
        # Se il gioco è finito, stop
        if self.check_win(board, 1) or self.check_win(board, -1) or 0 not in board:
            return

        # Genera mosse possibili
        possible_moves = [i for i, x in enumerate(board) if x == 0]
        
        for action in possible_moves:
            new_board = list(board)
            new_board[action] = current_player
            self.generate_all_states(tuple(new_board), -1 if current_player == 1 else 1)

    def check_win(self, board, player):
        """Controlla se il 'player' ha vinto"""
        b = np.array(board).reshape(3, 3)
        # Righe, Colonne, Diagonali
        for i in range(3):
            if np.all(b[i, :] == player) or np.all(b[:, i] == player):
                return True
        if np.all(np.diag(b) == player) or np.all(np.diag(np.fliplr(b)) == player):
            return True
        return False

    def get_valid_actions(self, state):
        return [i for i, x in enumerate(state) if x == 0]

    ## ------------------------------------------------------------------
    ## 2. MODELLO DI TRANSIZIONE (DYNAMICS) p(s', r | s, a)
    ## ------------------------------------------------------------------
    
    def transition(self, state, action):
        """
        Restituisce una lista di possibili (prob, next_state, reward, done).
        Simula: Mossa Agente -> (Se vince: Fine) -> Mossa Avversario Random -> (Se vince: Fine) -> Next State
        """
        board_list = list(state)
        
        # 1. Mossa Agente
        board_list[action] = 1 
        new_state = tuple(board_list)
        
        # Controllo vittoria Agente
        if self.check_win(new_state, 1):
            return [(1.0, new_state, 1.0, True)] # Prob 100%, Reward +1, Terminato
            
        # Controllo Pareggio (nessuna mossa rimasta)
        valid_opp_moves = [i for i, x in enumerate(new_state) if x == 0]
        if not valid_opp_moves:
            return [(1.0, new_state, 0.0, True)] # Pareggio, Reward 0
            
        # 2. Mossa Avversario (Gioca Random, come nel file gym originale)
        outcomes = []
        prob = 1.0 / len(valid_opp_moves) # Probabilità uniforme
        
        for opp_action in valid_opp_moves:
            opp_board_list = list(new_state)
            opp_board_list[opp_action] = -1
            final_state = tuple(opp_board_list)
            
            reward = 0.0
            done = False
            
            # Controllo vittoria Avversario
            if self.check_win(final_state, -1):
                reward = -1.0 # Penalità per sconfitta
                done = True
            elif 0 not in final_state:
                reward = 0.0 # Pareggio
                done = True
                
            outcomes.append((prob, final_state, reward, done))
            
        return outcomes

    ## ------------------------------------------------------------------
    ## 3. ALGORITMO DI VALUE ITERATION
    ## ------------------------------------------------------------------

    def value_iteration(self, theta=1e-4):
        """
        Implementazione basata sulla logica di Policy Evaluation/Improvement 
        (Vedi Slide 10 e 14)
        """
        print("Inizio Value Iteration...")
        iteration = 0
        while True:
            delta = 0
            for s in self.states:
                # Se è uno stato terminale (già vinto/perso/pieno), V(s) = 0
                # Nota: gestiamo i reward nelle transizioni, quindi ignoriamo l'update per i terminali puri
                if self.check_win(s, 1) or self.check_win(s, -1) or 0 not in s:
                    continue
                
                v_old = self.V[s]
                
                # Trova l'azione che massimizza il valore atteso (Bellman Optimality Equation)
                # V(s) = max_a sum [ p(s',r|s,a) * (r + gamma * V(s')) ]
                actions = self.get_valid_actions(s)
                action_values = []
                
                for a in actions:
                    expected_value = 0
                    outcomes = self.transition(s, a)
                    for prob, next_s, reward, done in outcomes:
                        # Se done, V(next_s) è 0
                        v_next = 0 if done else self.V[next_s]
                        expected_value += prob * (reward + self.gamma * v_next)
                    action_values.append(expected_value)
                
                best_value = max(action_values) if action_values else 0
                self.V[s] = best_value
                delta = max(delta, abs(v_old - self.V[s]))
            
            iteration += 1
            if delta < theta:
                print(f"Convergenza raggiunta in {iteration} iterazioni.")
                break
                
        # Estrazione della Policy Ottima (Greedy rispetto a V)
        # Slide 11: pi'(s) = argmax_a Q(s,a)
        print("Estrazione della Policy Ottima...")
        for s in self.states:
            if self.check_win(s, 1) or self.check_win(s, -1) or 0 not in s:
                continue
                
            actions = self.get_valid_actions(s)
            best_action = None
            best_val = -float('inf')
            
            for a in actions:
                val = 0
                outcomes = self.transition(s, a)
                for prob, next_s, reward, done in outcomes:
                    v_next = 0 if done else self.V[next_s]
                    val += prob * (reward + self.gamma * v_next)
                
                if val > best_val:
                    best_val = val
                    best_action = a
            
            self.policy[s] = best_action

    ## ------------------------------------------------------------------
    ## 4. TEST
    ## ------------------------------------------------------------------
    
    def play_game(self):
        state = self.empty_board
        print("\n--- Partita Dimostrativa (Agente vs Random) ---")
        self.render(state)
        
        while True:
            # Turno Agente
            if state not in self.policy:
                print("Nessuna mossa valida o stato terminale.")
                break
                
            action = self.policy[state]
            print(f"Agente sceglie pos: {action}")
            
            # Applica mossa agente
            l = list(state)
            l[action] = 1
            state = tuple(l)
            self.render(state)
            
            if self.check_win(state, 1):
                print("Agente VINCE!")
                return
            if 0 not in state:
                print("PAREGGIO!")
                return
                
            # Turno Avversario (Random)
            valid = [i for i, x in enumerate(state) if x == 0]
            opp_action = np.random.choice(valid)
            print(f"Avversario sceglie pos: {opp_action}")
            
            l = list(state)
            l[opp_action] = -1
            state = tuple(l)
            self.render(state)
            
            if self.check_win(state, -1):
                print("Avversario VINCE!")
                return
            if 0 not in state:
                print("PAREGGIO!")
                return

    def render(self, board):
        symbols = {0: '.', 1: 'X', -1: 'O'}
        arr = np.array(board).reshape(3,3)
        print("\n")
        for row in arr:
            print(" ".join([symbols[x] for x in row]))

if __name__ == "__main__":
    # Crea solver
    solver = TicTacToeDP()
    
    # Esegui Value Iteration (Policy Evaluation + Improvement combinati)
    solver.value_iteration()
    
    # Testa
    solver.play_game()