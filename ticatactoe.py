import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces

## ------------------------------------------------------------------
## SECTION 1: TIC TAC TOE ENVIRONMENT
## ------------------------------------------------------------------

class TicTacToeEnv(gym.Env):
    """
    Agent is Player 1 ('1' or 'X')
    Vs Player 2 ('-1' o 'O')
    
    Player 2 has random moves
    """
    def __init__(self):
        super(TicTacToeEnv, self).__init__()
        
        # Define space of actions (up to 9 moves)
        self.action_space = spaces.Discrete(9)
        
        # Define space of states
        self.observation_space = spaces.Box(low=-1, high=1, shape=(9,), dtype=np.int8)
        
        # Inizialize 
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1 

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = 1
        
        return self._get_obs(), {}

    def step(self, action):
        """
        Takes the action
        """
        
        # Check for validity
        if self.board[action] != 0:
            return self._get_obs(), -10, False, False, {} # obs, reward, terminated, truncated, info

        # Player 1 move
        self.board[action] = self.current_player
        
        # Check if agent won
        if self._check_win(self.current_player):
            reward = 1.0
            terminated = True
            return self._get_obs(), reward, terminated, False, {}

        # Controlla for draw
        if np.all(self.board != 0):
            reward = 0.5
            terminated = True
            return self._get_obs(), reward, terminated, False, {}

        # Player 2 move (random)
        self.current_player = -1
        valid_moves = np.where(self.board == 0)[0]
        
        if len(valid_moves) > 0:
            opponent_action = random.choice(valid_moves)
            self.board[opponent_action] = self.current_player
            
            # Check if player 2 won
            if self._check_win(self.current_player):
                reward = 0.0
                terminated = True
                return self._get_obs(), reward, terminated, False, {}

            # Check again for draw
            if np.all(self.board != 0):
                reward = 0.5
                terminated = True
                return self._get_obs(), reward, terminated, False, {}

        # Game continues, agent move
        self.current_player = 1
        reward = 0.0
        terminated = False
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """
        return status of the grid
        """
        return tuple(self.board)

    def _check_win(self, player):
        """
        check if player x won
        """
        board_2d = self.board.reshape((3, 3))

        # Check rows and columns
        for i in range(3):
            if np.all(board_2d[i, :] == player) or np.all(board_2d[:, i] == player):
                return True
        
        # Check diagonals
        if np.all(np.diag(board_2d) == player) or np.all(np.diag(np.fliplr(board_2d)) == player):
            return True
            
        return False

    def get_valid_moves(self):
        """
        Helper to obtain valid oves
        """
        return np.where(self.board == 0)[0]

    def render(self):
        """
        print grid
        """
        board_2d = self.board.reshape((3, 3))
        symbols = {1: 'X', -1: 'O', 0: '.'}
        
        print("\nGrid:") # Il tuo commento aggiornato
        for row in board_2d:
            print(" ".join([symbols[cell] for cell in row]))

## ------------------------------------------------------------------
## SECTION 2: AGENT'S POLICY FUNCTIONS
## ------------------------------------------------------------------

def get_value(state, v_table):
    return v_table.get(state, 0.5)

def choose_action(state, valid_moves, v_table, epsilon):
    if random.random() < epsilon:
        # Random valid move
        return random.choice(valid_moves)
    else:
        # Greedy action: choose best move
        best_value = -float('inf')
        best_action = None
        
        for action in valid_moves:
            # Simulate move and observe status (using temp status)
            temp_board = list(state)
            temp_board[action] = 1 # Agent move
            
            # Status right after agent move
            next_state = tuple(temp_board)
            
            state_value = get_value(next_state, v_table) # (Usa la v_table passata)
            
            if state_value > best_value:
                best_value = state_value
                best_action = action
                
        return best_action

## ------------------------------------------------------------------
## SECTION 3: TRAINING FUNCTION
## ------------------------------------------------------------------

def train_agent(env, num_episodes, alpha, epsilon_start, epsilon_decay, epsilon_min):
    """
    Trains the agent using TD(0) to learn the V-table.
    """
    print("Start training...") 
    
    # value table
    v_table = {} 
    epsilon = epsilon_start

    for episode in range(num_episodes):
        
        # Reset for new match
        state, info = env.reset()
        
        terminated = False
        last_state = state 
        while not terminated:
            valid_moves = env.get_valid_moves()
            action = choose_action(state, valid_moves, v_table, epsilon)
            
            # Take action in the environment
            # Player 2 move will be also managed
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # --- Update V function ---
            # V(s_t) <- V(s_t) + alpha * [V(s_t+1) - V(s_t)] 
            
            v_s_t = get_value(last_state, v_table)
            if terminated:
                v_s_t_plus_1 = reward
            else:
                v_s_t_plus_1 = get_value(next_state, v_table)

            # Apply formula
            v_table[last_state] = v_s_t + alpha * (v_s_t_plus_1 - v_s_t)
            
            # Update state for next cicle
            last_state = next_state
            state = next_state # including player 2 move

        # Reduce exploration
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (episode + 1) % 5000 == 0:
            print(f"Episode {episode + 1}/{num_episodes} completed. Epsilon: {epsilon:.4f}")

    print("\nTraining done!") 
    print(f"V-table has been trainend in {len(v_table)} states.")
    return v_table

## ------------------------------------------------------------------
## SECTION 4: TEST FUNCTION
## ------------------------------------------------------------------

def test_agent(env, v_table, num_test_games):
    """
    Executes test games between the trained agent and Player 2.
    """
    
    print("\nStart test: Agent (X) plays against Player 2 (O).")

    # Only greedy moves
    epsilon_test = 0.0 

    wins = 0
    losses = 0
    draws = 0

    for _ in range(num_test_games):
        state, info = env.reset()
        terminated = False
        
        print("\n--- New  match ---")
        env.render()
        
        while not terminated:
            # Agent chooses best move
            valid_moves = env.get_valid_moves()
            action = choose_action(state, valid_moves, v_table, epsilon_test)
            
            print(f"\nAgent chooses the move: {action}") 
            
            # Execute move (including player 2 move)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            env.render()
            
            if terminated:
                print("")
                if reward == 1.0:
                    print("--> Agent won (X)!") 
                    wins += 1
                elif reward == 0.0:
                    print("--> Player 2 (O) won!") 
                    losses += 1
                else: 
                    print("--> Draw!")
                    draws += 1
            
            state = next_state

    print("\n--- Results ---") 
    print(f"Number of matches: {num_test_games}")
    print(f"Wins (X): {wins}")
    print(f"Defeats (X): {losses}")
    print(f"Draws: {draws}")


## ------------------------------------------------------------------
## SECTION 5: MAIN SCRIPT
## ------------------------------------------------------------------

if __name__ == "__main__":
    
    # 1. Defines training parameters
    PARAM_NUM_EPISODES = 50000
    PARAM_ALPHA = 0.1
    PARAM_EPSILON_START = 1.0
    PARAM_EPSILON_DECAY = 0.9999
    PARAM_EPSILON_MIN = 0.01
    PARAM_TEST_GAMES = 10
    
    # 2. Create the environment
    ambiente_gioco = TicTacToeEnv()
    
    # 3. Start training
    tabella_valori_addestrata = train_agent(
        ambiente_gioco, 
        PARAM_NUM_EPISODES, 
        PARAM_ALPHA, 
        PARAM_EPSILON_START, 
        PARAM_EPSILON_DECAY, 
        PARAM_EPSILON_MIN
    )
    
    # 4. Start test
    test_agent(ambiente_gioco, tabella_valori_addestrata, PARAM_TEST_GAMES)