import numpy as np
import random

class TicTacToeEnv:

    def __init__(self):
        self.board = np.zeros(9, dtype=int)