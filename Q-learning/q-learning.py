import numpy as np
from random import random, choice, randint

# --> ENVIRONMENT
NUM_ROW = 4
NUM_COL = 3

ENV = [(0, 0, 1), 
       (0, -1, 0), 
       (0, 0, 0), 
       (-1, 0, 0)] 

ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)] # su, giu, sx, dx
NUM_ACTIONS = len(ACTIONS)

# --> IPERPARAMETRI Q-LEARNING
ALPHA = 0.1    # Learning Rate
GAMMA = 0.9    # Discount Factor
EPSILON = 0.1  # Exploration Rate

def step(state, action_ixd):
    pass

def q_learning():
    pass

def main():
    pass

main()