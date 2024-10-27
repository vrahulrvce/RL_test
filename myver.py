import numpy as np
import random

ALICE_ACTIONS = ['P', 'D', 'B', 'NB']  # Poison, Do nothing, Build patio, No patio
BOB_ACTIONS = ['C', 'NC']              # Call doctor, No doctor

# States
STATES = ['tree_healthy', 'tree_sick']


alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration rate


Q_Alice = np.zeros((len(STATES), len(ALICE_ACTIONS)))
Q_Bob = np.zeros((len(STATES), len(BOB_ACTIONS)))


def reward_alice(state, action_alice, action_bob):
    if action_alice == 'B':  
        if state == 'tree_sick':  
            return 10
        return 5  
    elif action_alice == 'P':  
        return -2 
    return 0  

def reward_bob(state, action_bob):
    if action_bob == 'C':  
        return -5  
    if state == 'tree_sick': 
        return -10 
    return 0  

# Transition function
def transition(state, action_alice, action_bob):
    if action_alice == 'P' and action_bob == 'NC':  
        return 'tree_sick'
    if action_bob == 'C':  
        return 'tree_healthy'
    return state


episodes = 10000
for episode in range(episodes):

    state = 'tree_healthy'
    
    for t in range(10000):  
    
        state_idx = STATES.index(state)

        if random.uniform(0, 1) < epsilon:
            action_alice = random.choice(ALICE_ACTIONS)
        else:
            action_alice = ALICE_ACTIONS[np.argmax(Q_Alice[state_idx])]
        

        if random.uniform(0, 1) < epsilon:
            action_bob = random.choice(BOB_ACTIONS)
        else:
            action_bob = BOB_ACTIONS[np.argmax(Q_Bob[state_idx])]

        reward_a = reward_alice(state, action_alice, action_bob)
        reward_b = reward_bob(state, action_bob)


        next_state = transition(state, action_alice, action_bob)
        next_state_idx = STATES.index(next_state)

   
        Q_Alice[state_idx, ALICE_ACTIONS.index(action_alice)] += alpha * (
            reward_a + gamma * np.max(Q_Alice[next_state_idx]) - Q_Alice[state_idx, ALICE_ACTIONS.index(action_alice)]
        )

        Q_Bob[state_idx, BOB_ACTIONS.index(action_bob)] += alpha * (
            reward_b + gamma * np.max(Q_Bob[next_state_idx]) - Q_Bob[state_idx, BOB_ACTIONS.index(action_bob)]
        )

     
        state = next_state

print("Q-table for Alice:")
print(Q_Alice)
print("\nQ-table for Bob:")
print(Q_Bob)


def best_strategies():
    for state in STATES:
        state_idx = STATES.index(state)
        best_action_alice = ALICE_ACTIONS[np.argmax(Q_Alice[state_idx])]
        best_action_bob = BOB_ACTIONS[np.argmax(Q_Bob[state_idx])]
        print(f"In state '{state}':")
        print(f"  Best action for Alice: {best_action_alice}")
        print(f"  Best action for Bob: {best_action_bob}\n")

best_strategies()
