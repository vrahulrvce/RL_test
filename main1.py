import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
from gym import spaces
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
bayesian_structure = [
    ("pt", "ts"),  # PT --> TS
    ("pt", "bp"),  # PT --> BP
    ("pt", "E"),   # PT --> E
    ("ts", "td"),  # TS --> TD
    ("ts", "tde"),  # TS --> TDE
    ("td", "bp"),  # TD --> BP
    ("td", "C"),   # TD --> C
    ("td", "tde"),  # TD --> TDE
    ("bp", "V"),   # BP --> V
    ("tde", "T")   # TDE --> T
]
bn = BayesianNetwork(bayesian_structure)
cpd_pt = TabularCPD(variable='pt', variable_card=2 , values=[[0.5],[0.5]]) #### considering PT as parent node and since this node starts all the process
cpd_ts = TabularCPD(variable='ts', variable_card=2,
                    values=[[0.8,0.3],
                    [0.2,0.7]],
                    evidence=['pt'],evidence_card=[2])
cpd_bp = TabularCPD(variable='bp', variable_card=2,
                    values=[[0.7, 0.4, 0.5, 0.2],  # P(BP=0 | PT, TD)
                            [0.3, 0.6, 0.5, 0.8]],  # P(BP=1 | PT, TD)
                    evidence=['pt', 'td'], evidence_card=[2, 2])
cpd_e = TabularCPD(variable='E', variable_card=2,
                   values=[[0.6, 0.3],  # P(E=0 | PT=0), P(E=0 | PT=1)
                           [0.4, 0.7]], # P(E=1 | PT=0), P(E=1 | PT=1)
                   evidence=['pt'], evidence_card=[2])
cpd_td = TabularCPD(variable='td', variable_card=2,
                    values=[[0.7, 0.2],  # P(TD=0 | TS=0), P(TD=0 | TS=1)
                            [0.3, 0.8]], # P(TD=1 | TS=0), P(TD=1 | TS=1)
                    evidence=['ts'], evidence_card=[2])
cpd_tde = TabularCPD(variable='tde', variable_card=2,
                     values=[[0.9, 0.5, 0.4, 0.2],  # P(TDE=0 | TS, TD)
                             [0.1, 0.5, 0.6, 0.8]],  # P(TDE=1 | TS, TD)
                     evidence=['ts', 'td'], evidence_card=[2, 2])
cpd_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.8, 0.4],  # P(C=0 | TD=0), P(C=0 | TD=1)
                           [0.2, 0.6]], # P(C=1 | TD=0), P(C=1 | TD=1)
                   evidence=['td'], evidence_card=[2])
cpd_v = TabularCPD(variable='V', variable_card=2,
                   values=[[0.7, 0.4],  # P(V=0 | BP=0), P(V=0 | BP=1)
                           [0.3, 0.6]], # P(V=1 | BP=0), P(V=1 | BP=1)
                   evidence=['bp'], evidence_card=[2])
cpd_t = TabularCPD(variable='T', variable_card=2,
                   values=[[0.6, 0.3],  # P(T=0 | TDE=0), P(T=0 | TDE=1)
                           [0.4, 0.7]], # P(T=1 | TDE=0), P(T=1 | TDE=1)
                   evidence=['tde'], evidence_card=[2])

bn.add_cpds(cpd_pt, cpd_ts, cpd_bp, cpd_tde, cpd_td, cpd_v, cpd_t, cpd_e, cpd_c)
assert bn.check_model()

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
class BayesianNetworkEnv(gym.Env):
    def __init__(self):
        super(BayesianNetworkEnv, self).__init__()

        self.state_size = len(bn.nodes())  
        self.action_space = spaces.Discrete(2)  
        self.observation_space = spaces.MultiBinary(self.state_size)

        self.state = np.random.randint(2, size=self.state_size)

    def reset(self):
        self.state = np.random.randint(2, size=self.state_size)
        return self.state

    def step(self, actions):
        action1, action2 = actions
        next_state = np.random.randint(2, size=self.state_size)
        
        reward1 = 1 if next_state[list(bn.nodes()).index("V")] == 1 else -0.9
        reward2 = 1 if next_state[list(bn.nodes()).index("T")] == 1 else -0.9
        
        done = next_state[list(bn.nodes()).index("V")] == 1 and next_state[list(bn.nodes()).index("T")] == 1

        self.state = next_state
        return next_state, [reward1, reward2], done, {}
env = BayesianNetworkEnv()
state_size = env.state_size
action_size = 2  
agent1_q_network = QNetwork(state_size, action_size)
agent2_q_network = QNetwork(state_size, action_size)
optimizer1 = optim.Adam(agent1_q_network.parameters(), lr=1e-3)
optimizer2 = optim.Adam(agent2_q_network.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def select_action(q_network, state, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(action_size)
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    q_values = q_network(state_tensor)
    return torch.argmax(q_values).item()

def update_q_network(q_network, optimizer, state, action, reward, next_state, done, gamma=0.99):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
    target = q_network(state_tensor).clone().detach()
    with torch.no_grad():
        if done:
            target[0][action] = reward
        else:
            next_q_values = q_network(next_state_tensor)
            max_next_q = torch.max(next_q_values)
            target[0][action] = reward + gamma * max_next_q
    q_values = q_network(state_tensor)
    loss = loss_fn(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

episodes = 2000
for episode in range(episodes):
    state = env.reset()
    done1, done2 = False, False
    while not (done1 and done2):
        action1 = select_action(agent1_q_network, state)
        action2 = select_action(agent2_q_network, state)
        
        next_state, rewards, done, _ = env.step([action1, action2])

        reward1, reward2 = rewards
        done1 = next_state[list(bn.nodes()).index("V")] == 1
        done2 = next_state[list(bn.nodes()).index("T")] == 1
        
        update_q_network(agent1_q_network, optimizer1, state, action1, reward1, next_state, done1)
        update_q_network(agent2_q_network, optimizer2, state, action2, reward2, next_state, done2)
        
        state = next_state

def evaluate_nash_equilibrium(agent1_q_network, agent2_q_network, state):
    action_combinations = list(itertools.product(range(action_size), repeat=2))
    nash_equilibria = []
    for action1, action2 in action_combinations:
        next_state, rewards, _, _ = env.step([action1, action2])
        reward1, reward2 = rewards

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        agent1_q_values = agent1_q_network(state_tensor).detach().numpy()
        agent2_q_values = agent2_q_network(state_tensor).detach().numpy()

        agent1_best_action = np.argmax(agent1_q_values)
        agent2_best_action = np.argmax(agent2_q_values)

        if action1 == agent1_best_action and action2 == agent2_best_action:
            nash_equilibria.append({
                "agent1_action": action1,
                "agent2_action": action2,
                "reward1": reward1,
                "reward2": reward2
            })

    return nash_equilibria

nash_equilibria = evaluate_nash_equilibrium(agent1_q_network, agent2_q_network, state)
for eq in nash_equilibria:
    print(f"Agent 1 Action: {eq['agent1_action']}, Agent 2 Action: {eq['agent2_action']}, "
          f"Reward 1: {eq['reward1']}, Reward 2: {eq['reward2']}")
