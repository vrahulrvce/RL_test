from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from graphviz import Source
import random

random_model = BayesianNetwork.get_random(n_nodes=8, edge_prob=0.4, n_states=2, latents=True)
"""
poision tree  = pt
Tree doctor = td
build patio = bp
tree sick = ts
tree dead = tde
tree = T
effort = E
view = V
cost = C
"""

Maid_model = BayesianNetwork( ### building network
    [
        ("pt","ts"),    ### PT ----> TS
        ("pt","bp"),    ### PT ----> BP
        ("pt","E"),     ### PT ----> E
        ("ts","td"),    ### TS ----> TD
        ("ts","tde"),   ### TS ----> TDE
        ("td","bp"),    ### TD ----> BP
        ("td","C"),     ### TD ----> C
        ("td","tde"),   ### TD ----> TDE
        ("bp","V"),     ### BP ----> V
        ("tde","T")     ### TDE ----> T
    ]
)
## Defining CPDS to all the utility node , nature node and decision nodes
tab_pt = TabularCPD(variable='pt', variable_card=2 , values=[[0.5],[0.5]]) #### considering PT as parent node and since this node starts all the process
tab_ts = TabularCPD(variable='ts', variable_card=2,
                    values=[[0.8,0.3],
                    [0.2,0.7]],
                    evidence=['pt'],evidence_card=[2])
tab_bp = TabularCPD(variable='bp', variable_card=2,
                    values=[[0.7, 0.4, 0.5, 0.2],  # P(BP=0 | PT, TD)
                            [0.3, 0.6, 0.5, 0.8]],  # P(BP=1 | PT, TD)
                    evidence=['pt', 'td'], evidence_card=[2, 2])
tab_e = TabularCPD(variable='E', variable_card=2,
                   values=[[0.6, 0.3],  # P(E=0 | PT=0), P(E=0 | PT=1)
                           [0.4, 0.7]], # P(E=1 | PT=0), P(E=1 | PT=1)
                   evidence=['pt'], evidence_card=[2])
tab_td = TabularCPD(variable='td', variable_card=2,
                    values=[[0.7, 0.2],  # P(TD=0 | TS=0), P(TD=0 | TS=1)
                            [0.3, 0.8]], # P(TD=1 | TS=0), P(TD=1 | TS=1)
                    evidence=['ts'], evidence_card=[2])
tab_tde = TabularCPD(variable='tde', variable_card=2,
                     values=[[0.9, 0.5, 0.4, 0.2],  # P(TDE=0 | TS, TD)
                             [0.1, 0.5, 0.6, 0.8]],  # P(TDE=1 | TS, TD)
                     evidence=['ts', 'td'], evidence_card=[2, 2])
tab_c = TabularCPD(variable='C', variable_card=2,
                   values=[[0.8, 0.4],  # P(C=0 | TD=0), P(C=0 | TD=1)
                           [0.2, 0.6]], # P(C=1 | TD=0), P(C=1 | TD=1)
                   evidence=['td'], evidence_card=[2])
tab_v = TabularCPD(variable='V', variable_card=2,
                   values=[[0.7, 0.4],  # P(V=0 | BP=0), P(V=0 | BP=1)
                           [0.3, 0.6]], # P(V=1 | BP=0), P(V=1 | BP=1)
                   evidence=['bp'], evidence_card=[2])
tab_t = TabularCPD(variable='T', variable_card=2,
                   values=[[0.6, 0.3],  # P(T=0 | TDE=0), P(T=0 | TDE=1)
                           [0.4, 0.7]], # P(T=1 | TDE=0), P(T=1 | TDE=1)
                   evidence=['tde'], evidence_card=[2])
Maid_model.add_cpds(tab_pt,tab_ts,tab_td,tab_tde,tab_bp,tab_e,tab_v,tab_t,tab_c)
assert Maid_model.check_model()
class Agents:
    def __init__(self,name,goal_node,state):
        self.name = name
        self.goal_node = goal_node
        self.state = state
    
    def act(self,action):
        if action == "call_td":
            Maid_model.get_cpds("td").values = [[0.9],[0.1]]
        elif action == "posion_tree":
            Maid_model.get_cpds("pt").values = [[0.1],[0.9]]
        elif action == "build_patio":
            Maid_model.get_cpds("bp").values = [[0.3],[0.7]]
        
        
        if self.goal_node == "T" and action == "call_td":
            return 1
        elif self.goal_node == "V" and action == "build_patio":
            return 1
        
        return 0
agent1_pois = Agents(name="agent1", goal_node="V", state=1)
agent2_save = Agents(name="agent2", goal_node="T", state=1)

actions = ["call_td", "postion_tree","build_patio"]
rewards = {"agent1":0 , "agent2":0}

for i in range(10000000):
    agent1_action = random.choice(actions)
    agent2_action = random.choice(actions)

    reward_agent1 = agent1_pois.act(agent1_action)
    rewards["agent1"] += reward_agent1
    print(f"agent1 performs action : {agent1_action}, Reward:{reward_agent1}")

    reward_agent2 = agent2_save.act(agent2_action)
    rewards["agent2"] += reward_agent2
    print(f"agent2 performs  action:{agent2_action}, rewards = {reward_agent2}")

print("final rewards")
print(f"agent1 : {rewards['agent1']}")
print(f"agent2 : {rewards['agent2']}")
