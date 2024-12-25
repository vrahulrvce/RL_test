import gym
from gym import spaces
import numpy as np
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from deepQ import DQN
from ray.tune.logger import pretty_print
import matplotlib.pyplot as plt
from ray.tune.registry import register_env
class BayesianGymEnv(MultiAgentEnv):
    def __init__(self, bayesian_structure, cpd_tables, agent_goals):
        super(BayesianGymEnv, self).__init__()
        self.structure = bayesian_structure
        self.cpd_tables = cpd_tables
        self.agent_goals = agent_goals
        self.agents = ["agent1", "agent2"]
        self.observation_space = spaces.Dict({
            node: spaces.Discrete(2) for node in bayesian_structure.keys()
        })
        self.action_space = spaces.Discrete(2)

    def reset(self):
        self.state = {node: np.random.choice([0, 1]) for node in self.structure.keys()}
        return {agent: self.state for agent in self.agents}

    def step(self, action_dict):
        rewards = {}
        dones = {}
        next_states = {}

        for agent, action in action_dict.items():
            goal = self.agent_goals[self.agents.index(agent)]
            if action == 0:
                ts = np.random.choice([0, 1], p=self._calculate_prob("ts", ["pt"]))
                bp = np.random.choice([0, 1], p=self._calculate_prob("bp", ["pt", "td"]))
                self.state.update({"ts": ts, "bp": bp})
            elif action == 1:
                tde = np.random.choice([0, 1], p=self._calculate_prob("tde", ["ts", "td"]))
                self.state.update({"tde": tde})

            dones[agent] = self.state[goal] == 1
            rewards[agent] = 1 if dones[agent] else -0.01 * sum(self.state.values())
            next_states[agent] = self.state

        # Check if all agents are done
        dones["__all__"] = all(dones.values())
        return next_states, rewards, dones, {}

    def _calculate_prob(self, node, parents):
        index = sum(self.state[parent] * (2 ** i) for i, parent in enumerate(parents))
        probabilities = self.cpd_tables[node][:, index]
        return probabilities
ray.init()
bayesian_structure = {
    "pt": ["ts", "bp"],
    "ts": ["td", "tde"],
    "td": ["bp", "tde"],
    "bp": ["V"],
    "tde": ["T"]
}


cpd_tables = {
    "ts": np.array([[0.8, 0.3], [0.2, 0.7]]),
    "bp": np.array([[0.7, 0.4, 0.5, 0.2], [0.3, 0.6, 0.5, 0.8]]),
    "tde": np.array([[0.9, 0.5, 0.4, 0.2], [0.1, 0.5, 0.6, 0.8]]),
}

agent_goals = ["bp", "tde"]

def env_creator(env_config):
    return BayesianGymEnv(bayesian_structure, cpd_tables, agent_goals)

register_env("BayesianGym-v0", env_creator)
config = {
    "env": "BayesianGym-v0",
    "multiagent": {
        "policies": {
            "policy_agent1": (None, gym.spaces.Dict({
                node: spaces.Discrete(2) for node in bayesian_structure.keys()
            }), spaces.Discrete(2), {}),
            "policy_agent2": (None, gym.spaces.Dict({
                node: spaces.Discrete(2) for node in bayesian_structure.keys()
            }), spaces.Discrete(2), {}),
        },
        "policy_mapping_fn": lambda agent_id: "policy_agent1" if agent_id == "agent1" else "policy_agent2",
    },
    "framework": "torch",
    "num_workers": 1,
    "gamma": 0.99,
    "lr": 1e-3,
    "train_batch_size": 32,
    "log_level": "ERROR"
}

trainer = DQN(config=config)
smoothed_rewards1 = []
smoothed_rewards2 = []
correlations = []

episodes = 2000
reward_window1 = []
reward_window2 = []

for episode in range(episodes):
    result = trainer.train()
    agent_rewards = result["policy_reward_mean"]
    reward1 = agent_rewards.get("policy_agent1", 0)
    reward2 = agent_rewards.get("policy_agent2", 0)

    reward_window1.append(reward1)
    reward_window2.append(reward2)

    if len(reward_window1) > 100:
        smoothed_rewards1.append(np.mean(reward_window1[-100:]))
        smoothed_rewards2.append(np.mean(reward_window2[-100:]))
        correlation = np.corrcoef(smoothed_rewards1, smoothed_rewards2)[0, 1]
        correlations.append(correlation)
    else:
        correlations.append(0)

    if episode % 100 == 0:
        print(f"Episode {episode}: Agent1 Avg Reward: {reward1}, Agent2 Avg Reward: {reward2}")
        if len(correlations) > 1:
            print(f"Correlation between agents: {correlations[-1]:.4f}")

ray.shutdown()
