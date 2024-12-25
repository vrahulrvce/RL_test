# RL for MAID application

netplot.py is implemented using bayesian networks

Uses Pgmpy to create a bayesian network and also to calculate the CPD's of the table

Installation : 

```bash
pip install pgmpy
```
Package Link : 
```bash
https://github.com/pgmpy/pgmpy
```
Package citation 

```bash
Ankur Ankan, & Johannes Textor (2024). pgmpy: A Python Toolkit for Bayesian Networks. Journal of Machine Learning Research, 25(265), 1–8.
```
Example of Bayesian network :

<a href="https://pgmpy.org/examples/Creating%20a%20Discrete%20Bayesian%20Network.html"> Creation of Bayesian Network</a>

<a href="https://pgmpy.org/models/bayesiannetwork.html">Bayesian Network</a>

<a href="https://pgmpy.org/models/dbn.html"> Dynamic Bayesian Network</a>

The Main.py contains the RL which uses only the torch library the reward is set to 1 and the negative is set to -0.9 (No of Epi = 2000)
```bash
Agent 1 Action V : 0, Agent 2 Action T: 0, Agent 1 Total Reward: 126.59999999999712, Agent 2 Total Reward: 111.39999999999803
Agent 1 Action V : 0, Agent 2 Action T: 1, Agent 1 Total Reward: 63.89999999999987, Agent 2 Total Reward: 63.89999999999879
Agent 1 Action V : 1, Agent 2 Action T: 0, Agent 1 Total Reward: 155.09999999999897, Agent 2 Total Reward: 189.29999999999663
Agent 1 Action V : 1, Agent 2 Action T: 1, Agent 1 Total Reward: 103.79999999999755, Agent 2 Total Reward: 77.19999999999834
Episode 1999: Agent 1 Q-values: [[0.00792653 0.07058309]]
Episode 1999: Agent 2 Q-values: [[0.08477707 0.07061243]]
Total Rewards for Agent 1: 391.3000000000077
Total Rewards for Agent 2: 516.7000000000338
```

The Main1.py contains the RL which uses the GYM and torch library Gym for creating environmental spaces ( which translates to bayesian network in binary) the reward is set to 1 and the negative is set to -0.9 (No of Epi = 2000)
```bash
Agent 1 Action V : 0, Agent 2 Action T: 0, Agent 1 Total Reward: 536.100000000037, Agent 2 Total Reward: 378.4000000000221
Agent 1 Action V : 0, Agent 2 Action T: 1, Agent 1 Total Reward: 545.6000000000504, Agent 2 Total Reward: 581.7000000000681
Agent 1 Action V : 1, Agent 2 Action T: 0, Agent 1 Total Reward: 513.3000000000408, Agent 2 Total Reward: 538.0000000000597
Agent 1 Action V : 1, Agent 2 Action T: 1, Agent 1 Total Reward: 610.2000000000288, Agent 2 Total Reward: 574.1000000000603
Episode 9999: Agent 1 Q-values: [[0.1675436  0.01277325]]
Episode 9999: Agent 2 Q-values: [[0.11744626 0.077337  ]]
Total Rewards for Agent 1: 2092.599999999244
Total Rewards for Agent 2: 1995.699999999229
```
The ray_test.py is still being built, since ray has phased out from agents to algorithms most of the libraries are not working .. (But extracted the deepQ agents from the initial source and modified them to the latest one but  **CONFLICT FOR NUMPY HENCE USE NUMPY <2.0 Modified the package to fit for NUMPY 1.6.3.0** ) 

