"""
Script (speedup) to simulate Deffuant-like dynamics on LFR benchmark graphs.

Written by Jade Dubbeld
17/02/2024
"""

import networkx as nx, random, numpy as np, matplotlib.pyplot as plt, pickle

from LFR_network import init, hamming_distance, simulate

exp_degree = 3.0
exp_community = 1.5

alpha = 1.0
beta = 0.0

# listFileNames = sorted(glob.glob(f'graphs/first-generation/tau1={exp_degree}-tau2={exp_community}-*.pickle'))

# filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'
# filename = 'graphs/test10-tau1=3.0-tau2=1.5-mu=0.3-avg_deg=2-min_comm=2-seed=0.pickle'
filename = 'graphs/first-generation/tau1=3.0-tau2=1.5-mu=0.25-avg_deg=10-min_comm=10-seed=12.pickle'


G_init = pickle.load(open(filename, 'rb'))
G = init(G_init)

print(filename)

M, meanStringDifference = simulate(G,alpha=alpha,beta=beta)
print(M)
