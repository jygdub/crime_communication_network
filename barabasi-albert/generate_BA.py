"""
Script to randomly generate Barabasi-Albert networks and compute network measures
    Save graphs and network measures

Written by Jade Dubbeld
18/01/2024
"""

from BA_network import generate, init_BA, visualize

import networkx as nx, numpy as np, pickle

n = 100
m = 4

for i in range(1,51):
    G = generate(n,m)

    pickle.dump(G, open(f'graphs/m={m}/graph{i}-n={n}-m={m}.pickle', 'wb')) 
        # load using G = pickle.load(open(f'graphs/test.pickle', 'rb'))



