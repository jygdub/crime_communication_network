"""
Script to randomly generate Barabasi-Albert networks and compute network measures
    Save graphs and network measures

Written by Jade Dubbeld
18/01/2024
"""

from BA_network import generate

import networkx as nx, numpy as np, pickle
from tqdm import tqdm

n = 1000

for m in range(1,5):
    for i in tqdm(range(1,1001)):
        G = generate(n,m)

        pickle.dump(G, open(f'graphs/n={n}/m={m}/graph{i}-n={n}-m={m}.pickle', 'wb')) 
            # load using G = pickle.load(open(f'graphs/test.pickle', 'rb'))



