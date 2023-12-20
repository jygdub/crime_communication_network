"""
Script for preliminary network of message passing until consensus in a lattice structured graph
    Simulation 

Written by Jade Dubbeld
12/12/2023
"""

import timeit, numpy as np, random
import matplotlib.pyplot as plt
import networkx as nx

from lattice import init_lattice, visualize, message

# initialize network
G = init_lattice((2,2))

attributes = nx.get_node_attributes(G, "state")

M = 0

start = timeit.default_timer()

# converge when all nodes agree on state
while np.unique(list(attributes.values())).size > 1:
    source = random.choice(list(G.nodes))
    destination = random.choice(list(G.neighbors(source)))
    print(f"{source} -> {destination}")

    G = message(G=G,source=source,destination=destination,alpha=1.0,beta=0.0)

    print(G.nodes(data=True))

    M += 1
    attributes = nx.get_node_attributes(G, "state")

stop = timeit.default_timer()
execution_time = stop - start

print(f"execution time = {execution_time} in seconds")
print(f"total messages = {M}")
