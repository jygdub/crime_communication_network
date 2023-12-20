"""
Script for preliminary network of message passing until consensus in a lattice structured graph
    Simulation 

Written by Jade Dubbeld
12/12/2023
"""

import timeit, numpy as np, random
import matplotlib.pyplot as plt
import networkx as nx

from lattice import init_lattice, simulate

# initialize network
G = init_lattice((5,5))
M = 0
alpha = 1.0
beta = 0.0

start = timeit.default_timer()

# simulate until consensus in network
G, M = simulate(G,M,alpha,beta)

stop = timeit.default_timer()
execution_time = stop - start

print(f"execution time = {execution_time} in seconds")
print(f"total messages = {M}")