"""
Script (speedup) to simulate Deffuant-like dynamics on LFR benchmark graphs.

Written by Jade Dubbeld
17/02/2024
"""

import networkx as nx, random, numpy as np, matplotlib.pyplot as plt, pickle, glob, re, time
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from tqdm import tqdm
from datetime import datetime

from LFR_network import init, hamming_distance, simulate

def message(G, source, destination, attributes, alpha=1.0, beta=0.0):
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - G: current state of networkx graph
    - source: selected sender node in network
    - destination: selected receiver node in network
    - alpha: probability of sender bias (sending match or mismatch bits)
    - beta: probability of receiver bias (flipping message or not)

    Returns:
    - G: new state of networkx graph

    """

    match = []

    # find correct bits in source node's state compared to destination node's state
    for position, bit in enumerate(G.nodes[source]['state']):
        if bit == G.nodes[destination]['state'][position]:
            match.append(position)

    mismatch = [i for i in [0,1,2] if i not in match]

    # generate random float representing matching/sender bias
    P_Matching = random.random()

    # random pick if no bits in common
    # pick from correct list with probability alpha (incorrect with probability 1 - alpha)
    if P_Matching > alpha and match != []:
        index = random.choice(match)
    elif P_Matching <= alpha and mismatch != []:
        index = random.choice(mismatch)
    else:
        index = random.choice([0,1,2])

    # generate message
    message = G.nodes[source]['state'][index]

    # print(f"index = {index}, message = {message}")

    # generate random float representing miscommunication/receiver bias
    P_Miss = random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == '0':
            message = '1'
        elif message == '1':
            message = '0'

    # get current state of selected downstream neighbor
    current_state = G.nodes[destination]['state']

    # copy received bit at given position (redundant if bit is already agreed)
    new_state = current_state[:index] + message + current_state[index + 1:]
    G.nodes[destination]['state'] = new_state
    attributes[destination] = new_state

    return G, attributes

def simulate(G, alpha=1.0, beta=0.0):
    """
    Function to run a simulation for n_iters on a lattice.

    Parameters:
    - G: generated networkx graph
    - alpha: probability of sender bias (sending match or mismatch bits)
    - beta: probability of receiver bias (flipping message or not)

    Returns:
    - total_messages: list of total messages sent in all simulations
    - all_diffScores: list of all string difference scores in all simulations
    """

    N = len(G.nodes())
    meanStringDifference = []
    stringDifference = np.zeros((N,N))
    M = 0
    attributes = np.array(list(nx.get_node_attributes(G, "state").values()))

    # compute hamming distance for initial configuration
    for index1, node1 in enumerate(G.nodes()):
        for index2, node2 in enumerate(G.nodes()):
            if node1 >= node2:
                continue
                
            hammingDistance = hamming_distance(attributes[node1],attributes[node2])

            # fill in normalized hamming distance array
            stringDifference[index1,index2] = hammingDistance / len(attributes[node1])
            stringDifference[index2,index1] = hammingDistance / len(attributes[node1])


    # print(stringDifference)

    # converge when all nodes agree on state
    while (np.unique(attributes).size > 1):

        source = random.choice(list(G.nodes))
        destination = random.choice(list(G.neighbors(source)))
        # print(f"{source} -> {destination}")
        # print(attributes[source], attributes[destination])

        G,attributes = message(G=G,source=source,destination=destination,attributes=attributes,alpha=alpha,beta=beta)

        # print(G.nodes(data=True))

        M += 1
        # attributes = nx.get_node_attributes(G, "state")   

        if M % 1000 == 0:
            print(M)

        # re-calculate normalized hamming distance for all pair combinations for node update
        for index, node in enumerate(G.nodes()): 
            if destination == node:
                continue

            hammingDistance = hamming_distance(attributes[destination],attributes[node])
            # print(hammingDistance)

            # fill in normalized hamming distance array
            stringDifference[node,destination] = hammingDistance / len(attributes[node])
            stringDifference[destination,node] = hammingDistance / len(attributes[node])

        meanStringDifference.append(np.mean(stringDifference))

        # print(stringDifference)
    
    return M, meanStringDifference


exp_degree = 3.0
exp_community = 1.5

alpha = 1.0
beta = 0.0

# listFileNames = sorted(glob.glob(f'graphs/first-generation/tau1={exp_degree}-tau2={exp_community}-*.pickle'))

filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'

G_init = pickle.load(open(filename, 'rb'))
G = init(G_init)

print(filename)

simulate(G,alpha=alpha,beta=beta)
