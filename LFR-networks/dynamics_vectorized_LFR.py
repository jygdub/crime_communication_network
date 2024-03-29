"""
Script containing all required functions to simulate Deffuant-like dynamics on networks.
- Vectorized hamming distance computation

Written by Jade Dubbeld (with contribution of Casper van Elteren)
13/03/2024
"""

import numpy as np, pandas as pd, networkx as nx, pickle, time

from multiprocessing.pool import Pool
from multiprocessing import cpu_count, Manager, Process
from itertools import product
from typing import Tuple
from functools import partial


def init(n: int, nbits: int) -> np.ndarray:
    """
    Function to initialize agents in a network with a random bit string.

    Parameters:
    - n (int): number of agents in network
    - nbits (int): length of bit string

    Returns:
    - _ (np.ndarray): array containing states of all agents
    """

    return np.random.randint(0, 2, size=(n, nbits))

def message_update(states: np.ndarray, source: int, destination: int, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - states (np.ndarray): states of all nodes in network
    - source (int): selected sender node in network
    - destination (int): selected receiver node in network
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - states (np.ndarray): updated states of all nodes in network
    """

    match = []

    # find correct bits in source node's state compared to destination node's state
    for position, bit in enumerate(states[source]):
        if bit == states[destination][position]:
            match.append(position)

    mismatch = [i for i in [0,1,2] if i not in match]

    # generate random float representing matching/sender bias
    P_Matching = np.random.random()

    # random pick if no bits in common
    # pick from correct list with probability alpha (incorrect with probability 1 - alpha)
    if P_Matching > alpha and match != []:
        index = np.random.choice(match)
    elif P_Matching <= alpha and mismatch != []:
        index = np.random.choice(mismatch)
    else:
        index = np.random.choice([0,1,2])

    # generate message
    message = states[source][index]

    # generate random float representing miscommunication/receiver bias
    P_Miss = np.random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == 0:
            message = 1
        elif message == 1:
            message = 0

    # update downstream neighbors' state according to message (redundant if bit is already agreed)
    states[destination][index] = message

    return states

def hamming_vector(states: np.ndarray, agents: slice) -> np.ndarray:
    """
    Function to compute Hamming distance in vectorized fashion.

    Parameters:
    - states (np.ndarray): states of all nodes in network
    - agents (slice): slice of agents to compare in Hamming distance calculation

    Returns:
    - _ (np.ndarray): Average pairwise Hamming distances
    """

    return (states[agents,np.newaxis] != states[np.newaxis,:]).mean(-1)


def simulate(G: nx.classes.graph.Graph, states: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> int | list:
    """
    Function to run simulation of communication dynamics on network.

    Parameters:
    - G (nx.classes.graph.Graph): generated networkx graph
    - states (np.ndarray): states of all nodes in network
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    meanHammingDistance = []
    M = 0

    hammingDistance = hamming_vector(states,range(len(states)))
    meanHammingDistance.append(hammingDistance.mean())

    print(M, meanHammingDistance[-1])

    nodes = list(G.nodes())

    # converge when all nodes agree on state
    while (meanHammingDistance[-1] != 0.0):
        source = np.random.choice(nodes)
        destination = np.random.choice(list(G.neighbors(source))) # TODO: pre-define neighbors

        states = message_update(states, source, destination, alpha=alpha, beta=beta)

        M += 1

        if M % 1000 == 0:
            print(M, meanHammingDistance[-1])

        # re-calculate normalized hamming distance for all pair combinations for node update
        hammingDistance = hamming_vector(states, destination)
        meanHammingDistance.append(hammingDistance.mean())

    return M, meanHammingDistance

def parallel_simulation(G: nx.classes.graph.Graph, states: np.ndarray, proc: int, return_dict: dict, alpha: float = 1.0, beta: float = 0.0) -> int | list:
    """
    Function to run a simulation of communication dynamics on network.
    - Parallel version

    Parameters:
    - G (nx.classes.graph.Graph): generated networkx graph
    - states (np.ndarray): states of all nodes in network
    - proc (int): process ID
    - return_dict (dict): simulation results
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    print(f"Starting Process {proc}!")

    meanHammingDistance = []
    M = 0

    hammingDistance = hamming_vector(states,range(len(states)))
    meanHammingDistance.append(hammingDistance.mean())

    print(M, meanHammingDistance[-1])

    nodes = list(G.nodes())

    # converge when all nodes agree on state
    while (meanHammingDistance[-1] != 0.0):
        source = np.random.choice(nodes)
        destination = np.random.choice(list(G.neighbors(source))) # TODO: pre-define neighbors

        states = message_update(states, source, destination, alpha=alpha, beta=beta)

        M += 1

        if M % 1000 == 0:
            print(f"messages={M}; dissimilarity={round(meanHammingDistance[-1],5)}; proc={proc}")

        # re-calculate normalized hamming distance for all pair combinations for node update
        hammingDistance = hamming_vector(states, destination)
        meanHammingDistance.append(hammingDistance.mean())

    return_dict[proc] = [M, meanHammingDistance]
    print(f"Process {proc} complete!")

def f(d, l):
    d[1] = '1'
    d['2'] = 2
    d[0.25] = None
    l.reverse()

if __name__ == "__main__":
    
    # filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'
    filename = 'graphs/official-generation/tau1=2.5-tau2=1.1-mu=0.45-avg_deg=25-min_comm=10-seed=99.pickle'
    G = pickle.load(open(filename, 'rb'))
    n = len(list(G))
    nbits = 3
    states = init(n, nbits)

    # M, meanStringDifference = simulate(G, states, alpha = 1.0, beta = 0.0)
    # print(M)

    print(filename)

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    start_time = []
    end_time = []
    start = time.time()

    for i in range(cpu_count()):
        p = Process(target=parallel_simulation, args = (G, states, i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    end = time.time()

    print(f'Execution time = {end-start} seconds')
    
    # with Pool(processes=cpu_count()) as p:

    #     S = partial(simulate, G=G, alpha=alpha, beta=beta)

    #     for result in p.imap(S,states=d):
    #         print(f'Got result: {result}', flush=True)

    # with Manager() as manager:
    #     return_dict = manager.dict()

    #     p = Process(target=parallel_simulation, args = (G, states, return_dict, alpha, beta))
    #     p.start()
    #     p.join()

    #     print(return_dict[0])
        