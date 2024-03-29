"""
Script containing all required functions to simulate Deffuant-like dynamics on networks.
- Non-vectorized hamming distance computation

Written by Jade Dubbeld
22/02/2024
"""

import networkx as nx, random, numpy as np, time

def init(G: nx.classes.graph.Graph) -> nx.classes.graph.Graph:
    """
    Function to initialize a network with initial 3-bit string states.
    Parameters:
    - G (nx.classes.graph.Graph): Randomly generated network.
    Returns:
    - G (nx.classes.graph.Graph): Randomly generated network with initial states.
    """
    # initialize all nodes with 3-bit string state
    for i, node in enumerate(G.nodes):
        binary_string = f'{random.getrandbits(3):=03b}'     # generate random 3-bit string
        bits = bytes(binary_string, 'utf-8')                # convert to bytes
        G.nodes[node]['state'] = bits
    return G

def message_update(G: nx.classes.graph.Graph, source: int, destination: int, attributes: dict, alpha: float = 1.0, beta: float = 0.0) -> nx.classes.graph.Graph | dict:
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - G (nx.classes.graph.Graph): current state of networkx graph
    - source (int): selected sender node in network
    - destination (int): selected receiver node in network
    - attributes (dict): states of all nodes in network
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - G: new state of networkx graph
    - attributes (dict): states of all nodes in network

    """

    match = []

    # find correct bits in source node's state compared to destination node's state
    for position, bit in enumerate(attributes[source]):
        if bit == attributes[destination][position]:
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
    message = attributes[source][index:index+1]

    # generate random float representing miscommunication/receiver bias
    P_Miss = random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == b'0':
            message = b'1'
        elif message == b'1':
            message = b'0'

    # get current state of selected downstream neighbor
    current_state = attributes[destination]

    # copy received bit at given position (redundant if bit is already agreed)
    new_state = current_state[:index] + message + current_state[index + 1:]
    attributes[destination] = new_state

    return G, attributes


def hamming_distance(string1: str, string2: str) -> int:
    """
    Function to compute string difference using Hamming distance.
    Parameters:
    - string1 (str): First string in comparison
    - string2 (str): Second string in comparison
    Returns:
    - distance (int): number differing characters between string1 and string2
    """
    distance = 0
    L = len(string1)
    for i in range(L):
        if string1[i] != string2[i]:
            distance += 1
    return distance


def simulate(G: nx.classes.graph.Graph, alpha: float =1.0, beta: float =0.0) -> int | list:
    """
    Function to run a simulation for n_iters on a lattice.
    Parameters:
    - G (nx.classes.graph.Graph): generated networkx graph
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)
    
    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    N = len(G.nodes())
    nPairs = (N*N/2-(N/2))
    meanStringDifference = []
    stringDifference = np.zeros((N,N))
    M = 0
    attributes = nx.get_node_attributes(G, "state")
    
    # for each node pair (no redundant calculations)
    for index1, node1 in enumerate(G.nodes()):
        for index2, node2 in enumerate(G.nodes()):
            if node1 >= node2:
                continue

            # compute hamming distance for initial configuration
            hammingDistance = hamming_distance(attributes[node1],attributes[node2]) / 3 # regular

            # fill in normalized hamming distance array
            stringDifference[index1,index2] = hammingDistance

    meanStringDifference.append(stringDifference.sum()/nPairs)

    print(M, time.time(), meanStringDifference[-1])

    # converge when all nodes agree on state
    while (np.unique(list(attributes.values())).size > 1):
        source = random.choice(list(G.nodes))
        destination = random.choice(list(G.neighbors(source)))

        G,attributes = message_update(G=G,source=source,destination=destination,attributes=attributes,alpha=alpha,beta=beta)

        M += 1

        if M % 1000 == 0:
            print(M, time.time(), meanStringDifference[-1])

        # re-calculate normalized hamming distance for all pair combinations for node update
        for node in G.nodes(): 
            if destination == node:
                continue

            hammingDistance = hamming_distance(attributes[destination],attributes[node]) / 3 # regular

            # fill in normalized hamming distance array
            if node < destination:
                stringDifference[node,destination] = hammingDistance
            elif node > destination:
                stringDifference[destination,node] = hammingDistance

        meanStringDifference.append(stringDifference.sum()/nPairs)

    return M, meanStringDifference


if __name__ == "__main__":
    import pickle
    # filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'
    filename = 'graphs/official-generation/tau1=2.5-tau2=1.1-mu=0.45-avg_deg=25-min_comm=10-seed=99.pickle'
    
    G = pickle.load(open(filename, 'rb'))
    G = init(G)

    M, meanStringDifference = simulate(G, alpha=1.0, beta=0.0)
    print(M)