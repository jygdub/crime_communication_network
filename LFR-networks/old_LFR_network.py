"""
Old version script containing all required functions to simulate Deffuant-like dynamics on networks.

Written by Jade Dubbeld
17/02/2024
"""

import networkx as nx, random, numpy as np, pickle, time

### test files
# filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'
# filename = 'graphs/test10-tau1=3.0-tau2=1.5-mu=0.3-avg_deg=2-min_comm=2-seed=0.pickle'
# filename = 'graphs/official-generation/tau1=2.5-tau2=1.1-mu=0.45-avg_deg=25-min_comm=10-seed=99.pickle'
# filename = 'graphs/first-generation/tau1=3.0-tau2=1.5-mu=0.25-avg_deg=10-min_comm=10-seed=12.pickle'


def init(G: nx.classes.graph.Graph, N: int, k: int) -> nx.classes.graph.Graph:
    """
    Function to initialize a network with initial states.

    Parameters:
    - G: Randomly generated network
    - N: network size
    - k: length of message (i.e. state)

    Returns:
    - G: Randomly generated network with initial states

    """

    # generate random integers in set {0,1} of size k for all agents in network
    bits = np.random.randint(0, 2, size=(N,k))

    # convert to dictionary to assign states in node attributes
    states  = dict(list(enumerate(bits)))
    nx.set_node_attributes(G, states,  'state')

    return G

def message(G: nx.classes.graph.Graph, source: int, destination: int, alpha: float = 1.0, beta: float = 0.0) -> nx.classes.graph.Graph:
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - G: current state of networkx graph
    - source (int): selected sender node in network
    - destination (int): selected receiver node in network
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

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
    message = G.nodes[source]['state'][index:index+1]

    # print(f"index = {index}, message = {message}")

    # generate random float representing miscommunication/receiver bias
    P_Miss = random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == b'0':
            message = b'1'
        elif message == b'1':
            message = b'0'

    # get current state of selected downstream neighbor
    current_state = G.nodes[destination]['state']

    # copy received bit at given position (redundant if bit is already agreed)
    new_state = current_state[:index] + message + current_state[index + 1:]
    G.nodes[destination]['state'] = new_state

    return G


def message_update(source: np.ndarray, destination: np.ndarray, alpha: float = 1.0, beta: float = 0.0) -> np.ndarray:
    """
    Function to send some message from source to destination.
    Correctness of message depends on probability alpha (1.0 is always correct - 0.0 is never correct).
    Receiver bias depends on probability beta (0.0 is never mistaken - 1.0 is always mistaken).

    Parameters:
    - source (numpy.ndarray): bit state of sender node
    - destination (numpy.ndarray): bit state of receiver node
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - destination (numpy.ndarray): updated state of receiver node

    """

    # compare source and destination states
    comparison = np.bitwise_xor(source,destination)
    
    # extract matching and mismatching position
    match = np.where(comparison==0)[0]
    mismatch = np.where(comparison==1)[0]

    # generate random float representing matching/sender bias
    P_Matching = random.random()

    # random pick if no bits in common
    # pick from correct list with probability alpha (incorrect with probability 1 - alpha)
    if P_Matching > alpha and not match.size == 0:
        index = random.choice(match)
    elif P_Matching <= alpha and not mismatch.size == 0:
        index = random.choice(mismatch)
    else:
        index = random.choice([0,1,2])

    # generate message
    message = source[index]

    # generate random float representing miscommunication/receiver bias
    P_Miss = random.random()

    # copy message given probability beta, otherwise bitflip
    if P_Miss <= beta:
        if message == 0:
            message = 1
        elif message == 1:
            message = 0

    # copy received bit at given position (NOTE: redundant if bit is already agreed)
    destination[index] = message

    return destination


def hamming_distance(state1: np.ndarray, state2: np.ndarray) -> int:
    """
    Function to compute string difference using Hamming distance.

    Parameters:
    - string1 (str): First string in comparison
    - string2 (str): Second string in comparison

    Returns:
    - distance (int): number differing characters between string1 and string2
    """

    distance = 0
    L = len(state1)

    for i in range(L):
        if state1[i] != state2[i]:
            distance += 1

    return distance


def hamming_distance_vector(arr: np.ndarray) -> float:
    """
    Function to compute pairwise bitstring difference (analogous to Hamming distance).

    Parameters:
    - arr (numpy.ndarray): array containing all agents' bit states

    Returns: 
    - hammDist (float): average pairwise hamming distance 
    """

    # hammDist = abs(arr[np.newaxis,:] - arr[:,np.newaxis]).mean() # give mean -1 if row-wise
    hammDist = np.bitwise_xor(arr[np.newaxis,:], arr[:,np.newaxis]).mean()
    return hammDist

# def count_ones_vectorized(arr: np.ndarray) -> int:
#     """
#     Function to count number of differing positions across all agents in a network.

#     Parameters:
#     - arr (numpy.ndarray): array of pairwise hamming distances (integer values)

#     Returns: 
#     - ones_count (int): number of differing positions in the entire network
#     """

#     # convert the array to binary representation
#     binary_arr = np.unpackbits(arr,axis=1)

#     # count the number of ones in each binary representation
#     ones_count = binary_arr.sum()
#     return ones_count

def simulate(G: nx.classes.graph.Graph, alpha: float = 1.0, beta: float = 0.0) -> int | list:
    """
    Function to run a single simulation.

    Parameters:
    - G (networkx.classes.graph.Graph): generated networkx graph
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - time_step (int): total messages sent in simulation (i.e. number of time steps)
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    N = len(G.nodes())
    meanStringDifference = []
    time_step = 0
    size = 1000
    
    attributes = nx.get_node_attributes(G, "state")
    attributes = np.array(list(attributes.values()))

    # compute hamming distance (vectorized)
    meanHammingDist = hamming_distance_vector(attributes)
    meanStringDifference.append(meanHammingDist)

    nodes = np.array(list(G.nodes))
    # neighbors = np.array(list(G.neighbors(nodes)))

    print(time_step, time.time(), meanStringDifference[-1])

    # converge when all nodes agree on state
    while (meanHammingDist != 0.0):

        randomSources = random.choices(nodes,weights=[1]*N,k=size)

        source = randomSources[time_step % size] 
        destination = np.random.choice(list(G.neighbors(source))) # TODO: list conversion uit loop en minder vaak random nummer

        attributes[destination] = message_update(source=attributes[source],destination=attributes[destination],alpha=alpha,beta=beta)

        # compute hamming distance (vectorized)
        meanHammingDist = hamming_distance_vector(attributes)
        meanStringDifference.append(meanHammingDist)

        time_step += 1

        if time_step % size == 0:
            randomSources = random.choices(nodes,weights=[1]*N,k=size)
            print(time_step, time.time(), meanStringDifference[-1])
    
    return time_step, meanStringDifference

def simulate_parallel(G: nx.classes.graph.Graph, proc: int, return_dict: dict, alpha: float = 1.0, beta: float = 0.0) -> dict:
    """
    Function to run a single simulation in parallel.

    Parameters:
    - G: generated networkx graph
    - alpha (float): probability of sender bias (sending match or mismatch bits)
    - beta (float): probability of receiver bias (flipping message or not)

    Returns:
    - M (int): total messages sent in simulation
    - meanStringDifference (list): list of all string difference scores in simulation
    """

    print(f"Starting Process {proc}!")

    N = len(G.nodes())
    nPairs = (N*N/2-(N/2))
    meanStringDifference = []
    stringDifference = np.zeros((N,N))
    M = 0
    attributes = nx.get_node_attributes(G, "state")

    """ Vectorized hamming distance computation """
    attributes = np.array(list(attributes.values()))

    # # compute hamming distance (vectorized)
    # hammingDistance = hamming_distance_vector(attributes)
    # meanHammDist = count_ones_vectorized(hammingDistance)/3/(N*N)
    # meanStringDifference.append(meanHammDist)
    """ Vectorized hamming distance computation """

    # for each node pair (no redundant calculations)
    for index1, node1 in enumerate(G.nodes()):
        for index2, node2 in enumerate(G.nodes()):
            if node1 >= node2:
                continue
            
            # compute hamming distance for initial configuration
            hammingDistance = hamming_distance(attributes[node1],attributes[node2]) / 3 # regular
            # hammingDistance = sum(hamming_distance_vector(attributes[node1],attributes[node2])) / 3 # vectorized

            # fill in normalized hamming distance array
            stringDifference[index1,index2] = hammingDistance

    meanStringDifference.append(stringDifference.sum()/nPairs)

    # converge when all nodes agree on state

    # while (np.unique(list(attributes.values())).size > 1):

    ### Vectorized hamming distance computation ###
    while ((attributes==attributes[0]).all()==False):

        source = random.choice(list(G.nodes))
        destination = random.choice(list(G.neighbors(source)))

        attributes[destination] = message_update(source=attributes[source],destination=attributes[destination],alpha=alpha,beta=beta)

        M += 1
        """ Uncomment when using message() and remove attributes in line 237, not for message_update()"""
        # attributes = nx.get_node_attributes(G, "state") 

        if M % 5000 == 0:
            print(f"messages={M}; dissimilarity={round(meanStringDifference[-1],5)}; proc={proc}")

        """ Vectorized hamming distance computation """
        # # compute hamming distance (vectorized)
        # hammingDistance = hamming_distance_vector(attributes)
        # meanHammDist = count_ones_vectorized(hammingDistance)/3/(N*N)
        # meanStringDifference.append(meanHammDist)
        """ Vectorized hamming distance computation """

        # re-calculate normalized hamming distance for all pair combinations for node update
        for node in G.nodes(): 
            if destination == node:
                continue

            hammingDistance = hamming_distance(attributes[destination],attributes[node]) / 3 # regular
            # hammingDistance = sum(hamming_distance_vector(attributes[destination],attributes[node])) / 3 # vectorized

            # fill in normalized hamming distance array
            if node < destination:
                stringDifference[node,destination] = hammingDistance
            elif node > destination:
                stringDifference[destination,node] = hammingDistance

        meanStringDifference.append(stringDifference.sum()/nPairs)
    
    return_dict[proc] = [M, meanStringDifference]