"""
Script for preliminary network of message passing until consensus in a binary tree graph

Written by Jade Dubbeld
07/12/2023
"""

import random, numpy as np
import matplotlib.pyplot as plt
import networkx as nx

def init_binary(depth=3):
    """
    Structure binary tree with given depth/height.
    Initialize each node with random 3-bit strings.

    Parameters:
    - depth: depth (or height) of binary tree

    Returns:
    - G: networkx graph
    """

    # set up binary tree with given depth/height (h) [and branching factor (r) = 2]
    G = nx.balanced_tree(r=2,h=depth)

    # initialize all nodes with 3-bit string state
    for node in G.nodes:
        binary_string = f'{random.getrandbits(3):=03b}'     # generate random 3-bit string
        G.nodes[node]['state'] = binary_string

    # show initial configuration
    # print(G.nodes(data=True))
    return G

def visualize(G):
    """
    Function to visualize a given graph/network G.
    Each node is labeled with its current state.

    Parameters:
    - G: networkx graph
    """
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))
    plt.show()

def forward_messaging(G,M):
    """
    Function to forward root node's message throughout the entire network.
    Every downstream neighbor copies root node's message into position.

    Parameters:
    - G: networkx graph
    - M: counter for total messages send in network simulation

    Returns:
    - M: counter for total messages send in network simulation
    """

    # pick random bit from state
    index = random.choice([0,1,2])
    # print(index)

    message = G.nodes[0]['state'][index]
    
    # forward root node's message to all downstream neighbors
    for node in G.nodes:
        for neighbor in G.neighbors(node):

            # only propagate downstream
            if neighbor < node:
                continue

            # get current state of selected neighbor
            current_state = G.nodes[neighbor]['state']

            # copy received bit at given position (redundant if bit is already agreed)
            new_state = current_state[:index] + message + current_state[index + 1:]
            G.nodes[neighbor]['state'] = new_state
            M += 1
    
    return M

def random_messaging(G,M):
    """
    Function to send messages containing a random bit through the entire network.
    Every downstream neighbor copies the bit from message into position.
    Then randomly sends a bit from its own state to its downstream neighbors, until leaf nodes are reached.

    Parameters:
    - G: networkx graph
    - M: counter for total messages send in network simulation

    Returns:
    - M: counter for total messages send in network simulation
    """
    
    # randomly generate node's message
    for node in G.nodes:

        # pick random bit from state
        index = random.choice([0,1,2])
        # print(f"Bit-string index = {index} | Node = {node}")
        
        message = G.nodes[node]['state'][index]

        # send message to its downstream neigbors
        for neighbor in G.neighbors(node):
            if neighbor < node:
                continue

            # get current state of selected neighbor
            current_state = G.nodes[neighbor]['state']

            # copy received bit at given position (redundant if bit is already agreed)
            new_state = current_state[:index] + message + current_state[index + 1:]
            G.nodes[neighbor]['state'] = new_state
            M += 1

        # print(f"{G.nodes(data=True)}\n")
    
    return M

def efficient_messaging(G,M):
    """
    Function to send messages through the entire network.
    Message contains a random but correct bit (according to kingpin's state) in current node's state 
        but nonmatching with its downstream neighbor.
    Every downstream neighbor copies the bit from message into position.
    Then randomly sends a nonmatching, but correct bit from its own state to its downstream neighbors, 
        until leaf nodes are reached.

    NOTE: Messages passed can differ for each neighbor!

    Parameters:
    - G: networkx graph
    - M: counter for total messages send in network simulation

    Returns:
    - M: counter for total messages send in network simulation
    """

    # randomly pick a nonmatching bit as node's message
    for node in G.nodes:

        if node == 0 or G.nodes[0]['state'] == G.nodes[node]['state']:
            continue

        correct = []

        # find correct bits in current node's state compared to kingpin's state
        for position, bit in enumerate(G.nodes[node]['state']):
            if bit == G.nodes[0]['state'][position]:
                correct.append(position)

        # get parent node of current node
        parent = [(neighbor) for neighbor in G.neighbors(node) if neighbor < node]

        # print(f"Kingpin: {G.nodes[0]['state']}")
        # print(f"Current node: {G.nodes[node]['state']}")
        # print(f"Parent node: {G.nodes[parent[0]]['state']}")


        # nonmatching = []
        # # find nonmatching bits between current node (out of correct bits) and current node
        # for position, bit in enumerate(G.nodes[parent[0]]['state']):
        #     if bit != G.nodes[node]['state'][position] and position in correct:
        #         nonmatching.append(position)

        # print(f"correct: {correct} | nonmatching: {nonmatching}")

        # pick random bit when no correct bits in current state
        if correct == []:
            index = random.choice([0,1,2])

        # pick random incorrect bit from parent's state
        else:
            incorrect = [i for i in [0,1,2] if i not in correct]
            index = random.choice(incorrect)

        # print(f"Index {index}")
        
        # message nonmatching but already correct bit from parent to child
        message = G.nodes[parent[0]]['state'][index]

        # get current state of selected downstream neighbor
        current_state = G.nodes[node]['state']

        # copy received bit at given position (redundant if bit is already agreed)
        new_state = current_state[:index] + message + current_state[index + 1:]
        G.nodes[node]['state'] = new_state
        M += 1
    
    return M

def hamming_distance(string1, string2):
    """
    Function to compute string similarity using Hamming distance.

    Parameters:
    - string1: First string in comparison
    - string2: Second string in comparison

    Returns:
    - distance: number differing characters between string1 and string2
    """

    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(string1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if string1[i] != string2[i]:
            distance += 1
    # Return the final count of differences
    return distance

def simulate(depth, n_iters, messaging='forward'):
    """
    Function to run network simulation on binary tree with given method of 
    message passing [forward, random]

    Parameters:
    - depth: depth of binary tree
    - n_iters: number of iterations in a single simulation
    - messaging: type of message passing framework [forward, random]

    Returns:
    - total_messages: list containing all total messages sent for each simulation run
    - all_similarites: nested list where each sublist contains the average string similarity of each simulation
    """

    # initials
    total_messages = []
    all_similarities = []

    # simulate for n_iters
    for iter in range(n_iters):
        print(f"iter={iter}")
        similarity = []

        # initialize binary tree
        G = init_binary(depth=depth)

        # initial settings
        M = 0
        attributes = nx.get_node_attributes(G, "state")

        round = 0

        # converge when all nodes agree on state
        while np.unique(list(attributes.values())).size > 1:

            # print(f"Round {round}")
            S = []

            # different message passing frameworks
            if messaging == 'forward':
                # propagate root node's message through network
                M = forward_messaging(G=G, M=M)

            elif messaging == 'random':
                # send random messages to downstream neighbors
                M = random_messaging(G=G, M=M)
            
            elif messaging == 'efficient':
                # send nonmatching and correct bit to downstream neighbors
                M = efficient_messaging(G=G, M=M)

            # update attributes dictionary
            attributes = nx.get_node_attributes(G, "state")

            # string similarity using Hamming distance    
            for node in G.nodes():
                distance = hamming_distance(attributes[0],attributes[node])
                S.append(distance)
                
            # print(f"Similarity = {S}")
            # print(f"Average string similarity in current round = {np.mean(S)}")
            
            # append mean string similarity of current round
            similarity.append(np.mean(S))

            round += 1

        print(f"Number of messages send until consensus = {M} \n")

        # append total messages sent in current iteration
        total_messages.append(M)

        # print(f"Average similarity in iteration {iter} = {similarity}")
        
        # append complete list of string similarity in current iteration
        all_similarities.append(list(similarity))
    
    print(f"All similarities = {all_similarities}")
    return total_messages, all_similarities