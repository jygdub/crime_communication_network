"""
Script to generate networks in accordance with a degree sequence following a power law distribution and 
    community sizes following a power law distribution.

Written by Jade Dubbeld
07/02/2024
"""

from networkx.generators.community import LFR_benchmark_graph
from tqdm import tqdm
import networkx as nx, matplotlib.pyplot as plt, numpy as np, pickle, random

def rewire_selfloops(G):
    """
    Function to rewire self-loops in graph.

    Parameters:
    - G: Graph with self-loops.

    Returns:
    - G: Graph without self-loops.
    """
    
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    # delete self-loop and randomly reconnect to giant component
    print("Rewiring self-loop(s)")
    
    for node in list(nx.nodes_with_selfloops(G)):
        G.remove_edge(node, node)
        G.add_edge(node, random.choice(list(G.nodes())))

    return G


def connect_components(G):
    """
    Function to interconnect separate components of graph.

    Parameters:
    - G: Graph with separated components.

    Returns:
    - G: Graph with a single giant component.
    """

    components = sorted(nx.connected_components(G), key=len, reverse=True)

    # connect random node of separated component to random node of giant component
    print("Linking disconnected component to giant component")

    for c in range(1,len(components)):
        G.add_edge(random.choice(list(components[c])), random.choice(list(components[0])))

    return G


""" Use for generating single network"""
# n = 1000
# exp_degree = 2.5    # exponent power law distribution for degree sequence
# exp_community = 1.1    # exponent power law distrubution for community sizes
# P_intercommunity = 0.05    # {(1-µ) fraction of links of every node is with nodes of the same community, while fraction μ is with the other nodes
# min_degree = None
# avg_degree = 5
# max_degree = int(0.1*n)
# min_community = 10
# max_community = int(0.1*n)
# seed=80
# max_iters=5000

# G = None

# try:
#     G = LFR_benchmark_graph(n=n,
#                         tau1=exp_degree,
#                         tau2=exp_community,
#                         mu=P_intercommunity,
#                         average_degree=avg_degree,
#                         max_degree=max_degree,
#                         min_community=min_community,
#                         max_community=max_community,
#                         seed=seed,
#                         max_iters=max_iters)
    
#     if nx.nodes_with_selfloops(G) or not nx.is_connected(G):
#         print(f'SEED {seed} has self-loops or is not connected')

#         if nx.nodes_with_selfloops(G):
#             print(f'SEED {seed} has self-loops')
#             G = rewire_selfloops(G)

#         if not nx.is_connected(G):
#             print(f'SEED {seed} is not connected')
#             G = connect_components(G)                     

#     print(nx.is_connected(G))
#     print([i for i in nx.nodes_with_selfloops(G)])

# except nx.exception.ExceededMaxIterations:
#     print(f"seed={seed}; µ={P_intercommunity}; <k>={avg_degree} exceeded max_iters")

"""
Use loops to generate multiple networks in one go

Interest in 
    - n = 1000
    - exp_degree (tau1) = 2.5
    - exp_community (tau2) = 1.1
    - min_community = 10
    - P_intercommunity (mu) = [0.05, 0.25, 0.45]
    - avg_degree (average_degree) = [5, 15, 25]
    - max_degree = max_community = 0.1*n
    
    Per parameter combination -> 100 networks
"""
n = 1000
exp_degree = 2.5
exp_community = 1.1
max_degree = int(0.1*n)
min_community = 10
max_community = int(0.1*n)
max_iters = 1000

P_intercommunity = 0.05
# for P_intercommunity in [0.05,0.25,0.45]:
for avg_degree in [25]:
    print(f'\n(µ,<k>) = {P_intercommunity,avg_degree}')
    
    seed = 445
    N = 0

    # for seed in range(0,101): # change seed range accordingly
    while N < (100 - 8):
        try:
            # generate network
            G = LFR_benchmark_graph(n=n,
                                    tau1=exp_degree,
                                    tau2=exp_community,
                                    mu=P_intercommunity,
                                    average_degree=avg_degree,
                                    max_degree=max_degree,
                                    min_community=min_community,
                                    max_community=max_community,
                                    seed=seed,
                                    max_iters=max_iters)

            # ensure network is connected without self-loops
            if nx.nodes_with_selfloops(G) or not nx.is_connected(G):
                print(f'SEED {seed} has self-loops or is not connected')
                pickle.dump(G, open(f'graphs/official-generation/pre-tau1={exp_degree}-tau2={exp_community}-mu={P_intercommunity}-avg_deg={avg_degree}-min_comm={min_community}-seed={seed}.pickle', 'wb')) 
                
                # rewire self-loops
                if nx.nodes_with_selfloops(G):
                    print(f'SEED {seed} has self-loops')
                    G = rewire_selfloops(G)

                # interconnect components
                if not nx.is_connected(G):
                    print(f'SEED {seed} is not connected')
                    G = connect_components(G)  
            else:
                print(f"SEED {seed} generated")
                
            # store connected graph without self-loops (load using G = pickle.load(open(f'graphs/test.pickle', 'rb'))
            pickle.dump(G, open(f'graphs/official-generation/tau1={exp_degree}-tau2={exp_community}-mu={P_intercommunity}-avg_deg={avg_degree}-min_comm={min_community}-seed={seed}.pickle', 'wb')) 

            N += 1
            seed += 1    
        
        except nx.exception.ExceededMaxIterations:
            print(f'seed {seed} exceeded max_iters!')
            seed += 1
            continue

"""Use to show assigned communities in given network G"""
# communities = {frozenset(G.nodes[v]["community"]) for v in G}
# print(communities)

"""Use to plot degree distribution"""
# filename = 'graphs/test10-tau1=3.0-tau2=1.5-mu=0.3-avg_deg=2-min_comm=2-seed=0.pickle'
# filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'
# filename = 'graphs/first-generation/tau1=3.0-tau2=1.5-mu=0.25-avg_deg=10-min_comm=10-seed=12.pickle'
# G = pickle.load(open(filename, 'rb'))

# if G:
#     fig = plt.figure("Degree of a LFR graph", figsize=(8, 8))

#     degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

#     # create a gridspec for adding subplots of different sizes
#     axgrid = fig.add_gridspec(5, 4)

#     ax0 = fig.add_subplot(axgrid[0:3, :])
#     # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
#     pos = nx.spring_layout(G, seed=10396953)
#     nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=20)
#     nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.4)
#     ax0.set_title("Network G")
#     ax0.set_axis_off()

#     ax1 = fig.add_subplot(axgrid[3:, :2])
#     ax1.plot(degree_sequence, "b-", marker="o")
#     ax1.set_title("Degree Rank Plot")
#     ax1.set_ylabel("Degree")
#     ax1.set_xlabel("Rank")

#     ax2 = fig.add_subplot(axgrid[3:, 2:])
#     ax2.bar(*np.unique(degree_sequence, return_counts=True))
#     ax2.set_title("Degree histogram")
#     ax2.set_xlabel("Degree")
#     ax2.set_ylabel("# of Nodes")

#     fig.tight_layout()
#     plt.show()
