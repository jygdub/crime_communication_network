"""
Script to generate networks in accordance with a degree sequence following a power law distribution and 
    community sizes following a power law distribution.

Written by Jade Dubbeld
07/02/2024
"""

from networkx.generators.community import LFR_benchmark_graph
from tqdm import tqdm
import networkx as nx, matplotlib.pyplot as plt, numpy as np, pickle, random


def generate(n, exp_degree, exp_community, P_intercommunity, min_degree=None, avg_degree=None, min_community=10,seed=0, max_iters=500):
    """
    Function to generate a LFR network given parameter settings.

    Parameters:
    - n: number of nodes in network
    - exp_degree: exponent of power law for degree sequence
    - exp_community: exponent of power law for community sizes
    - P_interconnected: probability of making interconnected links
    - min_degree: minimum degree for a node
    - avg_degee: average degree across the network to satifsy
    - seed: random seed
    - max_iters: maximum number of iteration to try to construct a successful network

    Returns:
    - G: Constructed LFR network in accordance with parameter settings.
    """   

    if min_degree is not None:
        G = LFR_benchmark_graph(n=n,
                                tau1=exp_degree,
                                tau2=exp_community,
                                mu=P_intercommunity,
                                min_degree=min_degree,
                                min_community=min_community,
                                seed=seed,
                                max_iters=max_iters)
    elif avg_degree is not None:
        G = LFR_benchmark_graph(n=n,
                                tau1=exp_degree,
                                tau2=exp_community,
                                mu=P_intercommunity,
                                average_degree=avg_degree,
                                min_community=min_community,
                                seed=seed,
                                max_iters=max_iters)
    return G

def rewire(G):
    """
    Function to interconnect separate components of graph and rewire self-loops.

    Parameters:
    - G: Constructed LFR network in accordance with parameter settings.

    Returns:
    - G: Constructed LFR network in accordance with parameter settings.
    """

    # delete self-loop and randomly reconnect to connected component
    if nx.nodes_with_selfloops(G):
        for node in list(nx.nodes_with_selfloops(G)):
            G.remove_edge(node, node)
            G.add_edge(node, random.choice(list(G.nodes())))

    # connect separated parts of graph randomly
    if not nx.is_connected(G):
        print("rewiring")
        components = sorted(nx.connected_components(G), key=len, reverse=True)

        for c in range(1,len(components)):
            G.add_edge(random.choice(components[c]), random.choice(list(G.nodes())))

    return G


""" Use for generating single network"""
# n = 1000
# exp_degree = 3.    # exponent power law distribution for degree sequence
# exp_community = 1.5    # exponent power law distrubution for community sizes
# P_intercommunity = 0.75    # {(1-µ) fraction of links of every node is with nodes of the same community, while fraction μ is with the other nodes
# # min_degree = 2
# avg_degree = 25
# min_community = 10
# seed=20
# max_iters=1000

# try:
#     G = LFR_benchmark_graph(n=n,
#                     tau1=exp_degree,
#                     tau2=exp_community,
#                     mu=P_intercommunity,
#                     # min_degree=min_degree,
#                     average_degree=avg_degree,
#                     min_community=min_community,
#                     seed=seed,
#                     max_iters=max_iters)
    
#     if nx.is_connected(G):
#         pickle.dump(G, open(f'graphs/tau1={exp_degree}-tau2={exp_community}-mu={P_intercommunity}-avg_deg={avg_degree}-min_comm={min_community}-seed={seed}.pickle', 'wb'))
#     else:
#         pickle.dump(G, open(f'graphs/pre-tau1={exp_degree}-tau2={exp_community}-mu={P_intercommunity}-avg_deg={avg_degree}-min_comm={min_community}-seed={seed}.pickle', 'wb')) 
#         G = rewire(G)
#         pickle.dump(G, open(f'graphs/tau1={exp_degree}-tau2={exp_community}-mu={P_intercommunity}-avg_deg={avg_degree}-min_comm={min_community}-seed={seed}.pickle', 'wb')) 
# except nx.exception.ExceededMaxIterations:
#     print(f"seed {seed} mu {P_intercommunity} avg_degree {avg_degree} exceeded max_iters")


"""
Use loops to generate multiple networks in one go

Interest in 
    - exp_degree (tau1) = [1.5,3.0,4.5,6.0,7.5]
    - exp_community (tau2) = [1.5,3.0,4.5,6.0,7.5]
    - P_intercommunity (mu) = [0.25,0.50,0.75]
    - avg_degree (average_degree) = [5,10,15,20,25]
    for various seeds
"""
n = 1000
min_community = 10
exp_degree = 3.0
exp_community = 1.5

for P_intercommunity in [0.25,0.5,0.75]:
        
    min_degree = None
    
    for avg_degree in [5,10,15,20,25]:
        print(f"Average degree - {P_intercommunity,avg_degree}")

        for seed in range(0,61): # change seed range accordingly

            # skip seeds that take forever
            if P_intercommunity == 0.5 and avg_degree == 15 and seed == 5:
                continue

            if P_intercommunity == 0.5 and avg_degree == 20 and seed == 7:
                continue

            if P_intercommunity == 0.75 and avg_degree == 15 and seed == 5:
                continue

            if P_intercommunity == 0.75 and avg_degree == 20 and (seed == 7 or seed == 19):
                continue

            if P_intercommunity == 0.5 and avg_degree == 20 and seed == 25:
                continue

            if P_intercommunity == 0.5 and avg_degree == 25 and seed == 25:
                continue

            if P_intercommunity == 0.75 and avg_degree == 10 and seed == 39:
                continue

            if P_intercommunity == 0.75 and avg_degree == 20 and (seed == 25 or seed == 27):
                continue

            if P_intercommunity == 0.75 and avg_degree == 25 and (seed == 25 or seed == 33):
                continue

            try:

                # generate network
                G = generate(n,exp_degree,exp_community,P_intercommunity,min_degree,avg_degree,min_community,seed,1000)

                # store graph if connect (load using G = pickle.load(open(f'graphs/test.pickle', 'rb'))
                if nx.is_connected(G):
                    print(f"seed={seed}")
                    pickle.dump(G, open(f'graphs/tau1={exp_degree}-tau2={exp_community}-mu={P_intercommunity}-avg_deg={avg_degree}-min_comm={min_community}-seed={seed}.pickle', 'wb')) 
                    
                    # degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
                    # print(np.mean(degree_sequence))
                    # print(nx.is_connected(G))
                    # print(f"Min. degree: {min(degree_sequence)} - Max. degree: {max(degree_sequence)}")
                else:
                    print(f'seed {seed} not connected')
                    pickle.dump(G, open(f'graphs/pre-tau1={exp_degree}-tau2={exp_community}-mu={P_intercommunity}-avg_deg={avg_degree}-min_comm={min_community}-seed={seed}.pickle', 'wb')) 
                    G = rewire(G)
                    pickle.dump(G, open(f'graphs/tau1={exp_degree}-tau2={exp_community}-mu={P_intercommunity}-avg_deg={avg_degree}-min_comm={min_community}-seed={seed}.pickle', 'wb')) 
                    
            
            except nx.exception.ExceededMaxIterations:
                print(f'Seed {seed} exceeded max_iters!')
                continue

"""Use to show assigned communities in given network G"""
# communities = {frozenset(G.nodes[v]["community"]) for v in G}
# print(communities)

"""Use to plot degree distribution"""
# fig = plt.figure("Degree of a LFR graph", figsize=(8, 8))

# degree_sequence = sorted((d for n, d in G.degree()), reverse=True)

# # create a gridspec for adding subplots of different sizes
# axgrid = fig.add_gridspec(5, 4)

# ax0 = fig.add_subplot(axgrid[0:3, :])
# # Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# pos = nx.spring_layout(G, seed=10396953)
# nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=20)
# nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.4)
# ax0.set_title("Network G")
# ax0.set_axis_off()

# ax1 = fig.add_subplot(axgrid[3:, :2])
# ax1.plot(degree_sequence, "b-", marker="o")
# ax1.set_title("Degree Rank Plot")
# ax1.set_ylabel("Degree")
# ax1.set_xlabel("Rank")

# ax2 = fig.add_subplot(axgrid[3:, 2:])
# ax2.bar(*np.unique(degree_sequence, return_counts=True))
# ax2.set_title("Degree histogram")
# ax2.set_xlabel("Degree")
# ax2.set_ylabel("# of Nodes")

# fig.tight_layout()
# plt.show()
