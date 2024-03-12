"""
Script to run communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
12/03/2024
"""

from dynamics_nonvectorized_Atlas import simulate, init
import pickle, networkx as nx, matplotlib.pyplot as plt, numpy as np, glob

def visualize(G):
    """
    Function to visualize a given graph/network G.
    Each node is labeled with its current state.

    Parameters:
    - G: networkx graph to display
    """

    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

    nx.draw(G, pos, labels=nx.get_node_attributes(G,'state'))
    plt.show()


def tolerant_mean(arrs):
    """
    Function to compute mean of curves with differing lengths

    Parameters:
    - arrs: nested list with sublists of different lengths

    Returns:
    - arr.mean: computed mean over all sublists
    - arr.std: computed standard deviation over all sublists
    """
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)

listFileNames = sorted(glob.glob(f'graphs/*.pickle'))

for j,file in enumerate(listFileNames):
    print(j)
    name = file[file.index('graphs/') + len('graphs/') + 0: file.index('.pickle')]

    G = pickle.load(open(file,'rb'))

    fig_scatter = plt.figure()
    fig_converge = plt.figure()
    fig_mean = plt.figure()

    total_messages = []
    allDifferences = []


    for i in range(100):

        G_init = init(G)

        M, meanStringDifference = simulate(G_init)

        total_messages.append(M)
        allDifferences.append(list(meanStringDifference))

        plt.figure(fig_converge)
        plt.plot(np.arange(0,len(meanStringDifference)), np.reshape(meanStringDifference,(-1,1)),label=f'Run {i}')

    plt.figure(fig_scatter)
    plt.scatter(range(100),total_messages)
    plt.xlabel("Iterations", fontsize=12)
    plt.ylabel("Total messages sent", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Consensus distribution (N=100) - Graph Atlas {name}")
    plt.savefig(f"images/simulation-results/scatter-{name}.png",bbox_inches='tight')

    plt.close(fig_scatter)

    plt.figure(fig_converge)
    plt.xlabel("Total messages sent", fontsize=12)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Convergence rates (N=100) - Graph Atlas {name}")
    # plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(f"images/simulation-results/convergence-{name}.png", bbox_inches='tight')
    plt.close(fig_converge)

    # mean convergence plot over all simulations
    plt.figure(fig_mean)
    y, error = tolerant_mean(allDifferences)
    plt.plot(np.arange(len(y))+1, y)
    plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5)
    plt.xlabel("Total messages sent", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Mean convergence (N=100) - Graph Atlas {name}")
    plt.savefig(f"images/simulation-results/mean-convergence-{name}.png",bbox_inches='tight')
    plt.close(fig_mean)
