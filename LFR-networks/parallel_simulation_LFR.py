"""
Script (speedup) to simulate multiple Deffuant-like dynamics on LFR benchmark graphs in parallel.

Written by Jade Dubbeld
22/02/2024
"""


import networkx as nx, random, numpy as np, matplotlib.pyplot as plt, pickle, time
from multiprocessing import Process, Manager

from LFR_network import init, simulate_parallel

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


if __name__ == "__main__":
    alpha = 1.0
    beta = 0.0

    filename = 'graphs/official-generation/tau1=2.5-tau2=1.1-mu=0.05-avg_deg=5-min_comm=10-seed=107.pickle'
    # filename = 'graphs/official-generation/tau1=2.5-tau2=1.1-mu=0.45-avg_deg=25-min_comm=10-seed=99.pickle'
    # filename = 'graphs/test10-tau1=3.0-tau2=1.5-mu=0.3-avg_deg=2-min_comm=2-seed=0.pickle'
    # filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'
    
    G_init = pickle.load(open(filename, 'rb'))
    G = init(G_init)

    print(filename)

    manager = Manager()
    return_dict = manager.dict()
    jobs = []

    start_time = []
    end_time = []
    start = time.time()

    for i in range(8):
        start_time.append(time.time())
        p = Process(target=simulate_parallel, args = (G, i, return_dict, alpha,beta))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()
        end_time.append(time.time())

    end = time.time()

    print(f'Execution time = {end-start} seconds')

    # print(f'Process completion time: {[a - b for a, b in zip(end_time, start_time)]}')
    # print(f'Script termination: {end - start}')

    # convergence plot simulation
    fig_converge = plt.figure(figsize=(13,8))
    fig_mean = plt.figure()

    allDifferences = []

    for proc in return_dict.keys():

        allDifferences.append(return_dict[proc][1])

        plt.figure(fig_converge)
        plt.plot(np.arange(0,len(return_dict[proc][-1])), np.reshape(return_dict[proc][-1],(-1,1)), label=f'run{proc}')

    plt.figure(fig_converge)
    plt.xlabel("Total messages sent", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Convergence on LFR")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(f"images/test1000-parallel-convergence.png", bbox_inches='tight')

    # mean convergence plot over all simulations
    plt.figure(fig_mean)
    y, error = tolerant_mean(allDifferences)
    plt.plot(np.arange(len(y))+1, y)
    plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5)
    plt.xlabel("Total messages sent", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f"Mean convergence (n_samples={len(return_dict.keys())})")
    plt.savefig(f"images/test1000-parallel-mean-convergence.png",bbox_inches='tight')