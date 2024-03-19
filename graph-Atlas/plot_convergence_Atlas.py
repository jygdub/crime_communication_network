"""
Script to plot convergence results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
import numpy as np, pandas as pd, pickle

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

def string_to_list(string: str) -> list:
    result = string.strip('][').split(', ')
    return np.array(result,dtype=float)



from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
import numpy as np, pandas as pd, pickle


df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

allDifferences = []

for idx in tqdm(range(len(df))):

    fig_scatter, ax_scatter = plt.subplots()
    fig_converge, ax_converge = plt.subplots()
    fig_mean, ax_mean = plt.subplots()

    name = 'G' + str(df['index'].iloc[idx])
    df_convergence = pd.read_csv(f'results/convergence-{name}.tsv',sep='\t')

    for iter in range(100):
        convergence = string_to_list(df_convergence['meanHammingDist'][iter])
        allDifferences.append(convergence)

        plt.figure(fig_converge)
        ax_converge.plot(np.arange(0,len(convergence)), convergence,label=f'Run {iter}')

    plt.figure(fig_scatter)
    ax_scatter.scatter(range(100),df_convergence['nMessages'])
    plt.xlabel("Networks", fontsize=12)
    plt.ylabel("Total messages sent", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Consensus distribution (N=100) - Graph Atlas {name} (n={df.nodes.iloc[idx]})")
    plt.savefig(f"images/simulation-results/scatter-{name}.png",bbox_inches='tight')
    plt.close(fig_scatter)

    plt.figure(fig_converge)
    plt.xlabel("State update", fontsize=12)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Convergence rates (N=100) - Graph Atlas {name} (n={df.nodes.iloc[idx]})")
    # plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(f"images/simulation-results/convergence-{name}.png", bbox_inches='tight')
    plt.close(fig_converge)

    # mean convergence plot over all simulations
    plt.figure(fig_mean)
    y, error = tolerant_mean(allDifferences)
    ax_mean.plot(np.arange(len(y))+1, y)
    plt.fill_between(np.arange(len(y))+1, y-error, y+error, alpha=0.5)
    plt.xlabel("State update", fontsize=14)
    plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"Mean convergence (N=100) - Graph Atlas {name} (n={df.nodes.iloc[idx]})")
    plt.savefig(f"images/simulation-results/mean-convergence-{name}.png",bbox_inches='tight')
    plt.close(fig_mean)