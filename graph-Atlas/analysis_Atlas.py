"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import numpy as np, pandas as pd

def relating_efficiency_convergence(metrics, efficiency, column):

    # if efficiency == 'global':
    #     column = 'globalEff'
    # elif efficiency == 'local':
    #     column = 'localEff'

    fig,ax = plt.subplots()

    for idx in tqdm(reversed(range(len(metrics)))):

        # retrieve number of nodes in graph
        n = metrics['nodes'].iloc[idx]

        # pre-determine colormap
        if  n == 2:
            color = "tab:blue"
        elif n == 3:
            color = "tab:orange"
        elif n == 4:
            color = "tab:green"
        elif n == 5:
            color = "tab:red"
        elif n == 6:
            color = "tab:purple"
        elif n == 7:
            color = "tab:pink"

        # generate filename
        name = 'G' + str(metrics['index'].iloc[idx])
        convergence = pd.read_csv(f'results/convergence-{name}.tsv', usecols=['nMessages'], sep='\t')

        # plot in figure
        ax.scatter(np.repeat(metrics[column][idx],100),convergence['nMessages'],color=color,alpha=0.5)
        handles = [
            plt.scatter([], [], color=c, label=l)
            for c, l in zip("tab:blue tab:orange tab:green tab:red tab:purple tab:pink".split(), "n=2 n=3 n=4 n=5 n=6 n=7".split())
        ]

        ax.legend(handles=handles)
        ax.set_xlabel(f"{efficiency.capitalize()} efficiency (metric)")
        ax.set_ylabel("Convergence rate (number of messages)")
        ax.set_title("Relation between structural and operational efficiency")

        plt.savefig(f"images/relations/relation-{efficiency}-convergence-scatter.png",bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":

    # for efficiency in ['global', 'local']:
    efficiency = 'local'

    if efficiency == 'global':
        column = 'globalEff'
    elif efficiency == 'local':
        column = 'localEff'

    metrics = pd.read_csv('data-GraphAtlas.tsv', usecols=['index', 'nodes', column], sep='\t')
    # print(metrics)

    relating_efficiency_convergence(metrics=metrics, efficiency=efficiency, column=column)

    # with Pool(processes=cpu_count()) as pool:
    #     for i in pool.imap_unordered(relating_efficiency_convergence, range(10)):
    #         print(i)

 




