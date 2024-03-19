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


if __name__ == "__main__":

    efficiency = 'global'
    fig,ax = plt.subplots()

    if efficiency == 'global':
        column = 'globalEff'
    elif efficiency == 'local':
        column = 'localEff'

    data = pd.read_csv('relationData-complete-Atlas.tsv', sep='\t')

    for n in reversed(data['nodes'].unique()):
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

        indices = np.where(data['nodes'] == n)[0]
        ax.scatter(data[column].iloc[indices],data['nMessages'].iloc[indices],color=color,alpha=0.3)

        if n != 2:
            p = np.poly1d(np.polyfit(data[column].iloc[indices],data['nMessages'].iloc[indices],3))
            t = np.linspace(min(data[column]), max(data[column]), 250)
            print(p)
            ax.plot(t,p(t),color)


    handles = [
        plt.scatter([], [], color=c, label=l)
        for c, l in zip("tab:blue tab:orange tab:green tab:red tab:purple tab:pink".split(), "n=2 n=3 n=4 n=5 n=6 n=7".split())
    ]

    ax.legend(handles=handles)
    ax.set_xlabel(f"{efficiency.capitalize()} efficiency (metric)")
    ax.set_ylabel("Convergence rate (number of messages)")
    ax.set_title(f"Relation between structural and operational efficiency ({efficiency})")
    plt.show()

    fig.savefig(f"images/relations/relation-{efficiency}-convergence-polynomial.png",bbox_inches='tight')
    plt.close(fig)





    # with Pool(processes=cpu_count()) as pool:
    #     for i in pool.imap_unordered(relating_efficiency_convergence, range(10)):
    #         print(i)

 




