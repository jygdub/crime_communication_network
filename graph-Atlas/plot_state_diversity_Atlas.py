"""
Script to plot state diversity results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
import numpy as np, pandas as pd, pickle


df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

for idx, iter in tqdm(product(range(len(df)),range(100))):

    name = 'G' + str(df['index'].iloc[idx])
    df_states = pd.read_csv(f'results/states-{name}-run{iter}.tsv',sep='\t')

    fig, ax = plt.subplots()

    for state in df_states.columns:
        ax.plot(np.arange(0,len(df_states)), df_states[state], label=state)

    plt.xlabel("State update", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.title(f"States trajectory - Graph Atlas {name} (n={df.nodes.iloc[idx]};run={iter})")
    plt.legend(bbox_to_anchor=(1,1))
    plt.savefig(f"images/simulation-results/states-{name}-run{iter}.png",bbox_inches='tight')
    plt.close(fig)    


