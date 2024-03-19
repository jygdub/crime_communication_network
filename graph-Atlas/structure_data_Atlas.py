"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np, pandas as pd


metrics = pd.read_csv('data-GraphAtlas.tsv', sep='\t')
print(metrics)

complete = pd.DataFrame(data=None, index=np.arange(len(metrics)*100), 
                        columns=['index',
                                 'nodes',
                                 'degree',
                                 'betweenness',
                                #  'CFbetweennesss',
                                 'closeness',
                                 'clustering',
                                 'globalEff',
                                 'localEff',
                                 'nMessages'])

for idx in tqdm(range(len(metrics))):

    # n100 = int(np.floor(idx / 100))

    n100 = idx*100

    # generate filename
    name = 'G' + str(metrics['index'].iloc[idx])
    convergence = pd.read_csv(f'results/convergence-{name}.tsv', usecols=['nMessages'], sep='\t')

    complete['index'].iloc[n100:n100+100] = metrics['index'].iloc[idx]
    complete['nodes'].iloc[n100:n100+100] = metrics['nodes'].iloc[idx]
    complete['degree'].iloc[n100:n100+100] = metrics['degree'].iloc[idx]
    complete['betweenness'].iloc[n100:n100+100] = metrics['betweenness'].iloc[idx]
    # complete['CFbetweenness'].iloc[n100:n100+100] = metrics['CFbetweenness'].iloc[idx]
    complete['closeness'].iloc[n100:n100+100] = metrics['closeness'].iloc[idx]
    complete['clustering'].iloc[n100:n100+100] = metrics['clustering'].iloc[idx]
    complete['globalEff'].iloc[n100:n100+100] = metrics['globalEff'].iloc[idx]
    complete['localEff'].iloc[n100:n100+100] = metrics['localEff'].iloc[idx]
    complete['nMessages'].iloc[n100:n100+100] = convergence['nMessages']

print(complete)
complete.to_csv('relationData-complete-Atlas.tsv',sep='\t',index=False)
