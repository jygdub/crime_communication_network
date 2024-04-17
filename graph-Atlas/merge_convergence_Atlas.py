"""
Script to merge convergence results per Graph Atlas network.
- Convergence results from Snellius were dumped per run
- For analysis purposes, all convergence results should be clustered per graph.

Written by Jade Dubbeld
27/03/2024
"""

import pandas as pd
from tqdm import tqdm

# df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')
df = pd.read_csv('data/data-GraphAtlas.tsv',sep='\t')


graphs = []

for j in range(len(df)):

    # generate graph ID
    name = 'G' + str(df['index'].iloc[j])

    graphs.append(name)

for graph in tqdm(graphs):

    data = pd.DataFrame(index=range(100),columns=['nMessages','meanHammingDist'])

    for i in range(100):
        run = pd.read_csv(f'results/alpha0_50-beta0_25/convergence-{graph}-run{i}.tsv', sep='\t', usecols=['nMessages','meanHammingDist'])
        # run = pd.read_csv(f'/Volumes/Disk-Jade/CLS-data/alpha0_75-beta0_50/convergence-{graph}-run{i}.tsv', sep='\t', usecols=['nMessages','meanHammingDist'])

        data.iloc[i] = run.iloc[0]

    data.to_csv(f'results/alpha0_50-beta0_25/merged/convergence-{graph}.tsv', sep='\t',index=False)
    # data.to_csv(f'/Volumes/Disk-Jade/CLS-data/alpha0_75-beta0_50/convergence-{graph}.tsv', sep='\t',index=False)
