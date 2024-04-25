"""
Script to merge convergence results per Graph Atlas network.
- Convergence results from Snellius were dumped per run
- For analysis purposes, all convergence results should be clustered per graph.

Written by Jade Dubbeld
27/03/2024
"""

import pandas as pd
from tqdm import tqdm

settings = f'alpha1_00-beta0_00'

df = pd.read_csv('data/data-GraphAtlas.tsv',sep='\t')

# # NOTE: USE FOR SPECIFIC CASE OF ONLY USING GRAPH SIZE 7
# df = df[df['nodes']==7]

graphs = []

for j in range(len(df)):

    # generate graph ID
    name = 'G' + str(df['index'].iloc[j])
    graphs.append(name)

for graph in tqdm(graphs):

    data = pd.DataFrame(index=range(100),columns=['nMessages','meanHammingDist'])

    for i in range(100): # NOTE: LOAD CORRECT CONVERGENCE RESULTS
        run = pd.read_csv(f'results/{settings}/convergence-{graph}-run{i}.tsv', sep='\t', usecols=['nMessages'])#,'meanHammingDist'])
        # run = pd.read_csv(f'results/efficient-{settings}/convergence-{graph}-run{i}.tsv', sep='\t', usecols=['nMessages'])#,'meanHammingDist'])

        data.iloc[i] = run.iloc[0]

    # NOTE: SAVE ACCORDINGLY
    data.to_csv(f'results/{settings}/merged/convergence-{graph}.tsv', sep='\t',index=False)   
    # data.to_csv(f'results/efficient-{settings}/merged/convergence-{graph}.tsv', sep='\t',index=False)