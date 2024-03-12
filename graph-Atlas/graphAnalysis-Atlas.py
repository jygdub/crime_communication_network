"""
Script to analyze variation in network characteristics of networkx's Graph Atlas.

Written by Jade Dubbeld
12/03/2024
"""

import pandas as pd, matplotlib.pyplot as plt, numpy as np

df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

for n in range(1,8):
    subset = df.iloc[np.where(df.nodes == n)[0]]
    plt.scatter(range(0,len(subset)),subset.localEff,label=f'n={n}')
plt.xlabel('Network')
plt.ylabel('Local efficiency')
plt.title('Distribution Graph Atlas - Local efficiency')
plt.legend(bbox_to_anchor=(1,1))
plt.savefig('images/variation-localEff.png',bbox_inches='tight')