"""
Script to analyze variation in network characteristics of networkx's Graph Atlas.

Written by Jade Dubbeld
12/03/2024
"""

import pandas as pd, matplotlib.pyplot as plt, numpy as np, pickle

df = pd.read_csv('data-GraphAtlas.tsv',sep='\t')

# variation = plt.figure()

xyz = plt.figure()
ax_xyz = plt.axes(projection ="3d")

for n in range(1,8):
    subset = df.iloc[np.where(df.nodes == n)[0]]

    # plt.figure(variation)
    # plt.scatter(range(0,len(subset)),subset.degree,label=f'n={n}')
    
    plt.figure(xyz)
    ax_xyz.scatter3D(subset.closeness,subset.clustering,subset.globalEff,label=f'n={n}')

# plt.figure(variation)
# plt.xlabel('Network')
# plt.ylabel('Degree centrality')
# plt.title('Distribution Graph Atlas - Degree centrality')
# plt.legend(bbox_to_anchor=(1,1))
# plt.savefig('images/variation-degree.png',bbox_inches='tight')


plt.figure(xyz)
ax_xyz.set_xlabel('Closeness centrality')
ax_xyz.set_ylabel('Clustering coefficient')
ax_xyz.set_zlabel('Global efficiency')
# pickle.dump((xyz, ax_xyz), open('images/3D-degree-closeness-global.pickle', 'wb'))
plt.show()
# plt.savefig(f"images/3D-degree-closeness-global.png")