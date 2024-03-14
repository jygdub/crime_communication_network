"""
Old script (speedup) to simulate Deffuant-like dynamics on LFR benchmark graphs.

Written by Jade Dubbeld
17/02/2024
"""

import networkx as nx, random, numpy as np, matplotlib.pyplot as plt, pickle, time

from test import init, hamming_distance, simulate

alpha = 1.0
beta = 0.0

# # retrieve filenames from directory
# listFileNames = sorted(glob.glob(f'graphs/first-generation/tau1={exp_degree}-tau2={exp_community}-*.pickle'))

# # test files
# filename = 'graphs/test100-tau1=3.0-tau2=1.5-mu=0.1-avg_deg=5-min_comm=5-seed=0.pickle'
# filename = 'graphs/test10-tau1=3.0-tau2=1.5-mu=0.3-avg_deg=2-min_comm=2-seed=0.pickle'
# filename = 'graphs/official-generation/tau1=2.5-tau2=1.1-mu=0.45-avg_deg=25-min_comm=10-seed=99.pickle'
filename = 'graphs/first-generation/tau1=3.0-tau2=1.5-mu=0.25-avg_deg=10-min_comm=10-seed=12.pickle'


G = pickle.load(open(filename, 'rb'))
N = len(list(G))
k = 3
G = init(G) #, N, k)

print(filename)

start = time.time()
M, meanStringDifference = simulate(G,alpha=alpha,beta=beta)
end = time.time()
execution_time = end-start
print(f"execution time = {execution_time} in seconds")
print(f"total messages = {M}")

# convergence plot simulation
fig_converge = plt.figure(figsize=(13,8))
plt.plot(np.arange(0,len(meanStringDifference)), np.reshape(meanStringDifference,(-1,1)))

plt.xlabel("Total messages sent", fontsize=14)
plt.ylabel("Average string difference (Hamming distance)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.title(f"Convergence on LFR")
# plt.legend(bbox_to_anchor=(1,1))
plt.savefig(f"images/test1000-vectorize-convergence.png", bbox_inches='tight')
