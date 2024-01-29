import pickle, pandas as pd, numpy as np

variation = True
first = 1
last = 50

if variation:
    path = f"images/efficiency/various-graphs{first}-{last}"
else:
    graph = 1
    path = f"images/efficiency/graph{graph}"

# load consensus formation for all runs on
consensus_m1 = np.array(pickle.load(open(f"{path}/simulation/consensus-rate-graphs{first}-{last}-alpha=1.0-beta=0.0-n=100-m=1-BA.pickle",'rb')))
consensus_m2 = np.array(pickle.load(open(f"{path}/simulation/consensus-rate-graphs{first}-{last}-alpha=1.0-beta=0.0-n=100-m=2-BA.pickle",'rb')))
consensus_m3 = np.array(pickle.load(open(f"{path}/simulation/consensus-rate-graphs{first}-{last}-alpha=1.0-beta=0.0-n=100-m=3-BA.pickle",'rb')))
consensus_m4 = np.array(pickle.load(open(f"{path}/simulation/consensus-rate-graphs{first}-{last}-alpha=1.0-beta=0.0-n=100-m=4-BA.pickle",'rb')))

# load structural efficiency measures for all graphs
structural_m1 = pickle.load(open("graphs/m=1/measures-m=1.pickle",'rb'))[first-1:last]
structural_m2 = pickle.load(open("graphs/m=2/measures-m=2.pickle",'rb'))[first-1:last]
structural_m3 = pickle.load(open("graphs/m=3/measures-m=3.pickle",'rb'))[first-1:last]
structural_m4 = pickle.load(open("graphs/m=4/measures-m=4.pickle",'rb'))[first-1:last]

data = pd.DataFrame(np.concatenate((structural_m1,structural_m2,structural_m3,structural_m4),axis=0),
                      columns=['link','degree','betweenness','closeness','clustering','transitivity',
                               'global_efficiency','local_efficiency'])

data['consensus'] = np.concatenate((consensus_m1,consensus_m2,consensus_m3,consensus_m4)) 
print(data)

data.to_csv('data/measures-consensus-varying-graphs1-50.csv',index=False)
