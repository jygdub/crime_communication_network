"""
Script to correlate structural efficiency measures with operational efficiency measure.

Written by Jade Dubbeld
24/01/2024
"""

import pickle, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression

first = 1
last = 20
column = 0
measures = {0: "Link density", 
            1: "Degree centrality", 
            2: "Betweenness centrality",
            3: "Closeness centrality",
            4: "Clustering coefficient",
            5: "Transitivity",
            6: "Global efficiency",
            7: "Local efficiency"}

eliminate1 = True
variation = True

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
structural_m1 = pickle.load(open("graphs/m=1/measures-m=1.pickle",'rb'))
structural_m2 = pickle.load(open("graphs/m=2/measures-m=2.pickle",'rb'))
structural_m3 = pickle.load(open("graphs/m=3/measures-m=3.pickle",'rb'))
structural_m4 = pickle.load(open("graphs/m=4/measures-m=4.pickle",'rb'))

X_data = np.append(structural_m1[first-1:last,column].reshape(-1,1), structural_m2[first-1:last,column].reshape(-1,1), axis=0)
X_data = np.append(X_data, structural_m3[first-1:last,column].reshape(-1,1), axis=0)
X_data = np.append(X_data, structural_m4[first-1:last,column].reshape(-1,1), axis=0)

Y_data = np.append(consensus_m1, consensus_m2)
Y_data = np.append(Y_data, consensus_m3)
Y_data = np.append(Y_data, consensus_m4)

if eliminate1:
    start = 20          # change if necessary (leave out m=1 -> set to 20; else set to 0)
    
    for i in range(2,5):    # change start of range (leave out m=1 -> set to 2; else set to 1)
        plt.scatter(X_data[0+start:20+start],Y_data[0+start:20+start],label=f"m={i}")
        start += 20
else:
    start = 0
    
    for i in range(1,5):    # change start of range (leave out m=1 -> set to 2; else set to 1)
        plt.scatter(X_data[0+start:20+start],Y_data[0+start:20+start],label=f"m={i}")
        start += 20

if eliminate1:
    start = 20          # change if necessary (leave out m=1 -> set to 20; else set to 0)
else:
    start = 0

LinReg = LinearRegression().fit(X_data[start:],Y_data[start:])

x_test = np.linspace(min(X_data[start:]),max(X_data[start:]),100)
# predict dummy y_test data based on the logistic model
y_test = x_test * LinReg.coef_ + LinReg.intercept_
# plt.figure(fig)
plt.plot(x_test,y_test,'k--',label=f"linear regression \n(coeff={round(LinReg.coef_[0])}; intercept={round(LinReg.intercept_)})")
plt.xlabel(f"{measures[column]}")
plt.ylabel("Total of messages until consensus")
plt.ylim(0)
plt.legend(bbox_to_anchor=(1,1))
plt.title(f"Relation between consensus formation and {measures[column].lower()}")

if eliminate1:
    if variation:
        plt.savefig(f"{path}/linear-regression/{measures[column].lower().split(' ')[0]}-efficiency-graph{first}-{last}-m=[2,3,4]-alpha=1.0-beta=0.0-n=100-BA.png",
                    bbox_inches='tight')
    else:
        plt.savefig(f"{path}/linear-regression/{measures[column].lower().split(' ')[0]}-efficiency-graphs{graph}-m=[2,3,4]-alpha=1.0-beta=0.0-n=100-BA.png",
                    bbox_inches='tight')
else:
    if variation:
        plt.savefig(f"{path}/linear-regression/{measures[column].lower().split(' ')[0]}-efficiency-graph{first}-{last}-m=[1,2,3,4]-alpha=1.0-beta=0.0-n=100-BA.png",
                bbox_inches='tight')
    else:
        plt.savefig(f"{path}/linear-regression/{measures[column].lower().split(' ')[0]}-efficiency-graphs{graph}-m=[1,2,3,4]-alpha=1.0-beta=0.0-n=100-BA.png",
                    bbox_inches='tight')