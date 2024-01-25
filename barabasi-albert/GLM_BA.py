"""
Script to acquire contribution factor of each of the strucutral network measures to consensus formation.

Using Generalized Linear Model - multivariable regression.

Written by Jade Dubbeld
25/01/2024
"""

import pickle, numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score

first = 1
last = 20

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
structural_m1 = pickle.load(open("graphs/m=1/measures-m=1.pickle",'rb'))[first-1:last]
structural_m2 = pickle.load(open("graphs/m=2/measures-m=2.pickle",'rb'))[first-1:last]
structural_m3 = pickle.load(open("graphs/m=3/measures-m=3.pickle",'rb'))[first-1:last]
structural_m4 = pickle.load(open("graphs/m=4/measures-m=4.pickle",'rb'))[first-1:last]

# combine all consensus
y_train = np.append(consensus_m1[first-1:15],consensus_m2[first-1:15],axis=0)
y_train = np.append(y_train, consensus_m3[first-1:15], axis=0)
y_train = np.append(y_train, consensus_m4[first-1:15], axis=0)

y_test = np.append(consensus_m1[15:last],consensus_m2[15:last],axis=0)
y_test = np.append(y_test,consensus_m3[15:last],axis=0)
y_test = np.append(y_test,consensus_m4[15:last],axis=0)

# combine all network measures
X_train = np.append(structural_m1[first-1:15],structural_m2[first-1:15], axis=0)
X_train = np.append(X_train,structural_m3[first-1:15],axis=0)
X_train = np.append(X_train,structural_m4[first-1:15],axis=0)

X_test = np.append(structural_m1[15:last],structural_m2[15:last],axis=0)
X_test = np.append(X_test,structural_m3[15:last],axis=0)
X_test = np.append(X_test,structural_m4[15:last],axis=0)

# linear regression
LinReg = LinearRegression().fit(X_train, y_train)
print(LinReg.coef_)
print(LinReg.intercept_)

y_pred = LinReg.predict(X_test)
print(y_pred)
print(y_test)
r2 = r2_score(y_test, y_pred)
print(r2)
