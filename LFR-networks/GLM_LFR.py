"""
Script to run multiple linear regression on LFR benchmark graph parameters vs. various network measures.

Written by Jade Dubbeld
12/02/2024
"""

import pandas as pd
import statsmodels.formula.api as smf 

data = pd.read_csv('data/all-measures.tsv',sep='\t')
print(data)


# multiple linear regression
model_degree = smf.ols(f'degreeCentrality ~  mu + average_degree', data=data).fit()
print(model_degree.summary())

model_betweenness = smf.ols(f'betweennessCentrality ~  mu + average_degree', data=data).fit()
print(model_betweenness.summary())

model_CFbetweenness = smf.ols(f'currentFlowBetweenness ~  mu + average_degree', data=data).fit()
print(model_CFbetweenness.summary())

model_closeness = smf.ols(f'closenessCentrality ~  mu + average_degree', data=data).fit()
print(model_closeness.summary())

model_clustering = smf.ols(f'clusteringCoefficient ~  mu + average_degree', data=data).fit()
print(model_clustering.summary())

model_globEff = smf.ols(f'globalEfficiency ~  mu + average_degree', data=data).fit()
print(model_globEff.summary())

model_locEff = smf.ols(f'localEfficiency ~  mu + average_degree', data=data).fit()
print(model_locEff.summary())