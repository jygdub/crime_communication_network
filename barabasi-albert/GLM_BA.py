"""
Script to acquire contribution factor of each of the strucutral network measures to predict consensus formation.

Using Generalized Linear Model - multiple linear regression.
- Main effects (single variables)
- Interactions (pairwise relations)

Written by Jade Dubbeld
25/01/2024
"""

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import statsmodels.formula.api as smf 

def plot_coefficients(model,vars,enlarge):
    variables = model.params[1:]
    y_pos = np.arange(len(vars))

    if enlarge:
        fig, ax = plt.subplots(figsize=(16,9))
    else:
        fig, ax = plt.subplots()

    bars = plt.bar(y_pos, variables, align='center')
    ax.bar_label(bars)

    plt.xticks(y_pos, vars, rotation=90)
    plt.xlabel('Structural network measure')
    plt.ylabel('Coefficient')
    plt.title('Predictive effects of structural network measures on consensus formation')

    return fig



data = pd.read_csv('data/measures-consensus-varying-graphs1-50.csv')

# create string to describe variables
single_OLS = 'link + degree + betweenness + closeness + clustering + transitivity + global_efficiency + local_efficiency'
link_OLS = 'link*degree + link*betweenness + link*closeness + link*clustering + link*transitivity + link*global_efficiency + link*local_efficiency'
degree_OLS = '+ degree*betweenness + degree*closeness + degree*clustering + degree*transitivity + degree*global_efficiency + degree*local_efficiency'
betweenness_OLS = '+ betweenness*closeness  + betweenness*clustering + betweenness*transitivity + betweenness*global_efficiency + betweenness*local_efficiency'
closeness_OLS = '+ closeness*clustering + closeness*transitivity + closeness*global_efficiency + closeness*local_efficiency'
clustering_OLS = '+ clustering*transitivity + clustering*global_efficiency + clustering*local_efficiency'
transitivity_OLS = '+ transitivity*global_efficiency + transitivity*local_efficiency'
globEff_OLS = '+ global_efficiency*local_efficiency'
pairwise_OLS = link_OLS + degree_OLS + betweenness_OLS + closeness_OLS + clustering_OLS + transitivity_OLS + globEff_OLS

# run multiple linear regression
model_single = smf.ols(f'consensus ~  {single_OLS}', data=data).fit()
# print(model_single.summary())

model_single_pair = smf.ols(f'consensus ~ {single_OLS} + {pairwise_OLS}', data=data).fit()
# print(model_single_pair.summary())

# set variables for plot
single_vars = ['link','degree','betweenness','closeness','clustering','transitivity','global efficiency', 'local efficiency']
pairs_link = ['link*degree','link*betweenness','link*closeness','link*clustering','link*transitivity','link*global_efficiency','link*local_efficiency']
pairs_degree = ['degree*betweenness','degree*closeness','degree*clustering','degree*transitivity','degree*global_efficiency','degree*local_efficiency']
pairs_betweenness = ['betweenness*closeness','betweenness*clustering','betweenness*transitivity','betweenness*global_efficiency','betweenness*local_efficiency']
pairs_closeness = ['closeness*clustering','closeness*transitivity','closeness*global_efficiency','closeness*local_efficiency']
pairs_clustering = ['clustering*transitivity','clustering*global_efficiency','clustering*local_efficiency']
pairs_transitivity = ['transitivity*global_efficiency','transitivity*local_efficiency']
pairs_globEff = ['global_efficiency*local_efficiency']
pairwise_vars = pairs_link + pairs_degree + pairs_betweenness + pairs_closeness + pairs_clustering + pairs_transitivity + pairs_globEff

# make plots and save
fig_single = plot_coefficients(model_single,single_vars,enlarge=False)
fig_single_pair = plot_coefficients(model_single_pair,single_vars+pairwise_vars,enlarge=True)

plt.figure(fig_single)
plt.savefig('images/multiple-linear-regression/barplot-single-effects.png',bbox_inches='tight')

plt.figure(fig_single_pair)
plt.savefig('images/multiple-linear-regression/barplot-single-pairwise-effects.png',bbox_inches='tight')