"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
from typing import Tuple
import numpy as np, pandas as pd, seaborn as sns, networkx as nx


def scatterALL(alpha: str, beta: str, draw_polynomial: bool):
    """
    Function to scatter all raw datapoints.
    - X-axis: Network metric
    - Y-axis: Timesteps

    Parameters:
    - alpha (str): Sender bias/noise value
    - beta (str): Receiver bias/noise value
    - draw_polynomial (bool): True indicates fitting a 3rd degree polynomial; False indicates no polynomial fit
    """

    settings = f'alpha{alpha}-beta{beta}'

    # plot for each network metric
    for metric in tqdm(['degree','betweenness','CFbetweenness','closeness','clustering','global','local']):
        
        # initialize booleans
        efficiency = False
        coefficient = False

        # set variables and booleans according to metric
        if metric == 'global':
            column = 'globalEff'
            efficiency = True
        elif metric == 'local':
            column = 'localEff'
            efficiency = True
        elif metric == 'clustering':
            column = 'clustering'
            coefficient = True
        else:
            column = metric

        # initilize figure
        fig,ax = plt.subplots()

        # load data
        data = pd.read_csv(f'data/meanRelationData-{settings}-Atlas.tsv', sep='\t')

        # plot from graph size n=7 to n=2 (for visualization purposes)
        for n in reversed(data['nodes'].unique()):

            # pre-determine colormap
            if  n == 2:
                color = "tab:blue"
            elif n == 3:
                color = "tab:orange"
            elif n == 4:
                color = "tab:green"
            elif n == 5:
                color = "tab:red"
            elif n == 6:
                color = "tab:purple"
            elif n == 7:
                color = "tab:pink"

            # find all data per graph size
            indices = np.where(data['nodes'] == n)[0]

            # plot data in scatterplot messages (y) against metric (x)
            ax.scatter(data[column].iloc[indices],data['nMessages'].iloc[indices],color=color,alpha=0.3)

            # fit a 3rd degree polynomial if chosen (and not for n=2)
            if n != 2 and draw_polynomial:
                p = np.poly1d(np.polyfit(data[column].iloc[indices],data['nMessages'].iloc[indices],3))
                t = np.linspace(min(data[column]), max(data[column]), 250)
                ax.plot(t,p(t),color)

        # decorate plot
        handles = [
            plt.scatter([], [], color=c, label=l)
            for c, l in zip("tab:blue tab:orange tab:green tab:red tab:purple tab:pink".split(), "n=2 n=3 n=4 n=5 n=6 n=7".split())
        ]

        ax.legend(handles=handles)

        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency")
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient")
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality")

        ax.set_ylabel("Convergence time")
        ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')}')
        # plt.show()

        # save figure with corresponding filename
        if draw_polynomial:
            fig.savefig(f"images/relations/{settings}/all-datapoints/relation-{metric}-convergence-polynomial.png",bbox_inches='tight')
        else:
            fig.savefig(f"images/relations/{settings}/all-datapoints/relation-{metric}-convergence-scatter.png",bbox_inches='tight')

        plt.close(fig)


def scatterMEAN(alpha: str, beta: str, draw_polynomial: bool):
    """
    Function to scatter all mean datapoints, where the mean is taken per graph.
    - X-axis: Network metric
    - Y-axis: Timesteps

    Parameters:
    - alpha (str): Sender bias/noise value
    - beta (str): Receiver bias/noise value
    - draw_polynomial (bool): True indicates fitting a 3rd degree polynomial; False indicates no polynomial fit
    """

    # set paths
    settings = f'alpha{alpha}-beta{beta}'                       
    images_path = f'images/relations/{settings}/averaged-convergence'  

    # plots for all 7 metrics
    for metric in tqdm(['degree','betweenness','CFbetweenness','closeness','clustering','global','local']):
        
        # set initial booleans
        efficiency = False
        coefficient = False

        # open figure
        fig,ax = plt.subplots()

        # conditional column assignment and boolean flip
        if metric == 'global':
            column = 'globalEff'
            efficiency = True
        elif metric == 'local':
            column = 'localEff'
            efficiency = True
        elif metric == 'clustering':
            column = 'clustering'
            coefficient = True
        else:
            column = metric

        # read data for corresponding parameter settings
        data = pd.read_csv(f'data/meanRelationData-{settings}-Atlas.tsv', sep='\t')
        
        # scatterplot all mean convergences per graph size (from n=7 to n=2)
        for i, index in enumerate(reversed(data['index'].unique())):
            n = data['nodes'].iloc[len(data)-i*100-1]

            # pre-determine colormap
            if  n == 2:
                color = "tab:blue"
            elif n == 3:
                color = "tab:orange"
            elif n == 4:
                color = "tab:green"
            elif n == 5:
                color = "tab:red"
            elif n == 6:
                color = "tab:purple"
            elif n == 7:
                color = "tab:pink"

            # find all data per graph size
            indices = np.where(data['index'] == index)[0]

            # plot data in scatterplot messages (y) against metric (x)
            ax.scatter(np.mean(data[column].iloc[indices]),np.mean(data['nMessages'].iloc[indices]),color=color, alpha=0.5)


        # fit 3rd polynomial to data per graph size (from n=7 to n=2)
        for n in reversed(data['nodes'].unique()):
            
            # pre-determine colormap
            if  n == 2:
                color = "tab:blue"
            elif n == 3:
                color = "tab:orange"
            elif n == 4:
                color = "tab:green"
            elif n == 5:
                color = "tab:red"
            elif n == 6:
                color = "tab:purple"
            elif n == 7:
                color = "tab:pink"

            indices = np.where(data['nodes'] == n)[0]

            # fit polynomial if desired
            if n != 2 and draw_polynomial:
                p = np.poly1d(np.polyfit(data[column].iloc[indices],data['nMessages'].iloc[indices],3))
                t = np.linspace(min(data[column]), max(data[column]), 250)
                ax.plot(t,p(t),color)
        
        # decorate figure
        handles = [
            plt.scatter([], [], color=c, label=l)
            for c, l in zip("tab:blue tab:orange tab:green tab:red tab:purple tab:pink".split(), "n=2 n=3 n=4 n=5 n=6 n=7".split())
        ]

        ax.legend(handles=handles)
        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency")
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient")
        elif metric == 'CFbetweenness':
            ax.set_xlabel(f"Current flow betweenness centrality")
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality")
        ax.set_ylabel("Convergence time")
        ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')}')
        
        # plt.show()

        # save figure with corresponding filename
        if draw_polynomial:
            fig.savefig(f"{images_path}/relation-{metric}-convergence-mean-polynomial.png",bbox_inches='tight')
        else:
            fig.savefig(f"{images_path}/relation-{metric}-convergence-mean-scatter.png",bbox_inches='tight')
        
        plt.close(fig)


def polynomialFit_compareSingleNoise(changing: str):
    """
    Function to compare model parameter settings per changing 'alpha' or 'beta' noise using 
    the 3rd degree polynomial fit through mean data per graph size.

    Parameters:
    - changing (str): Indicator for comparing varying 'alpha' or 'beta'
    """

    # plot for each network metric per graph size
    for metric,n in product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],
                            [3,4,5,6,7]):
        
        # initialize
        efficiency = False
        coefficient = False

        # set variables and booleans according to network metric
        if metric == 'global':
            column = 'globalEff'
            efficiency = True
        elif metric == 'local':
            column = 'localEff'
            efficiency = True
        elif metric == 'clustering':
            column = 'clustering'
            coefficient = True
        else:
            column = metric

        # initialize figure
        fig,ax = plt.subplots()

        # set noise parameters according to varying noise
        if changing == 'beta':
            values = ['00','25','50']
        elif changing == 'alpha':
            values = ['1_00','0_75','0_50']

        # plot 3rd degree polynomial fit in a single figure (for varying noise)
        for x in values:

            # load corresponding mean data
            if changing == 'beta':
                data = pd.read_csv(f'data/meanRelationData-alpha1_00-beta0_{x}-Atlas.tsv', sep='\t')
            elif changing == 'alpha':
                data = pd.read_csv(f'data/meanRelationData-alpha{x}-beta0_00-Atlas.tsv', sep='\t')

            # fit polynomial if desired
            p = np.poly1d(np.polyfit(data[column][data['nodes']==n],data['nMessages'][data['nodes']==n],3))
            t = np.linspace(min(data[column]), max(data[column]), 250)

            # set appropriate label
            if changing == 'beta':
                ax.plot(t,p(t),label=fr"$\beta$={x.replace('_','.')}")
            elif changing == 'alpha':
                ax.plot(t,p(t),label=fr"$\alpha$={x.replace('_','.')}")

        # decorate figure
        ax.legend(bbox_to_anchor=(1,1))

        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency")
            ax.set_title(f"Relation between structural and operational efficiency ({metric})")
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient")
            ax.set_title(f"Relation between {metric} coefficient and consensus formation")
        elif metric == 'CFbetweenness':
            ax.set_xlabel(f"Current flow betweenness centrality")
            ax.set_title(f"Relation between current flow betweenness centrality and consensus formation")
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality")
            ax.set_title(f"Relation between {metric} centrality and consensus formation")
        ax.set_ylabel("Convergence time")

        # plt.show()

        # save figure with corresponding filename
        if changing == 'beta':
            ax.set_title(fr'$\alpha$=1.00 & n={n}')
            fig.savefig(f"images/relations/varyingBeta-alpha1_00-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')
        
        elif changing == 'alpha':
            ax.set_title(fr'$\beta$=0.00 & n={n}')
            fig.savefig(f"images/relations/varyingAlpha-beta0_00-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')
        
        plt.close(fig)

def polynomialFit_compareALL():
    """
    Function to compare ALL model parameter settings using the 3rd degree polynomial fit 
    through mean data per graph size.
    """

    # plot for each network metric per graph size
    for metric,n in product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],
                            [3,4,5,6,7]):

        # initialize
        efficiency = False
        coefficient = False

        # set variables and booleans according to network measure
        if metric == 'global':
            column = 'globalEff'
            efficiency = True
        elif metric == 'local':
            column = 'localEff'
            efficiency = True
        elif metric == 'clustering':
            column = 'clustering'
            coefficient = True
        else:
            column = metric

        # initialize figure
        fig,ax = plt.subplots()

        # plot 3rd degree polynomial fit for all available parameter settings
        for alpha, beta in product(['1_00','0_75','0_50'],['0_00','0_25','0_50']):
            if alpha == '0_75' and (beta == '0_25' or beta == '0_50'):
                continue
            elif alpha == '0_50' and (beta == '0_25' or beta == '0_50'):
                continue

            # load corresponding data
            data = pd.read_csv(f'data/meanRelationData-alpha{alpha}-beta{beta}-Atlas.tsv', sep='\t')

            # plot polynomial fit
            p = np.poly1d(np.polyfit(data[column][data['nodes']==n],data['nMessages'][data['nodes']==n],3))
            t = np.linspace(min(data[column]), max(data[column]), 250)
            ax.plot(t,p(t),label=fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')}")

        # decorate figure
        ax.legend(bbox_to_anchor=(1,1))

        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency")
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient")
        elif metric == 'CFbetweenness':
            ax.set_xlabel(f"Current flow betweenness centrality")
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality")

        ax.set_ylabel("Convergence time")
        ax.set_title(f"Varying alpha and beta parameters (n={n})")

        # save figure
        fig.savefig(f"images/relations/combined-parameters-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')

        plt.close(fig)


def my_floor(a: float, precision: int = 0) -> np.float64:
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def violin_per_params(alpha: float, beta: float, perN: bool, fit: str, without2: bool, metric: str = None, fixed: bool = False):
    """ 
    Violinplot distribution of convergence time per metric per bin (size=0.1).

    Parameters:
    - alpha (float): Alpha noise
    - beta (float): Beta noise
    - perN (bool): False if combined data; True if per graph size
    - fit (str): Fitting linear fit ('linear') or exponential fit ('exponential) or no fit ('none')
    - without2 (bool): Indicator to remove graph size n=2 from data
    - metric (str): Choose a single metric to apply to figures (otherwise all networks are generated)
    - fixed (bool): False for random initialization; True for fixed pre-defined initialization
    """
    
    # set paths
    settings = f'alpha{alpha}-beta{beta}'   
    images_path = f'images/relations/{settings}/all-datapoints'  
    
    if fixed:
        settings = f'fixed-alpha{alpha}-beta{beta}'  
        images_path = f'images/relations/{settings}'                  
    
    # load all data
    data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')
    
    # eliminate graph size n=2, if desired
    if without2:
        data = data.drop(range(0,100))

    # set iterative for for-loop
    iterative = None
    if metric != None and not perN:
        iterative = product([metric], [0])
    elif metric != None and perN:
        iterative = product([metric], [3,4,5,6,7])
    elif perN:
        iterative = product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[3,4,5,6,7])
    else:
        iterative = product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[0])
    
    # generate plots according to iterative
    for metric, n in tqdm(iterative):

        # set initials
        efficiency = False
        coefficient = False

        # set variables and booleans according to metric
        if metric == 'global':
            column = 'globalEff'
            efficiency = True
        elif metric == 'local':
            column = 'localEff'
            efficiency = True
        elif metric == 'clustering':
            column = 'clustering'
            coefficient = True
        else:
            column = metric

        # select correct data according to desired iterative
        if perN:
            nData = data[data['nodes']==n] 
        else:
            nData = data

        # initialize figure
        fig,ax = plt.subplots(figsize=(13,8))
        max_density = []

        # plot violins per binsize = 0.1 in range 0 to 1
        for th in np.arange(0.0,1.0,0.1):

            # set subset accordingly
            if th == 0.9:
                subset = nData[nData[column]<=th+0.1][nData[column]>=th]
            else:
                subset = nData[nData[column]<th+0.1][nData[column]>=th]

            # skip empty subsets
            if not subset.empty:

                # keep track of higest probability density
                pdf = plt.figure()
                density = subset['nMessages'].plot.kde().get_lines()[0].get_xydata()
                max_density.append(density[np.argmax(density[:,1])][0])
                plt.close(pdf)

                # scatter plot all raw data per binsize
                ax.scatter(subset[column],subset['nMessages'],color='lightgrey',alpha=0.1)

                # plot violin per binsize
                plots = ax.violinplot(subset['nMessages'],positions=[th+0.05],widths=[0.1])

                # style violinplot
                for vp in plots['bodies']:
                    vp.set_facecolor("tab:blue")
                    vp.set_edgecolor("black")
                
                # more styling violinplot
                for i, partname in enumerate(('cbars', 'cmins', 'cmaxes')): 
                    vp = plots[partname]
                    vp.set_edgecolor("tab:blue")
                    vp.set_linewidth(1)

        # define x-axis
        min_th = my_floor(min(nData[column]),1)
        Xaxis = np.linspace(min_th+0.05,min_th+0.05+(len(max_density)-1.)*0.1,len(max_density))


        if fit == 'linear':
            
            # linear polyfit
            coef_lin = np.polyfit(Xaxis, max_density, 1)
            poly1d_fn_lin = np.poly1d(coef_lin) 

            # plot maximum probability density
            ax.plot(Xaxis, max_density, 'ro', label='Maximum probability density')
            ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), poly1d_fn_lin(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05)), 
                    'g--', 
                    label=f'C = {round(coef_lin[0],2)} * GE + {round(coef_lin[1],2)}')
        
        elif fit == 'exponential':

            # exponential polyfit
            coef_exp = np.polyfit(Xaxis, np.log(max_density), 1)
            poly1d_fn_exp = np.poly1d(coef_exp) 

            # plot maximum probability density
            ax.plot(Xaxis, max_density, 'ro', label='Maximum probability density')
            ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), 
                    np.exp(poly1d_fn_exp(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05))), 
                    'k--', 
                    label=f'C = exp({round(coef_exp[0],2)} * GE + {round(coef_exp[1],2)})')
        
        elif fit == 'none':
            # plot maximum probability density
            ax.plot(Xaxis, max_density, 'ro', label='Maximum probability density')

        # decorate figure
        plt.legend(loc='upper right',fontsize=10)

        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency",fontsize=16)
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient",fontsize=16)
        elif metric == 'CFbetweenness':
            ax.set_xlabel(f"Current flow betweenness centrality",fontsize=16)
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality",fontsize=16)

        ax.set_ylabel("Convergence time",fontsize=16)

        if perN:
            ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n}',fontsize=16)
        else:
            ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n$\in${{3,4,5,6,7}}",fontsize=16)

        # set log scale on y-axis (visualization purposes)
        ax.set_yscale("log")
        plt.tick_params(axis='both', which='major', labelsize=16)

        # plt.show()

        #######################################################
        # NOTE: CHANGE FILENAME ACCORDINGLY 
        # (add LOG if log-scale; add lineFit for linear line or expFit for exponential
        # change allN to withoutN=2 if applied)
        # fig.savefig(f"{images_path}/expFit-LOGdistribution-n={n}-convergence-per-{metric}-violin.png",bbox_inches='tight')
        fig.savefig(f"{images_path}/LOGdistribution-withoutN=2-convergence-per-{metric}-violin.png",bbox_inches='tight')
        plt.close(fig)


def hist_per_violin(alpha: float, beta: float, perN: bool):
    """ 
    Histogram distribution of convergence time per metric per bin/violin (size=0.1).

    Parameters:
    - alpha (float): Alpha noise
    - beta (float): Beta noise
    - perN (bool): False if combined data; True if per graph size
    """

    # set paths
    settings = f'alpha{alpha}-beta{beta}'                       
    images_path = f'images/relations/{settings}/all-datapoints'  

    # load data
    data = pd.read_csv(f'data/relationData-alpha{alpha}-beta{beta}-Atlas.tsv', sep='\t')

    # set iterative for for-loop
    iterative = None
    if perN:
        iterative = product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[3,4,5,6,7])
    else:
        iterative = product(['degree','betweenness','CFbetweenness','closeness','clustering','global','local'],[0])
    
    # generate plots according to iterative
    for metric, n in tqdm(iterative):

        # initialize booleans
        efficiency = False
        coefficient = False

        # set variables and booleans according to network metric
        if metric == 'global':
            column = 'globalEff'
            efficiency = True
        elif metric == 'local':
            column = 'localEff'
            efficiency = True
        elif metric == 'clustering':
            column = 'clustering'
            coefficient = True
        else:
            column = metric

        # select correct data according to desired iterative
        if perN:
            nData = data[data['nodes']==n] 
        else:
            nData = data

        counter = 0

        # plot histogram per binsize = 0.1 in range 0 to 1
        for th in np.arange(0.0,1.0,0.1):

            # choose subset accordingly
            if th == 0.9:
                subset = nData[nData[column]<=th+0.1][nData[column]>=th]
            else:
                subset = nData[nData[column]<th+0.1][nData[column]>=th]

                # skip empty subset
                if subset.empty:
                    continue

                counter += 1

                # plot histogram distribution per bin (histogram in binsize 100)
                fig,ax = plt.subplots(figsize=(13,8))
                ax.hist(subset['nMessages'],bins=100)

                # decorate plot
                if efficiency:
                    ax.set_xlabel(f"{metric.capitalize()} efficiency",fontsize=16)
                elif coefficient:
                    ax.set_xlabel(f"{metric.capitalize()} coefficient",fontsize=16)
                elif metric == 'CFbetweenness':
                    ax.set_xlabel(f"Current flow betweenness centrality",fontsize=16)
                else:
                    ax.set_xlabel(f"{metric.capitalize()} centrality",fontsize=16)

                ax.set_ylabel("Convergence time",fontsize=16)

                if perN:
                    ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n={n} | violin {counter}',fontsize=16)
                else:
                    ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} for all n's | violin {counter}",fontsize=16)
                        
                plt.tick_params(axis="both",which="major",labelsize=16)
                # plt.show()

                #######################################################
                # NOTE: CHANGE FILENAME ACCORDINGLY
                # fig.savefig(f"{images_path}/histograms-per-violin/histDistribution-n={n}-convergence-per-{metric}-violin{counter}.png",bbox_inches='tight')
                fig.savefig(f"{images_path}/histograms-per-violin/histDistribution-allN-convergence-per-{metric}-violin{counter}.png",bbox_inches='tight')
                #######################################################

                plt.close(fig)


def violin_noiseEffect(fixed_param: str, varying_param: list, variable: str, metric: str, fit: str, without2: bool):
    """ 
    Violinplot distribution of convergence time per metric per bin (size=0.1).

    Parameters:
    - fixed_param (str): value of fixed parameter (alpha or beta)
    - varying_param (list): list of values of varying paramter (alpha or beta)
    - variable (str): 'alpha' if alpha varying noise parameter; 'beta' if beta varying noise parameter
    - metric (str): network metric
    - fit (str): Fitting linear fit ('linear') or exponential fit ('exponential) or no fit ('none')
    - without2 (bool): Indicator to remove graph size n=2 from data
    """

    # initials
    alphas = []
    betas = []
    alpha = ''
    beta = ''

    # set noise values as indicated
    if variable == 'alpha':
        alphas = varying_param
        beta = fixed_param
    else:
        betas = varying_param
        alpha = fixed_param

    print(metric)
    thresholds = []

    # set thresholds according to metric
    if metric == 'degree':
        thresholds = np.linspace(0.2,0.9,8)  # degree
    elif metric == 'betweenness' or metric == 'CFbetweenness':
        thresholds = np.linspace(0.0,0.3,4)  # betweenness & CF betweenness | NOTE NOTE CFBETWEENNESS NOT WORKING ???
    elif metric == 'closeness':   
        thresholds = np.linspace(0.3,0.9,7)  # closeness
    elif metric == 'global':
        thresholds = np.linspace(0.5,0.9,5) # global efficiency
    elif metric == 'local' or metric == 'clustering':
        thresholds = np.linspace(0.0,0.9,10)  # clustering & local efficiency

    # set path
    images_path = 'images/relations'

    # initialize booleans
    efficiency = False
    coefficient = False

    # set variables and booleans according to metric
    if metric == 'global':
        column = 'globalEff'
        efficiency = True
    elif metric == 'local':
        column = 'localEff'
        efficiency = True
    elif metric == 'clustering':
        column = 'clustering'
        coefficient = True
    else:
        column = metric

    counter = 0

    # plot violins per threshold binsize 0.1 considering varying noise effect
    for th in tqdm(thresholds):

        # set initials
        fig,ax = plt.subplots(figsize=(13,8))
        counter += 1
        max_density = []

        colors = []
        if variable == 'beta':
            colors = ['maroon','forestgreen','indigo']
        elif variable == 'alpha':
            colors = ['darkorange','mediumblue','mediumvioletred']

        ########################################
        # NOTE: SET CORRECT FOR-LOOP
        # for i, alpha in enumerate(alphas):
        for i, beta in enumerate(betas):
        ########################################
            
            # set path
            settings = f'alpha{alpha}-beta{beta}'                       

            # load data
            data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')
            
            # remove graph size n=2 from data, if indicated
            if without2:
                data = data.drop(range(0,100))

            nData = data

            if th == 0.9:
                subset = nData[nData[column]<=th+0.1][nData[column]>=th]
            else:
                subset = nData[nData[column]<th+0.1][nData[column]>=th]
        
            # keep track of higest probability density
            pdf = plt.figure()
            density = subset['nMessages'].plot.kde().get_lines()[0].get_xydata()
            max_density.append(density[np.argmax(density[:,1])][0])
            # plt.show()
            plt.close(pdf)

            plots = ax.violinplot(subset['nMessages'],positions=[np.linspace(0.,4.,5)[i+1]])
        
            # style violinplot
            for vp in plots['bodies']:
                vp.set_facecolor(colors[i])
                vp.set_edgecolor("black")
            
            # more styling violinplot
            for _, partname in enumerate(('cbars', 'cmins', 'cmaxes')): 
                vp = plots[partname]
                vp.set_edgecolor(colors[i])
                vp.set_linewidth(1)

        labels = []

        # decorate for varying alphas, else varying betas
        if variable == 'alpha':

            labels = [fr"$\alpha$=1.00", fr"$\alpha$=0.75", fr"$\alpha$=0.50"]

            if efficiency:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} & n$\in${{3,4,5,6,7}} | {metric.capitalize()} efficiency {round(th,2)}-{round(th+0.1,2)}",fontsize=16)
            elif coefficient:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} & n$\in${{3,4,5,6,7}} | {metric.capitalize()} coefficient {round(th,2)}-{round(th+0.1,2)}",fontsize=16)
            elif metric == 'CFbetweenness':
                ax.set_title(fr"$\beta$={beta.replace('_','.')} & n$\in${{3,4,5,6,7}} | Current flow betweenness centrality {round(th,2)}-{round(th+0.1,2)}",fontsize=16)
            else:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} & n$\in${{3,4,5,6,7}} | {metric.capitalize()} centrality {round(th,2)}-{round(th+0.1,2)}",fontsize=16)
        else:

            labels = [fr"$\beta$=0.00", fr"$\beta$=0.25", fr"$\beta$=0.50"]        

            if efficiency:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & n$\in${{3,4,5,6,7}} | {metric.capitalize()} efficiency {round(th,2)}-{round(th+0.1,2)}",fontsize=16)
            elif coefficient:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & n$\in${{3,4,5,6,7}} | {metric.capitalize()} coefficient {round(th,2)}-{round(th+0.1,2)}",fontsize=16)
            elif metric == 'CFbetweenness':
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & n$\in${{3,4,5,6,7}} | Current flow betweenness centrality {round(th,2)}-{round(th+0.1,2)}",fontsize=16)
            else:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & n$\in${{3,4,5,6,7}} | {metric.capitalize()} centrality {round(th,2)}-{round(th+0.1,2)}",fontsize=16)

        # set x-axis
        Xaxis = np.arange(1, len(labels) + 1)

        if fit == 'linear':

            # fit linear
            coef = np.polyfit(Xaxis, max_density, 1)
            poly1d_fn = np.poly1d(coef) 

            ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), poly1d_fn(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05)), 'g--', label=f'{round(coef[0],2)} x + {round(coef[1],2)}')

        elif fit == 'exponential':

            # fit exponential
            coef_exp = np.polyfit(Xaxis, np.log(max_density), 1)
            poly1d_fn_exp = np.poly1d(coef_exp) 

            ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), 
                    np.exp(poly1d_fn_exp(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05))), 
                    'k--') 
                    # label=f'ln(y)={round(coef_exp[0],2)} x + {round(coef_exp[1],2)}')

        # plot maximum probability density
        ax.plot(Xaxis, max_density, 'ro', label='Maximum probability density')

        # decorate figure
        # plt.legend(bbox_to_anchor=(1,1),fontsize=10)

        ax.set_xticks(Xaxis, labels=labels)
        ax.set_xlim(0., len(labels) + 1.)

        ax.set_ylabel("Convergence time",fontsize=16)
        ax.set_yscale("log")

        plt.tick_params(axis="both",which="major",labelsize=16)

        # plt.show()

        #######################################################
        # NOTE: add LOG for regular log scale; add linePlot for linear fit; change allN to withoutN=2 if applied
        if variable == 'alpha':
            fig.savefig(f"{images_path}/LOG-noiseEffect-varyingAlpha-beta={beta}-withoutN=2-{metric}-violin{counter}.png",bbox_inches='tight')
            # fig.savefig(f"{images_path}/linePlot-noiseEffect-varyingAlpha-allN-{metric}-violin{counter}.png",bbox_inches='tight')
            # fig.savefig(f"{images_path}/expPlot-noiseEffect-varyingAlpha-beta={beta}-withoutN=2-{metric}-violin{counter}.png",bbox_inches='tight')
        elif variable == 'beta':
            fig.savefig(f"{images_path}/LOG-noiseEffect-varyingBeta-alpha={alpha}-withoutN=2-{metric}-violin{counter}.png",bbox_inches='tight')
            # fig.savefig(f"{images_path}/linePlot-noiseEffect-varyingBeta-allN-{metric}-violin{counter}.png",bbox_inches='tight')
            # fig.savefig(f"{images_path}/expPlot-noiseEffect-varyingBeta-alpha={alpha}-withoutN=2-{metric}-violin{counter}.png",bbox_inches='tight')
        #######################################################

        plt.close(fig)


def summary_noiseEffect(alphas: list, betas: list, vary: str, without2: bool):
    """ 
    Summary plot of noise effect on time to consensus using random sampling in communication framework.

    Parameters:
    - alphas (list): list of alpha noise values
    - betas (list): list of beta noise values
    - vary (str): 'alpha' if alpha varying noise parameter; 'beta' if beta varying noise parameter
    - without2 (bool): Indicator to remove graph size n=2 from data
    """

    # NOTE: CHANGE ACCORDING TO VARYING PARAMETER (choose fixed parameter)
    for alpha in alphas: 
    # for beta in betas:
        fig, ax = plt.subplots(figsize=(13,8))

        colors = []
        if vary == 'beta':
            colors = ['maroon','forestgreen','indigo']
        elif vary == 'alpha':
            colors = ['darkorange','mediumblue','mediumvioletred']

        # NOTE: CHANGE ACCORDING TO VARYING PARAMETER (choose varying parameter)
        for i, beta in enumerate(betas):
        # for i, alpha in enumerate(alphas):

            # set paths
            settings = f'alpha{alpha}-beta{beta}'   

            # load all data
            data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')
            
            # eliminate graph size n=2, if desired
            if without2:
                data = data.drop(range(0,100))

            ax.scatter(x=data['globalEff'],y=data['nMessages'],c=colors[i],alpha=0.2)

        handles = []
        if vary == 'beta':
            handles = [
                plt.scatter([], [], color=c, label=l)
                for c, l in zip("maroon forestgreen indigo".split(), fr"$\beta$=0.00 $\beta$=0.25 $\beta$=0.50".split())
            ]
        elif vary == 'alpha':
            handles = [
                plt.scatter([], [], color=c, label=l)
                for c, l in zip("darkorange mediumblue mediumvioletred".split(), fr"$\alpha$=1.00 $\alpha$=0.75 $\alpha$=0.50".split())
            ]

        ax.legend(handles=handles,fontsize=14,loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_xlabel("Global efficiency",fontsize=16)
        ax.set_ylabel("Convergence time",fontsize=16)

        if vary == 'beta':
            ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & n$\in${{3,4,5,6,7}}",fontsize=16)
        elif vary == 'alpha':
            ax.set_title(fr"$\beta$={beta.replace('_','.')} & n$\in${{3,4,5,6,7}}",fontsize=16)  
        
        plt.tick_params(axis="both",which="major",labelsize=16)
        
        ax.set_yscale("log")

        if vary == 'beta':
            fig.savefig(f"images/relations/noiseEffect/summaryNoise-alpha={alpha}-varyingBeta.png",bbox_inches="tight")
        elif vary == 'alpha':
            fig.savefig(f"images/relations/noiseEffect/summaryNoise-beta={beta}-varyingAlpha.png",bbox_inches="tight")
        
        plt.show()

        plt.close(fig)


def hellinger(p: np.ndarray, q: np.ndarray) -> np.float64:
    """
    Function to compute Hellinger distance between two probability distributions.

    Parameters:
    - p (np.ndarray): First probability distribution in comparison
    - q (np.ndarray): Second probability distribution in comparison

    Returns:
    - (np.float64): Computed Hellinger distance
    """
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def quantifyNoiseDifference():
    """
    Function to quantify difference in alpha- and beta-noise effect using baricenter and spread computation.
    """

    # load all data
    a1_00b0_00 = pd.read_csv(f'data/relationData-alpha1_00-beta0_00-Atlas.tsv', sep='\t')
    a1_00b0_25 = pd.read_csv(f'data/relationData-alpha1_00-beta0_25-Atlas.tsv', sep='\t')
    a1_00b0_50 = pd.read_csv(f'data/relationData-alpha1_00-beta0_50-Atlas.tsv', sep='\t')
    a0_75b0_00 = pd.read_csv(f'data/relationData-alpha0_75-beta0_00-Atlas.tsv', sep='\t')
    a0_75b0_25 = pd.read_csv(f'data/relationData-alpha0_75-beta0_25-Atlas.tsv', sep='\t')
    a0_75b0_50 = pd.read_csv(f'data/relationData-alpha0_75-beta0_50-Atlas.tsv', sep='\t')
    a0_50b0_00 = pd.read_csv(f'data/relationData-alpha0_50-beta0_00-Atlas.tsv', sep='\t')
    a0_50b0_25 = pd.read_csv(f'data/relationData-alpha0_50-beta0_25-Atlas.tsv', sep='\t')
    a0_50b0_50 = pd.read_csv(f'data/relationData-alpha0_50-beta0_50-Atlas.tsv', sep='\t')
    
    # eliminate graph size n=2
    a1_00b0_00 = a1_00b0_00.drop(range(0,100))
    a1_00b0_25 = a1_00b0_25.drop(range(0,100))
    a1_00b0_50 = a1_00b0_50.drop(range(0,100))

    a0_75b0_00 = a0_75b0_00.drop(range(0,100))
    a0_75b0_25 = a0_75b0_25.drop(range(0,100))
    a0_75b0_50 = a0_75b0_50.drop(range(0,100))

    a0_50b0_00 = a0_50b0_00.drop(range(0,100))
    a0_50b0_25 = a0_50b0_25.drop(range(0,100))
    a0_50b0_50 = a0_50b0_50.drop(range(0,100))

    arr1 = np.array([list(a1_00b0_00['globalEff']),list(a1_00b0_00['nMessages'])])
    arr2 = np.array([list(a1_00b0_25['globalEff']),list(a1_00b0_25['nMessages'])])
    arr3 = np.array([list(a1_00b0_50['globalEff']),list(a1_00b0_50['nMessages'])])

    arr4 = np.array([list(a0_75b0_00['globalEff']),list(a0_75b0_00['nMessages'])])
    arr5 = np.array([list(a0_75b0_25['globalEff']),list(a0_75b0_25['nMessages'])])
    arr6 = np.array([list(a0_75b0_50['globalEff']),list(a0_75b0_50['nMessages'])])

    arr7 = np.array([list(a0_50b0_00['globalEff']),list(a0_50b0_00['nMessages'])])
    arr8 = np.array([list(a0_50b0_25['globalEff']),list(a0_50b0_25['nMessages'])])
    arr9 = np.array([list(a0_50b0_50['globalEff']),list(a0_50b0_50['nMessages'])])

    # calculate baricenters
    bc1 = np.mean(arr1, axis=0)
    bc2 = np.mean(arr2, axis=0)
    bc3 = np.mean(arr3, axis=0)

    bc4 = np.mean(arr4, axis=0)
    bc5 = np.mean(arr5, axis=0)
    bc6 = np.mean(arr6, axis=0)

    bc7 = np.mean(arr7, axis=0)
    bc8 = np.mean(arr8, axis=0)
    bc9 = np.mean(arr9, axis=0)

    # calculate the distance between baricenters
    dist1_2=np.linalg.norm(bc1-bc3)
    dist2_3=np.linalg.norm(bc2-bc3)

    dist4_5=np.linalg.norm(bc4-bc5)
    dist5_6=np.linalg.norm(bc5-bc6)

    dist7_8=np.linalg.norm(bc7-bc8)
    dist8_9=np.linalg.norm(bc8-bc9)

    dist1_4=np.linalg.norm(bc1-bc4)
    dist4_7=np.linalg.norm(bc4-bc7)

    dist2_5=np.linalg.norm(bc2-bc5)
    dist5_8=np.linalg.norm(bc5-bc8)

    dist3_6=np.linalg.norm(bc3-bc6)
    dist6_9=np.linalg.norm(bc6-bc9)

    print("baricenter distance between distribution 1 and distribution 2 =", dist1_2)
    print("baricenter distance between distribution 2 and distribution 3 =", dist2_3)
    print ("\n")
    print("baricenter distance between distribution 4 and distribution 5 =", dist4_5)
    print("baricenter distance between distribution 5 and distribution 6 =", dist5_6)
    print ("\n")
    print("baricenter distance between distribution 7 and distribution 8 =", dist7_8)
    print("baricenter distance between distribution 8 and distribution 9 =", dist8_9)
    print ("\n")
    print("baricenter distance between distribution 1 and distribution 4 =", dist1_4)
    print("baricenter distance between distribution 4 and distribution 7 =", dist4_7)
    print ("\n")
    print("baricenter distance between distribution 2 and distribution 5 =", dist2_5)
    print("baricenter distance between distribution 5 and distribution 8 =", dist5_8)
    print ("\n")
    print("baricenter distance between distribution 3 and distribution 6 =", dist3_6)
    print("baricenter distance between distribution 6 and distribution 9 =", dist6_9)
    print ("\n")
    print ("\n")

    #calculate the spread of the distributions, e.g. their standard deviation
    stdev1 = np.std(arr1)
    stdev2 = np.std(arr2)
    stdev3 = np.std(arr3)

    stdev4 = np.std(arr4)
    stdev5 = np.std(arr5)
    stdev6 = np.std(arr6)

    stdev7 = np.std(arr7)
    stdev8 = np.std(arr8)
    stdev9 = np.std(arr9)

    # calculate the distance between spread (measured in standard deviation)
    spread1_2 = np.abs(stdev1-stdev2)
    spread2_3 = np.abs(stdev2-stdev3)

    spread4_5 = np.abs(stdev4-stdev5)
    spread5_6 = np.abs(stdev5-stdev6)

    spread7_8 = np.abs(stdev7-stdev8)
    spread8_9 = np.abs(stdev8-stdev9)

    spread1_4 = np.abs(stdev1-stdev4)
    spread4_7 = np.abs(stdev4-stdev7)

    spread2_5 = np.abs(stdev2-stdev5)
    spread5_8 = np.abs(stdev5-stdev8)

    spread3_6 = np.abs(stdev3-stdev6)
    spread6_9 = np.abs(stdev6-stdev9)

    print("spread distance between distribution 1 and distribution 2 = ", spread1_2)
    print("spread distance between distribution 2 and distribution 3 = ", spread2_3)
    print ("\n")

    print("spread distance between distribution 4 and distribution 5 = ", spread4_5)
    print("spread distance between distribution 5 and distribution 6 = ", spread5_6)
    print ("\n")

    print("spread distance between distribution 7 and distribution 8 = ", spread7_8)
    print("spread distance between distribution 8 and distribution 9 = ", spread8_9)
    print ("\n")

    print("spread distance between distribution 1 and distribution 4 = ", spread1_4)
    print("spread distance between distribution 4 and distribution 7 = ", spread4_7)
    print ("\n")

    print("spread distance between distribution 2 and distribution 5 = ", spread2_5)
    print("spread distance between distribution 5 and distribution 8 = ", spread5_8)
    print ("\n")

    print("spread distance between distribution 3 and distribution 6 = ", spread3_6)
    print("spread distance between distribution 6 and distribution 9 = ", spread6_9)
    print ("\n")

    # fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(3,3,figsize=(10,8))
    # # fig = plt.figure(figsize=(10,8))
    # # fig.add_subplot(111, frameon=False)
    # bins = (50,100)

    # h1, _, _, plot1 = ax1.hist2d(x=a1_00b0_00['globalEff'],y=a1_00b0_00['nMessages'],bins=bins)
    # ax1.set_xlabel("Global efficiency")
    # ax1.set_ylabel("Convergence time")
    # ax1.set_title(fr"$\alpha$=1.00 & $\beta$=0.00")
    
    # h2, _, _, plot2 = ax2.hist2d(x=a1_00b0_25['globalEff'],y=a1_00b0_25['nMessages'],bins=bins)
    # ax2.set_xlabel("Global efficiency")
    # ax2.set_ylabel("Convergence time")
    # ax2.set_title(fr"$\alpha$=1.00 & $\beta$=0.25")
    
    # h3, _, _, plot3 = ax3.hist2d(x=a1_00b0_50['globalEff'],y=a1_00b0_50['nMessages'],bins=bins)
    # ax3.set_xlabel("Global efficiency")
    # ax3.set_ylabel("Convergence time")
    # ax3.set_title(fr"$\alpha$=1.00 & $\beta$=0.50")

    # h4, _, _, plot4 = ax4.hist2d(x=a0_75b0_00['globalEff'],y=a0_75b0_00['nMessages'],bins=bins)
    # ax4.set_xlabel("Global efficiency")
    # ax4.set_ylabel("Convergence time")
    # ax4.set_title(fr"$\alpha$=0.75 & $\beta$=0.00")
    
    # h5, _, _, plot5 = ax5.hist2d(x=a0_75b0_25['globalEff'],y=a0_75b0_25['nMessages'],bins=bins)
    # ax5.set_xlabel("Global efficiency")
    # ax5.set_ylabel("Convergence time")
    # ax5.set_title(fr"$\alpha$=0.75 & $\beta$=0.25")
    
    # h6, _, _, plot6 = ax6.hist2d(x=a0_75b0_50['globalEff'],y=a0_75b0_50['nMessages'],bins=bins)
    # ax6.set_xlabel("Global efficiency")
    # ax6.set_ylabel("Convergence time")
    # ax6.set_title(fr"$\alpha$=0.75 & $\beta$=0.50")

    # h7, _, _, plot7 = ax7.hist2d(x=a0_50b0_00['globalEff'],y=a0_50b0_00['nMessages'],bins=bins)
    # ax7.set_xlabel("Global efficiency")
    # ax7.set_ylabel("Convergence time")
    # ax7.set_title(fr"$\alpha$=0.50 & $\beta$=0.00")
    
    # h8, _, _, plot8 = ax8.hist2d(x=a0_50b0_25['globalEff'],y=a0_50b0_25['nMessages'],bins=bins)
    # ax8.set_xlabel("Global efficiency")
    # ax8.set_ylabel("Convergence time")
    # ax8.set_title(fr"$\alpha$=0.50 & $\beta$=0.25")

    # h9, _, _, plot9 = ax9.hist2d(x=a0_50b0_50['globalEff'],y=a0_50b0_50['nMessages'],bins=bins)
    # ax9.set_xlabel("Global efficiency")
    # ax9.set_ylabel("Convergence time")
    # ax9.set_title(fr"$\alpha$=0.50 & $\beta$=0.50")

    # # ax1.set_yscale("log")
    # # ax2.set_yscale("log")
    # # ax3.set_yscale("log")
    # # ax4.set_yscale("log")
    # # ax5.set_yscale("log")
    # # ax6.set_yscale("log")
    # # ax7.set_yscale("log")
    # # ax8.set_yscale("log")
    # # ax9.set_yscale("log")

    # plt.subplots_adjust(left=0.1,
    #                     bottom=0.1, 
    #                     right=0.9, 
    #                     top=0.9, 
    #                     wspace=0.4, 
    #                     hspace=0.4) 
    
    # # plt.show()

    # alpha1_00, (ax1,ax2,ax3) = plt.subplots(1,3)
    # # transform scatterplots to 2D histograms with identical bins taken from highest noise (alpha=1.00)
    # h3, xedges, yedges, plot3 = ax3.hist2d(x=a1_00b0_50['globalEff'],y=a1_00b0_50['nMessages'],bins=bins)
    # h1, _, _, plot1 = ax1.hist2d(x=a1_00b0_00['globalEff'],y=a1_00b0_00['nMessages'],bins=(xedges,yedges))
    # h2, _, _, plot2 = ax2.hist2d(x=a1_00b0_25['globalEff'],y=a1_00b0_25['nMessages'],bins=(xedges,yedges))

    # # compute probability density function
    # pdf1 = h1 / h1.sum()
    # pdf2 = h2 / h2.sum()
    # pdf3 = h3 / h3.sum()

    # print(hellinger(pdf1, pdf2))
    # print(hellinger(pdf2, pdf3))
    # print(hellinger(pdf1,pdf3))
    # print()

    # alpha0_75, (ax4,ax5,ax6) = plt.subplots(1,3)
    # # transform scatterplots to 2D histograms with identical bins taken from highest noise (alpha=0.75)
    # h6, xedges, yedges, plot6 = ax6.hist2d(x=a0_75b0_50['globalEff'],y=a0_75b0_50['nMessages'],bins=bins)
    # h4, _, _, plot4 = ax4.hist2d(x=a0_75b0_00['globalEff'],y=a0_75b0_00['nMessages'],bins=(xedges,yedges))
    # h5, _, _, plot5 = ax5.hist2d(x=a0_75b0_25['globalEff'],y=a0_75b0_25['nMessages'],bins=(xedges,yedges))

    # # compute probability density function
    # pdf4 = h4 / h4.sum()
    # pdf5 = h5 / h5.sum()
    # pdf6 = h6 / h6.sum()
    
    # print(hellinger(pdf4, pdf5))
    # print(hellinger(pdf5, pdf6))
    # print(hellinger(pdf4,pdf6))
    # print() 

    # alpha0_50, (ax7,ax8,ax9) = plt.subplots(1,3)
    # # transform scatterplots to 2D histograms with identical bins taken from highest noise (alpha=0.50)
    # h9, xedges, yedges, plot9 = ax9.hist2d(x=a0_50b0_50['globalEff'],y=a0_50b0_50['nMessages'],bins=bins)
    # h7, _, _, plot7 = ax7.hist2d(x=a0_50b0_00['globalEff'],y=a0_50b0_00['nMessages'],bins=(xedges,yedges))
    # h8, _, _, plot8 = ax8.hist2d(x=a0_50b0_25['globalEff'],y=a0_50b0_25['nMessages'],bins=(xedges,yedges))

    # # compute probability density function
    # pdf7 = h7 / h7.sum()
    # pdf8 = h8 / h8.sum()
    # pdf9 = h9 / h9.sum()

    # print(hellinger(pdf7, pdf8))
    # print(hellinger(pdf8, pdf9))
    # print(hellinger(pdf7,pdf9))
    # print()

    # beta0_00, (ax1,ax4,ax7) = plt.subplots(1,3)
    # # transform scatterplots to 2D histograms with identical bins taken from highest noise (beta=0.00)
    # h7, xedges, yedges, plot7 = ax7.hist2d(x=a0_50b0_00['globalEff'],y=a0_50b0_00['nMessages'],bins=bins)
    # h1, _, _, plot1 = ax1.hist2d(x=a1_00b0_00['globalEff'],y=a1_00b0_00['nMessages'],bins=(xedges,yedges))
    # h4, _, _, plot4 = ax4.hist2d(x=a0_75b0_00['globalEff'],y=a0_75b0_00['nMessages'],bins=(xedges,yedges))

    # # compute probability density function
    # pdf1 = h1 / h1.sum()
    # pdf4 = h4 / h4.sum()
    # pdf7 = h7 / h7.sum()

    # print(hellinger(pdf1, pdf4))
    # print(hellinger(pdf4, pdf7))
    # print(hellinger(pdf1,pdf7))
    # print()

    # beta0_25, (ax2,ax5,ax8) = plt.subplots(1,3)
    # # transform scatterplots to 2D histograms with identical bins taken from highest noise (beta=0.25)
    # h8, xedges, yedges, plot8 = ax8.hist2d(x=a0_50b0_25['globalEff'],y=a0_50b0_25['nMessages'],bins=bins)
    # h2, _, _, plot2 = ax2.hist2d(x=a1_00b0_25['globalEff'],y=a1_00b0_25['nMessages'],bins=(xedges,yedges))
    # h5, _, _, plot5 = ax5.hist2d(x=a0_75b0_25['globalEff'],y=a0_75b0_25['nMessages'],bins=(xedges,yedges))

    # # compute probability density function
    # pdf2 = h2 / h2.sum()
    # pdf5 = h5 / h5.sum()
    # pdf8 = h8 / h8.sum()

    # print(hellinger(pdf2, pdf5))
    # print(hellinger(pdf5, pdf8))
    # print(hellinger(pdf2,pdf8))
    # print()

    # beta0_50, (ax3,ax6,ax9) = plt.subplots(1,3)
    # # transform scatterplots to 2D histograms with identical bins taken from highest noise (beta=0.50)
    # h9, xedges, yedges, plot9 = ax9.hist2d(x=a0_50b0_50['globalEff'],y=a0_50b0_50['nMessages'],bins=bins)
    # h3, _, _, plot3 = ax3.hist2d(x=a1_00b0_50['globalEff'],y=a1_00b0_50['nMessages'],bins=(xedges,yedges))
    # h6, _, _, plot6 = ax6.hist2d(x=a0_75b0_50['globalEff'],y=a0_75b0_50['nMessages'],bins=(xedges,yedges))

    # # compute probability density function
    # pdf3 = h3 / h3.sum()
    # pdf6 = h6 / h6.sum()
    # pdf9 = h9 / h9.sum()

    # print(hellinger(pdf3, pdf6))
    # print(hellinger(pdf6, pdf9))
    # print(hellinger(pdf3,pdf9))
    # print()

    # plt.show()
    # plt.close(fig)

def check_initEffect(alpha: str, beta: str, without2: bool = True):
    """ 
    Function to account for effect of random initialization process.
    - Answer question "Can the convergence time be predicted by the initial state configuration?"

    Parameters:
    - alpha (float): Alpha noise
    - beta (float): Beta noise
    - without2 (bool): Indicator to remove graph size n=2 from data
    """

    # set paths
    settings = f"alpha{alpha}-beta{beta}"
    dataPath = f"results/{settings}"

    # load graph data
    graphData = pd.read_csv("data/data-GraphAtlas.tsv",sep='\t')

    # omit graph size 2 from data, if applicable
    if without2:
        graphData = graphData[graphData['nodes']!=2]

    fig, ax = plt.subplots(figsize=(13,8))

    cmap = {'0.5-0.6': 'tab:blue',
            '0.6-0.7': 'tab:orange',
            '0.7-0.8': 'tab:green',
            '0.8-0.9': 'tab:red',
            '0.9-1.0': 'tab:purple'}

    print(graphData)
    # scatter all repeats per graph with distinction between global efficiency bins
    for i in graphData['index'].index:
        
        ID = graphData['index'][i]

        # load convergence data
        data = pd.read_csv(f"{dataPath}/convergence-G{ID}.tsv",sep='\t')

        # construct list object
        data['meanHammingDist'] = data['meanHammingDist'].apply(lambda x : x.strip('][').split(', '))
        data['nMessages'] = data['nMessages'].apply(lambda x : float(x))

        # retrieve convergence times
        nMessages = data['nMessages']

        # retrieve initial Hamming distances per repeat
        initStates = data['meanHammingDist'].apply(lambda x : float(x[0]))

        # retrieve global efficiecy score
        GE = graphData['globalEff'][i]

        color = ''

        # set color according global efficiency bins
        if GE >= 0.9:
            color = cmap['0.9-1.0']
        elif GE >= 0.8:
            color = cmap['0.8-0.9']
        elif GE >= 0.7:
            color = cmap['0.7-0.8']
        elif GE >= 0.6:
            color = cmap['0.6-0.7']
        elif GE >= 0.5:
            color = cmap['0.5-0.6']
        
        # scatter convergence against Hamming distance with pre-defined colormap
        ax.scatter(x=initStates,y=nMessages,c=color,alpha=0.3)

    # decorate plot
    handles = [
        plt.scatter([], [], color=c, label=l)
        for c, l in zip(list(cmap.values()), list(cmap.keys()))
    ]
    ax.legend(handles=handles)
    ax.set_xlabel("Initial Hamming distance",fontsize=16)
    ax.set_ylabel("Convergence time",fontsize=16)
    ax.set_title(fr"$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')} & n$\in${{3,4,5,6,7}}",fontsize=16)
    plt.tick_params(axis="both",which="major",labelsize=16)
    
    ax.set_yscale("log")

    fig.savefig(f"images/relations/LOG-convergence-initialStates-{settings}.png",bbox_inches="tight")
    # plt.show()

    plt.close(fig)


def GE_distribution(n: int = 0, without2: bool = True):
    """
    Function to show global efficiency distribution (in histogram) of all graphs in Graph Atlas.
    - Optional to show distribution per graph size

    Parameters:
    - n (int): Indicates graph size, if preferred to cluster graphs
    - without2 (bool): Include graph size of 2 (True) or not (False) 
    """

    # load data with relevant columns
    data = pd.read_csv(f"data/data-GraphAtlas.tsv", sep='\t', usecols=['index','nodes','edges','globalEff'])
    
    # initialize figure
    fig, ax = plt.subplots()

    if n==0:
        ax.hist(x=data.globalEff,bins=20)
    else:
        data = data[data['nodes']==n]
        ax.hist(x=data.globalEff,bins=10)

    ax.set_xlabel("Global efficiency",fontsize=16)
    ax.set_ylabel("Frequency",fontsize=16)
    ax.tick_params(axis="both",which="major",labelsize=16)

    plt.show()

    if n==0:
        if without2:
            ax.set_title(fr"n$\in#{{3,4,5,6,7}}",fontsize=16)
            fig.savefig(fname=f"images/GE-distribution-ALL-without2.png",bbox_inches='tight')
        else:
            ax.set_title(fr"n$\in${{2,3,4,5,6,7}}",fontsize=16)
            fig.savefig(fname=f"images/GE-distribution-ALL.png",bbox_inches='tight')
    else:
        ax.set_title(fr"n={n}",fontsize=16)
        fig.savefig(fname=f"images/GE-distribution-n={n}.png",bbox_inches='tight')

    plt.close(fig)


def GE_pathGraphSize(N: int):
    """
    Function to show relation between path graph size and global efficiency.

    Parameters:
    - N (int): maximum number of nodes
    """

    globalEfficiency = []

    for n in range(2,N):
        G = nx.path_graph(range(n))
        globalEfficiency.append(nx.global_efficiency(G))

    fig,ax = plt.subplots()
    ax.scatter(x=range(2,N),y=globalEfficiency)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Global efficiency")
    ax.set_title("Path graphs")

    plt.show()
    fig.savefig("images/GE-pathGraphSizes.png",bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    #######################################################
    # NOTE: Set simulation settings to save appropriately #
    alpha = '1_00'
    beta = '0_00'      

    alphas = ['1_00','0_75','0_50']
    betas = ['0_00','0_25', '0_50']                                                               
    #######################################################

    # # show global efficiency measure per path graph size
    # GE_pathGraphSize(100)

    # # show raw data per metric (optional 3rd degree polynomial fit)
    # scatterALL(alpha=alpha,beta=beta,draw_polynomial=False)

    # # show mean data per metric (optional 3rd degree polynomial fit)
    # scatterMEAN(alpha=alpha,beta=beta,draw_polynomial=False)

    # # compare effect of noise using 3rd degre polynomial fit (varying only one noise parameter)
    # polynomialFit_compareSingleNoise(changing='alpha')

    # # compare effect of all noise settings using 3rd degre polynomial fit
    # polynomialFit_compareALL()

    # # show global efficiency distribution
    # GE_distribution(n=0,without2=False)
    
    # for alpha, beta in product(alphas,betas):

    #     print(f'alpha={alpha} & beta={beta}')

        # # NOTE NOTE: RUN SCRIPT USING -W "ignore" :NOTE NOTE #
        # # show probability distribution per parameter settings per metric (optionally per graph size)
        # violin_per_params(alpha=alpha,
        #                     beta=beta,
        #                     perN=False,
        #                     fit='none',
        #                     without2=True,
        #                     metric='global',
        #                     fixed=False) # NOTE: CHANGE FILENAME (@end function!)

        # # show histogram distribution per violin (per parameter settings, per metric, optionally per graph size)
        # hist_per_violin(alpha=alpha,
        #                 beta=beta,
        #                 perN=False) # NOTE: CHANGE FILENAME (@end function!)


    # for alpha in alphas:
    # # for beta in betas:
    #     # show shift in probability distribution for varying noise per metric
    #     violin_noiseEffect(fixed_param=alpha,
    #                     varying_param=betas,
    #                     variable='beta',
    #                     metric='global',
    #                     fit='none',
    #                     without2=True) # NOTE: CHANGE FOR-LOOP AND FILENAME AS DESIRED (in function!)

    quantifyNoiseDifference()

    # # summary plot of noise effect
    # summary_noiseEffect(alphas=alphas,
    #                     betas=betas,
    #                     vary='beta',
    #                     without2=True) # NOTE: CHANGE FOR-LOOPS ACCORDING TO VARYING NOISE PARAMETER

    # # show relation between convergence and initial mean Hamming distance
    # check_initEffect(alpha=alpha,
    #                  beta=beta,
    #                  without2=True) # NOTE: CHANGE FILENAME (@end function!)