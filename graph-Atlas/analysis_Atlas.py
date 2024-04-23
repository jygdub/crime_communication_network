"""
Script analysis results of communication dynamics on networkx's Graph Atlas networks.

Written by Jade Dubbeld
18/03/2024
"""

from matplotlib import pyplot as plt
from tqdm import tqdm
from itertools import product
import numpy as np, pandas as pd, seaborn as sns


def scatterALL(alpha: str, beta: str, draw_polynomial: bool) -> None:
    """
    Function to scatter all raw datapoints.
    - X-axis: Network metric
    - Y-axis: Timesteps

    Parameters:
    - alpha (str): Sender bias/noise value
    - beta (str): Receiver bias/noise value
    - draw_polynomial (bool): True indicates fitting a 3rd degree polynomial; False indicates no polynomial fit

    Returns:
    - None
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

        ax.set_ylabel("Convergence rate (number of messages)")
        ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')}')
        # plt.show()

        # save figure with corresponding filename
        if draw_polynomial:
            fig.savefig(f"images/relations/{settings}/all-datapoints/relation-{metric}-convergence-polynomial.png",bbox_inches='tight')
        else:
            fig.savefig(f"images/relations/{settings}/all-datapoints/relation-{metric}-convergence-scatter.png",bbox_inches='tight')

        plt.close(fig)


def scatterMEAN(alpha: str, beta: str, draw_polynomial: bool) -> None:
    """
    Function to scatter all mean datapoints, where the mean is taken per graph.
    - X-axis: Network metric
    - Y-axis: Timesteps

    Parameters:
    - alpha (str): Sender bias/noise value
    - beta (str): Receiver bias/noise value
    - draw_polynomial (bool): True indicates fitting a 3rd degree polynomial; False indicates no polynomial fit

    Returns:
    - None
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
        ax.set_ylabel("Convergence rate (number of messages)")
        ax.set_title(fr'$\alpha$={alpha.replace('_','.')} & $\beta$={beta.replace('_','.')}')
        
        # plt.show()

        # save figure with corresponding filename
        if draw_polynomial:
            fig.savefig(f"{images_path}/relation-{metric}-convergence-mean-polynomial.png",bbox_inches='tight')
        else:
            fig.savefig(f"{images_path}/relation-{metric}-convergence-mean-scatter.png",bbox_inches='tight')
        
        plt.close(fig)


def polynomialFit_compareSingleNoise(changing: str) -> None:
    """
    Function to compare model parameter settings per changing 'alpha' or 'beta' noise using 
    the 3rd degree polynomial fit through mean data per graph size.

    Parameters:
    - changing (str): Indicator for comparing varying 'alpha' or 'beta'

    Returns:
    - None
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
        ax.set_ylabel("Convergence rate (number of messages)")

        # plt.show()

        # save figure with corresponding filename
        if changing == 'beta':
            ax.set_title(fr'$\alpha$=1.00 & n={n}')
            fig.savefig(f"images/relations/varyingBeta-alpha1_00-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')
        
        elif changing == 'alpha':
            ax.set_title(fr'$\beta$=0.00 & n={n}')
            fig.savefig(f"images/relations/varyingAlpha-beta0_00-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')
        
        plt.close(fig)

def polynomialFit_compareALL() -> None:
    """
    Function to compare ALL model parameter settings using the 3rd degree polynomial fit 
    through mean data per graph size.

    Parameters:
    - None

    Returns:
    - None
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

        ax.set_ylabel("Convergence rate (number of messages)")
        ax.set_title(f"Varying alpha and beta parameters (n={n})")

        # save figure
        fig.savefig(f"images/relations/combined-parameters-{metric}-n={n}-convergence-mean-polynomial.png",bbox_inches='tight')

        plt.close(fig)


def my_floor(a: float, precision: int = 0):
    return np.true_divide(np.floor(a * 10**precision), 10**precision)


def violin_per_params(alpha: float, beta: float, perN: bool, fit: str, without2: bool) -> None:
    """ 
    Violinplot distribution of convergence rates per metric per bin (size=0.1).

    Parameters:
    - alpha (float): Alpha noise
    - beta (float): Beta noise
    - perN (bool): False if combined data; True if per graph size
    - fit (str): Fitting linear fit ('linear') or exponential fit ('exponential) or no fit ('none')
    - without2 (bool): Indicator to remove graph size n=2 from data

    Returns:
    - None
    """
    
    # set paths
    settings = f'alpha{alpha}-beta{beta}'                       
    images_path = f'images/relations/{settings}/all-datapoints'  

    # load all data
    data = pd.read_csv(f'data/relationData-{settings}-Atlas.tsv', sep='\t')
    
    # eliminate graph size n=2, if desired
    if without2:
        data = data.drop(range(0,100))

    # set iterative for for-loop
    iterative = None
    if perN:
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
                    label=f'{round(coef_lin[0],2)} x + {round(coef_lin[1],2)}')
        
        elif fit == 'exponential':

            # exponential polyfit
            coef_exp = np.polyfit(Xaxis, np.log(max_density), 1)
            poly1d_fn_exp = np.poly1d(coef_exp) 

            # plot maximum probability density
            ax.plot(Xaxis, max_density, 'ro', label='Maximum probability density')
            ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), 
                    np.exp(poly1d_fn_exp(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05))), 
                    'k--', 
                    label=f'ln(C)={round(coef_exp[0],2)} * GE + {round(coef_exp[1],2)}')

        # decorate figure
        plt.legend(bbox_to_anchor=(1,1),fontsize=14)

        if efficiency:
            ax.set_xlabel(f"{metric.capitalize()} efficiency",fontsize=16)
        elif coefficient:
            ax.set_xlabel(f"{metric.capitalize()} coefficient",fontsize=16)
        elif metric == 'CFbetweenness':
            ax.set_xlabel(f"Current flow betweenness centrality",fontsize=16)
        else:
            ax.set_xlabel(f"{metric.capitalize()} centrality",fontsize=16)

        ax.set_ylabel("Convergence rate (number of messages)",fontsize=16)

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
        fig.savefig(f"{images_path}/expFit-LOGdistribution-withoutN=2-convergence-per-{metric}-violin.png",bbox_inches='tight')
        plt.close(fig)


def hist_per_violin(alpha: float, beta: float, perN: bool) -> None:
    """ 
    Histogram distribution of convergence rates per metric per bin/violin (size=0.1).

    Parameters:
    - alpha (float): Alpha noise
    - beta (float): Beta noise
    - perN (bool): False if combined data; True if per graph size

    Returns:
    - None
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

                ax.set_ylabel("Convergence rate (number of messages)",fontsize=16)

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


def violin_noiseEffect(fixed_param: str, varying_param: list, variable: str, metric: str, fit: str, without2: bool) -> None:
    """ 
    Violinplot distribution of convergence rates per metric per bin (size=0.1).

    Parameters:
    - fixed_param (str): value of fixed parameter (alpha or beta)
    - varying_param (list): list of values of varying paramter (alpha or beta)
    - variable (str): 'alpha' if alpha varying noise parameter; 'beta' if beta varying noise parameter
    - metric (str): network metric
    - fit (str): Fitting linear fit ('linear') or exponential fit ('exponential) or no fit ('none')
    - without2 (bool): Indicator to remove graph size n=2 from data

    Returns:
    - None
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

        ########################################
        # NOTE: SET CORRECT FOR-LOOP
        for i, alpha in enumerate(alphas):
        # for i, beta in enumerate(betas):
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

            ax.violinplot(subset['nMessages'],positions=[np.linspace(0.,4.,5)[i+1]])

        labels = []

        # decorate for varying alphas, else varying betas
        if variable == 'alpha':

            labels = [fr"$\alpha$=1.00", fr"$\alpha$=0.75", fr"$\alpha$=0.50"]

            if efficiency:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} for all n's | {metric.capitalize()} efficiency {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            elif coefficient:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} for all n's | {metric.capitalize()} coefficient {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            elif metric == 'CFbetweenness':
                ax.set_title(fr"$\beta$={beta.replace('_','.')} for all n's | Current flow betweenness centrality {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            else:
                ax.set_title(fr"$\beta$={beta.replace('_','.')} for all n's | {metric.capitalize()} centrality {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
        else:

            labels = [fr"$\beta$=0.00", fr"$\beta$=0.25", fr"$\beta$=0.50"]        

            if efficiency:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} for all n's | {metric.capitalize()} efficiency {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            elif coefficient:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} for all n's | {metric.capitalize()} coefficient {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            elif metric == 'CFbetweenness':
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} for all n's | Current flow betweenness centrality {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)
            else:
                ax.set_title(fr"$\alpha$={alpha.replace('_','.')} for all n's | {metric.capitalize()} centrality {round(TH,2)}-{round(TH+0.1,2)}",fontsize=16)

        # set x-axis
        Xaxis = np.arange(1, len(labels) + 1)

        if fit == 'linear':

            # fit linear
            coef = np.polyfit(Xaxis, max_density, 1)
            poly1d_fn = np.poly1d(coef) 

            # plot maximum probability density
            ax.plot(Xaxis, max_density, 'ro', label='Maximum probability density')
            ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), poly1d_fn(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05)), 'g--', label=f'{round(coef[0],2)} x + {round(coef[1],2)}')

        elif fit == 'exponential':

            # fit exponential
            coef_exp = np.polyfit(Xaxis, np.log(max_density), 1)
            poly1d_fn_exp = np.poly1d(coef_exp) 

            # plot maximum probability density
            ax.plot(Xaxis, max_density, 'ro', label='Maximum probability density')
            ax.plot(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05), np.exp(poly1d_fn_exp(np.linspace(Xaxis[0]-0.05,Xaxis[-1]+0.05))), 'k--', label=f'ln(y)={round(coef_exp[0],2)} x + {round(coef_exp[1],2)}')

        # decorate figure
        plt.legend(bbox_to_anchor=(1,1),fontsize=14)

        ax.set_xticks(Xaxis, labels=labels)
        ax.set_xlim(0., len(labels) + 1.)

        ax.set_ylabel("Convergence rate (number of messages)",fontsize=16)
        ax.set_yscale("log")

        plt.tick_params(axis="both",which="major",labelsize=16)

        plt.show()

        #######################################################
        # NOTE: add LOG for regular log scale; add linePlot for linear fit; change allN to withoutN=2 if applied
        if variable == 'alpha':
            # fig.savefig(f"{images_path}/linePlot-noiseEffect-varyingAlpha-allN-{metric}-violin{counter}.png",bbox_inches='tight')
            fig.savefig(f"{images_path}/expPlot-noiseEffect-varyingAlpha-withoutN=2-{metric}-violin{counter}.png",bbox_inches='tight')

        else:
            # fig.savefig(f"{images_path}/linePlot-noiseEffect-varyingBeta-allN-{metric}-violin{counter}.png",bbox_inches='tight')
            fig.savefig(f"{images_path}/expPlot-noiseEffect-varyingBeta-withoutN=2-{metric}-violin{counter}.png",bbox_inches='tight')
        #######################################################

        plt.close(fig)


if __name__ == "__main__":
    #######################################################
    # NOTE: Set simulation settings to save appropriately #
    alpha = '1_00'
    beta = '0_00'      

    alphas = ['1_00','0_75','0_50']
    betas = ['0_00','0_25', '0_50']                                                               
    #######################################################

    # scatterALL(alpha=alpha,beta=beta,draw_polynomial=False)
    # scatterMEAN(alpha=alpha,beta=beta,draw_polynomial=False)
    # polynomialFit_compareSingleNoise(changing='alpha')
    # polynomialFit_compareALL()

    # NOTE NOTE: RUN SCRIPT USING -W "ignore" :NOTE NOTE #
    
    for alpha, beta in product(alphas,betas):
        if beta == '0_50' and alpha in ['0_75','0_50']:
            continue
        print(f'alpha={alpha} & beta={beta}')
        violin_per_params(alpha=alpha,beta=beta) # NOTE: CHANGE FILENAME (@end function!)

    # hist_per_violin(alphas=alphas,betas=betas) # NOTE: CHANGE FILENAME (@end function!)

    # violin_noiseEffect(fixed_param=beta,varying_param=alphas,A=True,metric='global') # NOTE: CHANGE FOR-LOOP AND FILENAME AS DESIRED (in function!)