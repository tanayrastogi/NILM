#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 13:09:27 2018

@author: tanay
"""

import random 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse
import pandas as pd
from scipy.stats import multivariate_normal 
plt.style.use('ggplot')



def generate_square_signal(time, on_time, on_duration, WATT, VAR, noise, bulb_number):
    # Defining two signals for features Real and Reactive power for each
    sng = np.zeros((time.shape[0],2))
    sng_label = np.empty(time.shape[0], dtype = ('str', 5))
    
    
    # Duration of ON time
    start = int(on_time / 10)
    end = int(start + (on_duration/10))    
    
    # Real Power
    sng[start:end,0] += WATT    
    sng[start:end,0] += [noise*random.random() for _ in range(0, sng[start:end,0].shape[0])]
    
    # Reactive power
    sng[start:end,1] += VAR    
    sng[start:end,1] += [(noise/10)*random.random() for _ in range(0, sng[start:end,1].shape[0])]
    
    # Label for the signal
    sng_label[start:end] = np.array(['BULB{}'.format(bulb_number)]*sng[start:end,0].shape[0]) 

    
    return sng, sng_label
   
    
    
    
def generate_dummy_data(no_of_hours, sampling_frequency, no_of_signal, random_signal, on_time, on_duration, WATT, VAR, baseReal, baseReactive, noise, lineplot = False, scatterplot = False, agglineplot = False):
    
    # Time instances
    t = np.linspace(0, int(3600*no_of_hours), int(3600*sampling_frequency), endpoint=False)
    
    # The data will be stored in variable Sng  
    Sng = {}
    Sng_label = {}
    
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
           
    for i in range(0,no_of_signal):
        if(random_signal):
            sng = generate_random_square_signal(t, on_duration[i], WATT[i], VAR[i], baseReal[i], baseReactive[i], noise[i])
        else:
            sng, sng_label = generate_square_signal(t, on_time[i], on_duration[i], WATT[i], VAR[i], noise[i], i)
        
        # Plotting
        if(lineplot):        
            line_ax[0].plot(t, sng[:,0], label='Bulb {}'.format(i))
            line_ax[1].plot(t, sng[:,1], label='Bulb {}'.format(i))
            
            line_ax[0].set_ylabel('Real Power [Watt]')
            line_ax[0].legend()
            
            line_ax[1].set_xlabel('Time [sec]')
            line_ax[1].set_ylabel('Reactive Power [Var]')
            line_ax[1].legend()
        
        if(scatterplot):
            scatter_ax.scatter(sng[:,0], sng[:,1], s=10, label='Bulb {}'.format(i))
            scatter_ax.set_xlabel('Real Power [Watt]')
            scatter_ax.set_ylabel('Reactive Power [Var]')
            scatter_ax.legend()
        
        Sng['BULB{}'.format(i)] = sng
        Sng_label['BULB{}'.format(i)] = sng_label
    
    
    # Generate Agg data
    Agg = np.zeros((t.shape[0],2))
    Agg_Label = np.empty(t.shape, dtype = ('str', 10))
    
    for i in range(0, no_of_signal):
        Agg = np.add(Agg, Sng['BULB{}'.format(i)])
        Agg_Label = np.core.defchararray.add(Agg_Label, Sng_label['BULB{}'.format(i)])
    
    # Plotting
    if(agglineplot):
        fig, agglineax = plt.subplots(2,1, sharex=True)
        agglineax[0].plot(t, Agg[:,0])
        agglineax[0].set_ylabel('Real Power [Watt]')
        
        agglineax[1].plot(t, Agg[:,1])
        agglineax[1].set_xlabel('Time [sec]')
        agglineax[1].set_ylabel('Reactive Power [Var]')
   
    
    return Sng, Sng_label, Agg, Agg_Label






















def generate_random_square_signal(time, on_duration, WATT, VAR, baseReal, baseReactive, noise):
    # Defining two signals for features Real and Reactive power for each
    sng = np.zeros((time.shape[0],2))
    
    # Real Power
    sng[:,0] += baseReal
    # Reactive Power
    sng[:,1] += baseReactive
    
    # Duration of ON time
    on_time = random.choice(time-on_duration)
    start = np.where(time == on_time)
    if(start[0].size != 0):
        start = start[0][0]
    else:
        on_time = 50
        start = np.where(time == on_time)
        start = start[0][0]
            
    off_time = on_time + on_duration
    end = np.where(time == off_time)
    end = end[0][0]
    
    # Real Power
    sng[start:end,0] += WATT 
    sng[:,0] = sng[:,0] + [noise*random.random() for _ in range(0, time.shape[0])]
    
    # Reactive power
    sng[start:end,1] += VAR
    sng[:,1] = sng[:,1] + [(noise/50)*random.random() for _ in range(0, time.shape[0])]
    
    
    return sng
    








def gaussian_model_estimation(sng, name, plot=False):
    # Finding the best number of clusters
    n_estimator = np.arange(1, 10)
    clfs = [GaussianMixture(n_components = n, covariance_type='full', init_params='kmeans').fit(sng) for n in n_estimator]
    bics = pd.DataFrame([clf.bic(sng) for clf in clfs])
    aics = pd.DataFrame([clf.aic(sng) for clf in clfs])

    # Finding the optimum clusters
    bics_shift = bics.shift(1) - bics
    opt_cluster = int(bics_shift.idxmax()) + 1
    
    if(plot):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('GMM Model Estimation for {}'.format(name))
        ax.plot(n_estimator, bics, label = 'BIC')
        ax.plot(n_estimator, aics, label = 'AIC')
        ax.legend()
        ax.set_xlabel('Number of Clusters')
    
    return aics, bics, opt_cluster



def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw x at means
    ax.scatter(position[0], position[1], s = 60, marker = '+', c='r', zorder = 3)
    
    # Draw the Ellipse
    for nsig in range(1, 5):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height, angle, lw = 0.5, ec = 'k', fc='none'))


def plot_gmm(gmm, X, label=True, ax=None):
       
    ax = ax or plt.gca()
    
    labels = gmm.fit(X).predict(X)
    if label:
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=10, zorder=2)
    else:
        ax.scatter(X[:, 0], X[:, 1], s=10, zorder=2)
    
    
    w_factor = 0.6 / gmm.weights_.max()
    for pos, covar, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        draw_ellipse(pos, covar, ax = ax, alpha=w * w_factor)
    

    
    
    
def Gaussian_clustering(sng, name, components, plot=False):
    
    gmm = GaussianMixture(n_components = components, covariance_type='full', init_params='kmeans').fit(sng)
    print('Gaussian Cluster for {}'.format(name))
    print('Mean: ',gmm.means_)
    print('Variance: ',gmm.covariances_)
    print('Weights: ', gmm.weights_)
    print('\n')
    
    if (plot):
        fig, cluster_ax = plt.subplots() 
        plot_gmm(gmm, sng, label=True, ax=cluster_ax)  
        cluster_ax.set_ylim(-5, 15)
        cluster_ax.set_xlim(-5, 270)
        cluster_ax.set_title('Gaussian Clusters for {}'.format(name))
        cluster_ax.set_xlabel('Real Power [Watt]')
        cluster_ax.set_ylabel('Reactive Power [Var]')
        
      
    # Cluster labels
    gmm_name = np.array(np.array(['GND', name], dtype = ('str', 5)))
    
    return gmm, gmm.means_, gmm.covariances_, gmm.weights_, gmm_name
    








def merge_clusters(App, unq, merge_level,  plot = False):
    ## Merging Clusters to make Super clusters
    merge_mean = []
    merge_var = []
    merge_wght = []
    merge_name = []
    
    
    # Merge Level 2
    if(merge_level > 1):
        for sng1 in range(0, len(unq)):
            for sng2 in range(sng1+1, len(unq)):
                for i in range(0, Models[unq[sng1]]['Cluster']):
                    for j in range(0, Models[unq[sng2]]['Cluster']):
                        merge_mean.append(Models[unq[sng1]]['Mean'][i] + Models[unq[sng2]]['Mean'][j])
                        merge_var.append(Models[unq[sng1]]['Vars'][i] + Models[unq[sng2]]['Vars'][j])
                        merge_wght.append(Models[unq[sng1]]['Wght'][i] * Models[unq[sng2]]['Wght'][j])
                        merge_name.append(Models[unq[sng1]]['Name'] + 'Clt{}'.format(i) + ' + ' + Models[unq[sng2]]['Signal'] + 'Clt{}'.format(j) )     
#                        merge_name.append('{}'.format(sng1) + '{}'.format(i) + '{}'.format(sng2) +  '{}'.format(j))     
    # Merge Level 3
    if(merge_level > 2):
        for sng1 in range(0, len(unq)):
            for sng2 in range(sng1+1, len(unq)):
                for sng3 in range(sng2+1, len(unq)):
                    
                    for i in range(0, Models[unq[sng1]]['Cluster']):
                        for j in range(0, Models[unq[sng2]]['Cluster']):
                            for k in range(0, Models[unq[sng3]]['Cluster']):
                                merge_mean.append(Models[unq[sng1]]['Mean'][i] + Models[unq[sng2]]['Mean'][j] + Models[unq[sng3]]['Mean'][k])
                                merge_var.append(Models[unq[sng1]]['Vars'][i] + Models[unq[sng2]]['Vars'][j] + Models[unq[sng3]]['Vars'][k])
                                merge_wght.append(Models[unq[sng1]]['Wght'][i] * Models[unq[sng2]]['Wght'][j] * Models[unq[sng3]]['Wght'][k])
                                merge_name.append(Models[unq[sng1]]['Name'] + 'Clt{}'.format(i) +' + '+ Models[unq[sng2]]['Signal'] + 'Clt{}'.format(j) + ' + ' + Models[unq[sng3]]['Signal'] + 'Clt{}'.format(k))
#                                merge_name.append('{}'.format(sng1) + '{}'.format(i) + '{}'.format(sng2) +  '{}'.format(j) + '{}'.format(sng3) +  '{}'.format(k))
    
    
    # Merge Level 4
    if(merge_level > 3):
        for sng1 in range(0, len(unq)):
            for sng2 in range(sng1+1, len(unq)):
                for sng3 in range(sng2+1, len(unq)):
                    for sng4 in range(sng3+1, len(unq)):
                        
                        for i in range(0, Models[unq[sng1]]['Cluster']):
                            for j in range(0, Models[unq[sng2]]['Cluster']):
                                for k in range(0, Models[unq[sng3]]['Cluster']):
                                    for l in range(0, Models[unq[sng4]]['Cluster']):
                                        merge_mean.append(Models[unq[sng1]]['Mean'][i] + Models[unq[sng2]]['Mean'][j] + Models[unq[sng3]]['Mean'][k] + Models[unq[sng4]]['Mean'][l])
                                        merge_var.append(Models[unq[sng1]]['Vars'][i] + Models[unq[sng2]]['Vars'][j] + Models[unq[sng3]]['Vars'][k] + Models[unq[sng4]]['Vars'][l])
                                        merge_wght.append(Models[unq[sng1]]['Wght'][i] * Models[unq[sng2]]['Wght'][j] * Models[unq[sng3]]['Wght'][k] * Models[unq[sng4]]['Wght'][l])
                                        merge_name.append(Models[unq[sng1]]['Name'] + 'Clt{}'.format(i) + ' + ' + Models[unq[sng2]]['Signal'] + 'Clt{}'.format(j) +' + '+ Models[unq[sng3]]['Signal'] + 'Clt{}'.format(k) +' + '+ Models[unq[sng4]]['Signal'] + 'Clt{}'.format(l))
#                                        merge_name.append('{}'.format(sng1) + '{}'.format(i) + '{}'.format(sng2) +  '{}'.format(j) + '{}'.format(sng3) +  '{}'.format(k) + '{}'.format(sng4) +  '{}'.format(l) )
    
    
    
    
    
    
    
    # Convert to Numpy array just for plotting
    merge_mean = np.array(merge_mean)
    merge_var = np.array(merge_var)
    merge_wght = np.array(merge_wght) 
    merge_name = np.array(merge_name, dtype=np.unicode_)
    
    number_of_cluster = len(merge_mean)
    
    if(plot):        
        ## Plotting the super clusters
        fig, ax = plt.subplots() 
        w_factor = 0.6 / merge_wght.max()
        for pos, covar, w in zip(merge_mean, merge_var, merge_wght):
            draw_ellipse(pos, covar, ax = ax, alpha=w * w_factor)
        ax.set_title('Number of Super Clusters: {}'.format(number_of_cluster)) 
        ax.set_xlabel('Real Power [Watt]')
        ax.set_ylabel('Reactive Power [Var]')
        ax.set_xlim(-20, 500)
        ax.set_ylim(0, 20)
    
    
    # Pass the merged clusters as a dictonary
    merged_Model = {}
    for i in range(0, len(merge_mean)):
        merged_Model[i] = {'Mean': merge_mean[i], 'Vars': merge_var[i], 'Wght' : merge_wght[i] , 'Signal' : merge_name[i]}
    
    return merged_Model





def kl_divergence(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    
    pv = np.diag(pv)
    qv = np.diag(qv)
    
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()
    dqv = qv.prod(axis)
    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)               # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm))) # - N


#        # KL Divergence / My implementation
#    mean1 = merge_mean[0]
#    var1 = merge_var[0]
#    mean2 = merge_mean[1]
#    var2 = merge_var[1]
#    
#    
#    inv_var2 =np.linalg.inv(var2)
#    
#    t1 = np.log((np.linalg.det(var2))/ (np.linalg.det(var1)))
#    
#    mean_sub1 = mean2 - mean1 
#    mean_sub2 = (mean2 - mean1).reshape((len(mean1),1)) 
#    t2 = np.dot(mean_sub1, np.dot(inv_var2, mean_sub2))
#    
#    t3 = np.trace(np.multiply(inv_var2,var1))
#    
#    ans = 0.5*(t3 + t2 - len(mean1) + t1)







def inference(Agg, Merged_Model):
    signal_identified = []
    
    for x in Agg:
         pdf = []
         for i in Merged_Model:
             m = Merged_Model[i]['Mean']
             c = Merged_Model[i]['Vars']
             pdf.append(multivariate_normal.pdf(x, mean=m, cov=c))

         signal_identified.append(Merged_Model[pdf.index(max(pdf))]['Signal'])
     
    return signal_identified



# ------------------------------- Generate Data -----------------------------------#

# Generating a pulse signal
no_of_hours = 1
sampling_frequency = 0.1

no_of_signal = 3
random_signal = False
on_duration = [1500, 1000, 1000] # in sec multiple of 10 only
on_time = [1000, 1800, 1200] # Values matter only random_signal = False
WATT = [200, 100, 50] # Amplitude
VAR =  [2, 5, 3] # Amplitude
baseReal = [0, 0, 0] 
baseReactive = [0, 0, 0] 
noise = [40, 40, 40] # in Watts

lineplot = True
scatterplot = True
agglineplot = True
Data, Data_label, Agg, Agg_Label = generate_dummy_data(no_of_hours, sampling_frequency, no_of_signal, random_signal, on_time, on_duration, WATT, VAR, baseReal, baseReactive, noise, lineplot = lineplot, scatterplot = scatterplot, agglineplot = agglineplot)




# Signal names
unq = list(Data.keys())
unq.sort()

# ------------------------------- Create clusters -----------------------------------#
# Mean, Variance and Weights for all the clusters
Models = {}
for signal in unq:  
    # Running Gaussian Mixutre Model for each appliance
    _, _, opt_cluster = gaussian_model_estimation(Data[signal], signal, plot=True)
    GMM, Mean, Vars, Wght, Name = Gaussian_clustering(Data[signal], signal, opt_cluster, plot=True)
  
    Models[signal] ={'Mean': np.sort(Mean, axis=0), 'Vars': np.sort(Vars, axis=0) , 'Wght': Wght, 'Cluster': opt_cluster, 'Labels': Name}


# Creating Super Clusters
aggregation_level = 3  # Defines how many clusters will be added together to create super cluster
Merged_Model = merge_clusters(Models, unq, aggregation_level, plot=True)

#
##KL_Distance= []
### Calculate the distance between two clusters
##for i in range(0, len(Merged_Model)):
##    for j in range(i+1, len(Merged_Model)):
##        dist = kl_divergence(Merged_Model[i]['Mean'], Merged_Model[i]['Vars'], Merged_Model[j]['Mean'], Merged_Model[j]['Vars'])
##        name = Merged_Model[i]['Signal'] + Merged_Model[j]['Signal']
##        KL_Distance.append([dist, name])
#
## ------------------------------- Inference -----------------------------------#
#signal_identified = inference(Agg, Merged_Model)
#
#
#plt.show()


