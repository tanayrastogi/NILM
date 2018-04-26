#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 11:35:53 2018

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
from sklearn.metrics import confusion_matrix
import itertools


def bulb(Amplitude_real_power = 200, Amplitude_reac_power = 5, lineplot = False, scatterplot = False):
    
    # Parameters
    noise = 100
    Signal_Length = 150
    
    
    # Define 
    bulb = pd.DataFrame(np.zeros((Signal_Length,2)), columns=['power', 'reacPower'])
    bulb['Label'] = pd.Series(['bulb']*bulb.shape[0])
    bulb['Label'] = bulb['Label'].astype('str')
    
    
    # Watt and VAR of Bulb
    bulb['power'] += Amplitude_real_power
    bulb['reacPower'] += Amplitude_reac_power
    
    # Add noise
    bulb.loc[:,'power'] += [noise*random.random() for _ in range(0, len(bulb))]
    bulb.loc[:,'reacPower'] += [(1/10)*noise*random.random() for _ in range(0, len(bulb))]
     
    
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(bulb.index.values, bulb['power'])
        line_ax[1].plot(bulb.index.values, bulb['reacPower'])
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()
        
    
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(bulb['power'], bulb['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.legend()


    return bulb


def fsm(lineplot = False, scatterplot = False):
    
    # Parameters for the fsm
    noise = 10
    Amplitude_real_power = 800
    Amplitude_reac_power = 80
    periodic_signal = [0,10,10,0]*5
    
    # Define the fsm signal
    fsm = pd.DataFrame(np.zeros((150,2)), columns=['power', 'reacPower'])
    fsm['Label'] = pd.Series(['fsm']*fsm.shape[0])
    fsm['Label'] = fsm['Label'].astype('str')
    
    
    # Real Power
    fsm['power'] += Amplitude_real_power
    fsm.loc[0:len(periodic_signal) - 1, 'power'] = periodic_signal
    fsm.loc[0:len(periodic_signal) - 1, 'power'] *= 200 
    fsm.loc[60:60+len(periodic_signal) - 1, 'power'] = periodic_signal
    fsm.loc[60:60+len(periodic_signal) - 1, 'power'] *= 200 
    fsm.loc[120:120+len(periodic_signal) - 1, 'power'] = periodic_signal
    fsm.loc[120:120+len(periodic_signal) - 1, 'power'] *= 200 
 
    # Reactive Power
    fsm['reacPower'] += Amplitude_reac_power
    fsm.loc[0:len(periodic_signal) - 1, 'reacPower'] += 20 
    fsm.loc[60:60+len(periodic_signal) - 1, 'reacPower'] += 20
    fsm.loc[120:120+len(periodic_signal) - 1, 'reacPower'] += 20
    
    
    fsm.loc[:,'power'] += [20*noise*random.random() for _ in range(0, len(fsm))]
    fsm.loc[:,'reacPower'] += [0.9*noise*random.random() for _ in range(0, len(fsm))]
    
    
  
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(fsm.index.values, fsm['power'])
        line_ax[1].plot(fsm.index.values, fsm['reacPower'])
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
                

        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()

    
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(fsm['power'], fsm['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.legend()
        scatter_ax.set_xlim(0, 2500)
        scatter_ax.set_ylim(0, 150)
    return fsm






def tsm(lineplot = False, scatterplot = False):
    
    noise = 10
    
    tsm = pd.DataFrame(np.zeros((150,2)), columns=['power', 'reacPower'])
    tsm['Label'] = pd.Series(['tsm']*tsm.shape[0])
    tsm['Label'] = tsm['Label'].astype('str')
    
    tsm.loc[0:40, 'power'] += 500
    tsm.loc[40:150, 'power'] += 1000
    
    tsm.loc[0:40, 'reacPower'] += 100
    tsm.loc[40:150, 'reacPower'] += 50 

    
    tsm.loc[:,'power'] += [20*noise*random.random() for _ in range(0, len(tsm))]
    tsm.loc[:,'reacPower'] += [0.9*noise*random.random() for _ in range(0, len(tsm))]
    
    
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(tsm.index.values, tsm['power'])
        line_ax[1].plot(tsm.index.values, tsm['reacPower'])
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()

        
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(tsm['power'], tsm['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.legend()

    
    return tsm





def single_data(unq, Data):
    rand = random.choice(unq)
    print(rand)
    return Data[rand].as_matrix()


def double_data(unq, Data):
    rand = random.sample(unq,2)
    print(rand)
    return Data[rand[0]].add(Data[rand[1]], axis='index', fill_value=True).as_matrix()
    
def triple_data(unq, Data):
    rand = random.sample(unq,3)
    print(rand)
    temp = Data[rand[0]].add(Data[rand[1]], axis='index', fill_value=True)
    return Data[rand[2]].add(temp, axis='index', fill_value=True).as_matrix()



def aggregated_signal(bulb, fsm, tsm, lineplot = False, scatterplot = False, ind_lineplot = False, ind_scatterplot = False):
    
    # Make a dictonaru for the signals
    Sng = {}
    Sng['bulb'] = bulb
    Sng['fsm'] = fsm
    Sng['tsm'] = tsm
    
    unq = list(Sng.keys())
    unq.sort()

    
    
    
    # Aggregated signal
    agg = pd.DataFrame(np.zeros((1500,2)), columns=['power', 'reacPower'])
    agg['Label'] = pd.Series(['GND']*agg.shape[0])
    agg['Label'] = agg['Label'].astype('str')
    
    # Interval 1: Only Bulb
    agg.loc[0:len(bulb)-1, :] = single_data(unq, Sng)
    
    # Interval 2: Only fms
    agg.loc[200: 200 +len(fsm)-1, :]  = fsm.as_matrix()
    
    # Interval 3: Only tms
    agg.loc[450: 450 +len(tsm)-1, :]  = tsm.as_matrix()
    
    # Interval 4: Bulb + fsm
    agg.loc[650: 650 +len(fsm)-1, :]  = bulb.add(fsm, axis='index', fill_value=True).as_matrix()
    
    # Interval 5: Bulb + tsm
    agg.loc[850: 850 +len(bulb)-1, :]  = bulb.add(tsm, axis='index', fill_value=True).as_matrix()
    
    # Interval 6: fsm + tsm
    agg.loc[1050: 1050 +len(fsm)-1, :]  = fsm.add(tsm, axis='index', fill_value=True).as_matrix()
    
    # Interval 7: bulb + fsm + tsm
    temp = fsm.add(tsm, axis='index', fill_value=True)
    agg.loc[1250: 1250 +len(fsm)-1, :]  = bulb.add(temp, axis='index', fill_value=True).as_matrix()
    
    
    
    
    if(lineplot):
        _, line_ax = plt.subplots(2,1, sharex=True)        
        line_ax[0].plot(agg.index.values, agg['power'])
        line_ax[1].plot(agg.index.values, agg['reacPower'])
        
        line_ax[0].set_ylabel('Real Power [Watt]')
        line_ax[0].legend()
        
        line_ax[1].set_xlabel('Time [sec]')
        line_ax[1].set_ylabel('Reactive Power [Var]')
        line_ax[1].legend()
        
        line_ax[0].set_title('Plot for aggregated signal')
    
    
    # Convert the Dtype to category
    agg['Label'] = agg['Label'].astype('category')
    
    if(scatterplot):
        _,  scatter_ax = plt.subplots()
        scatter_ax.scatter(agg['power'], agg['reacPower'], s=10)
        scatter_ax.set_xlabel('Real Power [Watt]')
        scatter_ax.set_ylabel('Reactive Power [Var]')
        scatter_ax.legend()
        scatter_ax.set_title('Plot for Aggregated Signal')
        
        
    if(ind_lineplot):
        _, line_Ax = plt.subplots(2,1, sharex=True) 
    if(ind_scatterplot):
        _,  scatter_Ax = plt.subplots()
    for sng in [bulb, fsm, tsm]:
        # Plotting
        if(ind_lineplot):
                   
            line_Ax[0].plot(sng.index.values, sng['power'])
            line_Ax[1].plot(sng.index.values, sng['reacPower'])
            
            line_Ax[0].set_ylabel('Real Power [Watt]')
            
            line_Ax[1].set_xlabel('Time [sec]')
            line_Ax[1].set_ylabel('Reactive Power [Var]')
            line_Ax[1].legend()
        
        if(ind_scatterplot):
            scatter_Ax.scatter(sng['power'], sng['reacPower'], s=10)
            scatter_Ax.set_xlabel('Real Power [Watt]')
            scatter_Ax.set_ylabel('Reactive Power [Var]')
            
            
            
    # Make a dictonaru for the signals
    Sng = {}
    Sng['bulb'] = bulb
    Sng['fsm'] = fsm
    Sng['tsm'] = tsm
    
    
    
    return Sng, agg