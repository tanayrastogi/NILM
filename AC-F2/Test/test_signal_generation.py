#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 11:28:34 2018

@author: tanay
"""

import random 
import numpy as np
import matplotlib.pyplot as plt 
plt.style.use('ggplot')



def generate_random_square_signal(time, on_duration, WATT, VAR, baseReal, baseReactive, noise):
    # Defining two signals for features Real and Reactive power for each
    sng = np.zeros((time.shape[0],2))
    
    # Real Power
    sng[:,0] += baseReal
    # Reactive Power
    sng[:,1] += baseReactive
    
    # Duration of ON time
    start = int(random.choice(range(0, len(time)-int(on_duration/10))))
    end = int((on_duration/10) + start)
    
   
    # Real Power
    sng[start:end,0] += WATT 
    sng[:,0] = sng[:,0] + [(WATT*(noise/100))*random.random() for _ in range(0, time.shape[0])]
    
    # Reactive power
    sng[start:end,1] += VAR
    sng[:,1] = sng[:,1] + [(VAR*(noise/100))*random.random() for _ in range(0, time.shape[0])]
    
    
    return sng
    

    
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
    sng[start:end,1] += WATT    
    sng[start:end,1] += [noise*random.random() for _ in range(0, sng[start:end,1].shape[0])]
    
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
