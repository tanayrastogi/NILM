#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 11:28:37 2018

@author: tanay
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# plt.style.use('ggplot')


# Plotting area
fig, ax = plt.subplots()
ax.set(xlim=(-3, 3), ylim=(-1, 1))

# Data
x = np.linspace(-3, 3, 91)
t = np.linspace(1, 25, 30)
X2, T2 = np.meshgrid(x, t) 
sinT2 = np.sin(2*np.pi*T2/T2.max())
F = 0.9*sinT2*np.sinc(X2*(1 + sinT2))
F = F[10]

# Line to be plotted
ax.plot(x, F, color='k', lw=2)

# Scatter point
point = ax.scatter(x[0], F[0])


def animate(i):  
    ax.set_title('Frame ' + str(i))
    point.set_offsets((x[i], F[i]))    

anim = FuncAnimation(fig, animate, interval=200, frames=len(x)-1)
 
plt.draw()
plt.show()


