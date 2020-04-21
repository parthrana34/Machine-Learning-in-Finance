# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 10:56:09 2018

Brownian Motion

@author: Yizhen Zhao
"""
import numpy as np
import matplotlib.pyplot as plt

T = 100
N = 10 
B = np.zeros((T,N))
for n in range(1,N):
    u = np.random.normal(0,1,T)
    B[:,n] = np.cumsum(u)

timevec = np.linspace(1,T,T)
plt.plot(timevec, B)
