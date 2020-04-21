# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 10:35:07 2020

@author: parth
"""
from pandas_datareader import DataReader
from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from math import sqrt

def GARCH(Y):
 "Initialize Params:"
 mu = param0[0]
 omega = param0[1]
 alpha = param0[2]
 beta = param0[3]
 T = Y.shape[0]
 GARCH_Dens = np.zeros(T) 
 sigma2 = np.zeros(T)   
 F = np.zeros(T)   
 v = np.zeros(T)   
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    F[t] = Y[t] - mu-np.sqrt(sigma2[t])*np.random.normal(0,1,1)
    v[t] = sigma2[t]
    GARCH_Dens[t] = (1/2)*np.log(2*np.pi)+(1/2)*np.log(v[t])+\
                    (1/2)*(F[t]/v[t])     
    Likelihood = np.sum(GARCH_Dens[1:-1])  
    return Likelihood

def GARCH_PROD(params, Y0, T):
 mu = params[0]
 omega = params[1]
 alpha = params[2]
 beta = params[3]
 Y = np.zeros(T)  
 sigma2 = np.zeros(T)
 Y[0] = Y0
 sigma2[0] = 0.0001
 for t in range(1,T):
    sigma2[t] = omega + alpha*((Y[t-1]-mu)**2)+beta*(sigma2[t-1]); 
    Y[t] = mu+np.sqrt(sigma2[t])*np.random.normal(0,1,1)    
 return Y    

PG = DataReader('PG',  'yahoo', datetime(2019,1,1), datetime(2019,12,31));
Y = PG['Close'].values;
PG = pd.DataFrame()
PG['days'] = range(1,253)
PG.head()
PG['index'] = Y

T=252
PG['log_return'] = np.log(PG['index'] / PG['index'].shift(periods=1))
mu = PG['log_return'].mean()
omega = PG['log_return'].var()
Y = PG['log_return']
param0 = np.array([mu, omega, 0.2, 0.5])
param_star = minimize(GARCH, param0, method='BFGS', options={'xtol': 1e-16, 'disp': True})
Y_GARCH = GARCH_PROD(param_star.x, Y[1], T)
timevec = np.linspace(1,T,T)
plt.figure(figsize=(20,10))
plt.plot(timevec, Y,'b',timevec, Y_GARCH,'r:')

Y_dash = np.exp(Y_GARCH) * PG['index'].shift(periods=1)
mean_error = np.mean(Y_dash - PG['index'])
rms = sqrt(mean_squared_error(PG['index'], Y_dash.fillna(0)))
