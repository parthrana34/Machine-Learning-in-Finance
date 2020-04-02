# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:09:54 2020

@author: yizhen
"""
import numpy as np
import pandas as pd
#import pyflux as pf
from scipy.optimize import minimize
from pandas_datareader import DataReader
from datetime import datetime
from sklearn.neighbors import KernelDensity
import matplotlib as plt
import seaborn as sns


PG = DataReader('PG',  'yahoo', datetime(2019,1,1), datetime(2019,12,31));
NFLX = DataReader('NFLX',  'yahoo', datetime(2019,1,1), datetime(2019,12,31));


X = NFLX['Close'];
Y = PG['Close'].values;

PG = pd.DataFrame()
PG['days'] = range(1,253)
PG.head()
PG['index'] = Y


kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(PG)

abc = kde.score_samples(PG)

sns.distplot(PG['index'], hist = True, kde = True,
             kde_kws = {'linewidth': 2})