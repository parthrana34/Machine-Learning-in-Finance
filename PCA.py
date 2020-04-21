# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:20:02 2020

@author: parth
"""

from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
from pandas_datareader import DataReader
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


JNJ = DataReader('JNJ',  'yahoo', datetime(2019,1,1), datetime(2019,12,31));
UNH = DataReader('UNH',  'yahoo', datetime(2019,1,1), datetime(2019,12,31));
ABT = DataReader('ABT',  'yahoo', datetime(2019,1,1), datetime(2019,12,31));
MRK = DataReader('MRK',  'yahoo', datetime(2019,1,1), datetime(2019,12,31));
PFE = DataReader('PFE',  'yahoo', datetime(2019,1,1), datetime(2019,12,31));

df = pd.DataFrame()
desired_value = 'Close'

df['JNJ'] = JNJ[desired_value].values
df['UNH'] = UNH[desired_value].values
df['ABT'] = ABT[desired_value].values
df['MRK'] = MRK[desired_value].values
df['PFE'] = PFE[desired_value].values


# define a matrix
A = array(df)

print(A)
# calculate the mean of each column
M = mean(A.T, axis=1)
print(M)
# center columns by subtracting column means
C = A - M
print(C)
# calculate covariance matrix of centered matrix
V = cov(C.T)
print(V)
# eigendecomposition of covariance matrix
values, vectors = eig(V)
print(vectors)
print(values)
# project data
P = vectors.T.dot(C.T)
print(P.T)

P.var(axis=1)
CompetitorIndex = P[0]

plt.plot(CompetitorIndex)

#====================================Prrof Check using readymade package====================
from sklearn.decomposition import PCA

pca = PCA(n_components=5)

principalComponents = pca.fit_transform(A)

principalDf = pd.DataFrame(data = principalComponents)

df.var().sum()
np.var(A,axis=0)
principalDf.var()